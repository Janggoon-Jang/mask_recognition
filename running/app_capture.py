'''
init camera
init YOLO model (pt)
init SQLite (create table if not exists)

tracks = {}          # id -> {bbox, ttl, history_deque, state}
next_id = 0
frame_bucket = []    # [(masked, unmasked), ...] within the current second
current_second = now_sec()

loop:
  frame = read camera
  results = yolo(frame)  # det: [x1,y1,x2,y2, conf, cls]
  dets = filter_by_conf_and_nms(results)

  # --- Hungarian + IoU matching ---
  M = iou_matrix(tracks, dets)
  matches, unassigned_tracks, unassigned_dets = hungarian(M, iou_thresh)
  update matched tracks (update bbox, push cls to history)
  create tracks for unassigned_dets
  decrement ttl for unassigned_tracks, remove if ttl expired
  for each track: state = majority(history_window)

  masked = count(track.state == MASKED)
  unmasked = count(track.state == UNMASKED)
  frame_bucket.append((masked, unmasked))

  # --- 1초 스냅샷 집계 & DB write ---
  if now_sec() != current_second:
     avg_masked, avg_unmasked = mean(frame_bucket)
     INSERT INTO mask_counts (ts, masked_count, unmasked_count)
     frame_bucket = []
     current_second = now_sec()

  (optional) draw boxes + ids + FPS and imshow
'''


import argparse
import time
import sqlite3
from collections import deque, defaultdict
from typing import List, Tuple, Dict

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from running_utils import *

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/mask_yolo.pt")
    parser.add_argument("--db", type=str, default="data/mask_counts.db")
    parser.add_argument("--conf", type=float, default=0.35)      # detection threshold
    parser.add_argument("--iou", type=float, default=0.5)        # NMS IoU (ultralytics 내부)
    parser.add_argument("--show", action="store_true")           # show window
    # Tracker params
    parser.add_argument("--match_iou", type=float, default=0.3)  # IoU >= this to accept match
    parser.add_argument("--ttl", type=int, default=10)           # frames to keep lost tracks
    parser.add_argument("--hist", type=int, default=5)           # history window for majority vote
    parser.add_argument("--fps_limit", type=float, default=0.0)  # 0 for unlimited
    args = parser.parse_args()

    class_names = {0: "unmasked", 1: "masked"}

    # --- YOLO load (PT) ---
    model = YOLO(args.model)

    # --- DB init ---
    conn = sqlite3.connect(args.db, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mask_counts(
      ts INTEGER PRIMARY KEY,
      masked_count INTEGER NOT NULL,
      unmasked_count INTEGER NOT NULL
    )""")
    conn.commit()

    # --- Camera ---
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Warmup
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            continue
        _ = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)

    # Tracker state
    tracks = {}   # id -> dict(bbox, ttl, history deque, state)
    next_id = 0

    # 1-sec snapshot buffer
    def now_sec():
        return int(time.time())

    cur_sec = now_sec()
    frame_bucket = []  # list of (masked_count, unmasked_count) per frame

    last_tick = time.time()
    try:
        while True:
            # FPS throttle (optional)
            if args.fps_limit > 0:
                wait = max(0.0, (1.0 / args.fps_limit) - (time.time() - last_tick))
                if wait > 0:
                    time.sleep(wait)
                last_tick = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            # --- Inference (Ultralytics returns boxes in original frame coords) ---
            res = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]

            # Convert to [x1,y1,x2,y2,score,cls] numpy
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    c = int(cls[i])
                    if c not in (0, 1):  # skip other classes if any
                        continue
                    dets.append([*xyxy[i], float(conf[i]), float(c)])
            dets = np.array(dets, dtype=np.float32) if dets else np.zeros((0, 6), dtype=np.float32)

            # --- Hungarian + IoU matching ---
            # 1) Build cost matrix(1 - IoU)
            if len(tracks) > 0 and len(dets) > 0:
                cost = iou_matrix(tracks, dets)
                row_ind, col_ind = linear_sum_assignment(cost)
                # Filter by IoU threshold
                matches = []
                assigned_tracks = set()
                assigned_dets = set()
                for r, c in zip(row_ind, col_ind):
                    t_id = list(tracks.keys())[r]
                    iou_val = 1.0 - cost[r, c]
                    if iou_val >= args.match_iou:
                        matches.append((t_id, c))
                        assigned_tracks.add(t_id)
                        assigned_dets.add(c)
                unassigned_tracks = [tid for tid in tracks.keys() if tid not in assigned_tracks]
                unassigned_dets = [i for i in range(len(dets)) if i not in assigned_dets]
            else:
                matches = []
                unassigned_tracks = list(tracks.keys())
                unassigned_dets = list(range(len(dets)))

            # 2) Update matched tracks
            for t_id, di in matches:
                tracks[t_id]["bbox"] = dets[di, :4].copy()
                tracks[t_id]["ttl"] = args.ttl
                cls_id = int(dets[di, 5])
                tracks[t_id]["history"].append(cls_id)
                # majority vote
                tracks[t_id]["state"] = majority_vote(tracks[t_id]["history"])

            # 3) Create tracks for unassigned detections
            for di in unassigned_dets:
                cls_id = int(dets[di, 5])
                tracks[next_id] = {
                    "bbox": dets[di, :4].copy(),
                    "ttl": args.ttl,
                    "history": deque([cls_id], maxlen=args.hist),
                    "state": cls_id
                }
                next_id += 1

            # 4) Decrease TTL for unassigned tracks, and remove expired
            to_remove = []
            for t_id in unassigned_tracks:
                tracks[t_id]["ttl"] -= 1
                if tracks[t_id]["ttl"] <= 0:
                    to_remove.append(t_id)
            for t_id in to_remove:
                tracks.pop(t_id, None)

            # --- Count masked/unmasked for this frame
            masked = sum(1 for tr in tracks.values() if tr.get("state", -1) == 0)
            unmasked = sum(1 for tr in tracks.values() if tr.get("state", -1) == 1)
            frame_bucket.append((masked, unmasked))

            # --- 1-second snapshot flush to DB ---
            sec = now_sec()
            if sec != cur_sec:
                if frame_bucket:
                    arr = np.array(frame_bucket, dtype=np.float32)
                    avg = np.mean(arr, axis=0)
                    masked_avg = int(round(float(avg[0])))
                    unmasked_avg = int(round(float(avg[1])))
                else:
                    masked_avg = 0
                    unmasked_avg = 0

                cur.execute(
                    "INSERT OR REPLACE INTO mask_counts (ts, masked_count, unmasked_count) VALUES (?, ?, ?)",
                    (cur_sec, masked_avg, unmasked_avg)
                )
                conn.commit()

                frame_bucket = []
                cur_sec = sec

            # --- Optional display ---
            if args.show:
                vis = frame.copy()
                draw_boxes(vis, tracks, class_names)

                # small hud
                cv2.putText(vis, f"masked: {masked}  unmasked: {unmasked}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
                cv2.imshow("capture", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    except KeyboardInterrupt:
        pass
    finally:
        # flush last second if any
        if frame_bucket:
            arr = np.array(frame_bucket, dtype=np.float32)
            avg = np.mean(arr, axis=0)
            masked_avg = int(round(float(avg[0])))
            unmasked_avg = int(round(float(avg[1])))
            cur.execute(
                "INSERT OR REPLACE INTO mask_counts (ts, masked_count, unmasked_count) VALUES (?, ?, ?)",
                (cur_sec, masked_avg, unmasked_avg)
            )
            conn.commit()

        cap.release()
        cv2.destroyAllWindows()
        conn.close()


if __name__ == "__main__":
    main()
