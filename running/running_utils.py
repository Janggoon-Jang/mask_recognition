import numpy as np
import _scs_direct
import cv2
from collections import deque, defaultdict
from typing import List, Tuple, Dict

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two boxes [x1,y1,x2,y2] (float)
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def iou_matrix(tracks: Dict[int, dict], dets: np.ndarray) -> np.ndarray:
    """
    Build IoU cost matrix for Hungarian (we will convert to cost = 1 - IoU)
    tracks: id -> {"bbox": np.array([x1,y1,x2,y2]), ...}
    dets:   shape [N, 6] where det[:, :4] are boxes
    """
    t_ids = list(tracks.keys())
    T = len(t_ids)
    N = len(dets)
    if T == 0 or N == 0:
        return np.zeros((T, N), dtype=np.float32)

    mat = np.zeros((T, N), dtype=np.float32)
    for ti, tid in enumerate(t_ids):
        tb = tracks[tid]["bbox"]
        for di in range(N):
            db = dets[di, :4]
            mat[ti, di] = iou_xyxy(tb, db)
    # Hungarian solves min-cost; we want max IoU, so cost = 1 - IoU
    return 1.0 - mat


def majority_vote(history: deque, num_classes: int = 2) -> int:
    """
    Return the majority class id from history (ties -> last element)
    """
    if not history:
        return -1
    cnt = defaultdict(int)
    for c in history:
        cnt[int(c)] += 1
    # tie-break by latest
    best = max(cnt.items(), key=lambda x: (x[1], history.count(x[0])))
    return best[0]


def draw_boxes(frame, tracks, class_names):
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr["bbox"].astype(int)
        state = tr.get("state", -1)
        label = f"ID {tid}"
        if state in (0, 1):
            label += f" | {class_names[state]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)