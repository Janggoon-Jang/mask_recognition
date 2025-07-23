from PIL import Image, ImageOps
import torchvision.transforms as T
import os
import json
from tqdm import tqdm

class ResizeWithPadding:
    def __init__(self, target_size=640):
        self.target_size = target_size
        self.to_tensor = T.ToTensor()

    def resize_and_adjust_labels(self, image: Image.Image, labels: list):
        """
        image: PIL.Image
        labels: list of [class_id, x_center, y_center, width, height] (all normalized)
        return: resized image, adjusted label list
        """
        w, h = image.size
        max_len = max(w, h)
        diff_x = max_len - w
        diff_y = max_len - h
        xpad1 = diff_x // 2
        ypad1 = diff_y // 2

        padding = (xpad1, ypad1, diff_x - xpad1, diff_y - ypad1)
        image = ImageOps.expand(image, padding, fill=0)
        resized_img = image.resize((self.target_size, self.target_size))

        # adjust labels
        adjusted_labels = []
        for line in labels:
            cls, x, y, bw, bh = line
            x_abs = x * w + xpad1
            y_abs = y * h + ypad1
            bw_abs = bw * w
            bh_abs = bh * h

            x_new = x_abs / max_len
            y_new = y_abs / max_len
            bw_new = bw_abs / max_len
            bh_new = bh_abs / max_len

            adjusted_labels.append([cls, x_new, y_new, bw_new, bh_new])

        return resized_img, adjusted_labels

def load_args(json_path='args_preprocess.json'):
    with open(json_path, 'r') as f:
        args = json.load(f)
    args['image_size'] = int(args['image_size'])
    return args

def load_yolo_label(txt_path):
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                labels.append([int(cls), x, y, w, h])
    return labels

def save_yolo_label(txt_path, labels):
    with open(txt_path, 'w') as f:
        for lbl in labels:
            line = f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}"
            f.write(line + '\n')