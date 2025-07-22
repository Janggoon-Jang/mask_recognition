from PIL import Image, ImageOps
import torchvision.transforms as T
import os
import json
from tqdm import tqdm
from preprocessor import ResizeWithPadding

'''
1. resizer : jpg_path, txt_path를 받아 resized tensor, list 반환
2. preprocess : 파일을 순회하며 


'''


class ResizeWithPadding():
    def __init__(self, target_size = 640):
        self.target_size = target_size
        self.to_tensor = T.ToTensor()

    def resize_image_n_label(self, image:Image.Image):
        '''
        image : PIL Image

        returns : torch.Tensor [C x H x W] with RGB, resized to (target_size, target_size) 
        '''

        w, h = image.size
        # w와 h가 다를 때 padding
        if w != h:
            max_len =  max(w, h)
            diff_x = max_len - w
            diff_y = max_len - h

            xpad1 = diff_x // 2
            ypad1 = diff_y // 2
            xpad2 = diff_x - xpad1
            ypad2 = diff_y - ypad1

            padding = (xpad1, ypad1, xpad2, ypad2)
            image = ImageOps.expand(image, padding, fill = 0)

        image = image.resize((self.target_size, self.target_size))

        return self.to_tensor(image)
    




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




img = Image.open(jpg_path).convert("RGB")
labels = load_yolo_label(txt_path) if os.path.exists(txt_path) else []

resized_img, adjusted_labels = preprocessor.resize_and_adjust_labels(img, labels)

# 저장
resized_img.save(os.path.join(dst_dir, fname))
if adjusted_labels:
    save_yolo_label(os.path.join(dst_dir, f'{name}.txt'), adjusted_labels)
