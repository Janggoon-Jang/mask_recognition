import os
import json
from PIL import Image, ImageOps
from tqdm import tqdm
from preprocessor import ResizeWithPadding
from torchvision import transforms as T



img = Image.open(jpg_path).convert("RGB")
labels = load_yolo_label(txt_path) if os.path.exists(txt_path) else []

resized_img, adjusted_labels = preprocessor.resize_and_adjust_labels(img, labels)

# 저장
resized_img.save(os.path.join(dst_dir, fname))
if adjusted_labels:
    save_yolo_label(os.path.join(dst_dir, f'{name}.txt'), adjusted_labels)
