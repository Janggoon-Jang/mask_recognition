import os
import json
from PIL import Image, ImageOps
from tqdm import tqdm
from preprocessor import ResizeWithPadding, load_yolo_label, save_yolo_label
from torchvision import transforms as T


with open('args_preprocess.json', 'r') as f:
    args = json.load(f)

src_dir = os.path.join(args, 'renamed')
dst_dir = os.path.join(args, 'processed')
os.makedirs(dst_dir, exist_ok=True)

jpg_paths = [path for path in os.listdir(src_dir) if path.endswith('jpg')]
preprocessing = ResizeWithPadding()

for jpg_path in jpg_paths:
    name, ext = os.path.splitext(jpg_path)
    txt_path = name + '.txt'
    try:
        image = Image.open(jpg_path)
        label = load_yolo_label(txt_path)
    except Exception as e:
        print(f'{e} for loading {jpg_path}')
        continue

    try:
        processed_img, processed_lbl = preprocessing(image, label)
    except Exception as e:
        print(f'{e} for processing {jpg_path}')
        continue

    try:
        save_yolo_label()
        # image saving code
    except Exception as e:
        print(f'{e} for saving {jpg_path}')
        continue
        
        
    


    
    



img = Image.open(jpg_path).convert("RGB")
labels = load_yolo_label(txt_path) if os.path.exists(txt_path) else []

resized_img, adjusted_labels = preprocessor.resize_and_adjust_labels(img, labels)

# 저장
resized_img.save(os.path.join(dst_dir, fname))
if adjusted_labels:
    save_yolo_label(os.path.join(dst_dir, f'{name}.txt'), adjusted_labels)
