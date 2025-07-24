from PIL import Image, ImageOps
import os
import json

def load_args(json_path='/mask_recognition/preprocess/args_preprocess.json') -> dict:
    '''
    Load the args_preprocess.json and return it as a dictionary.
    '''
    with open(json_path, 'r') as f:
        return json.load(f)
    
def get_data_paths(src_dir) -> list:
    '''
    Return a list of tuples [(jpg_path, txt_path), ...] for files in the given directory.
    '''
    data_paths = []
    jpg_paths = [jpg_path for jpg_path in os.listdir(src_dir) if jpg_path.endswith('.jpg')]
    for jpg_path in jpg_paths:
        filename, _ = os.path.splitext(jpg_path)
        txt_path = filename + '.txt'
        data_paths.append((jpg_path, txt_path))
    return data_paths

def load_yolo_label(txt_path):
    '''
    Load YOLO-format label data from a .txt file.

    Returns
        A list of tuples [(class_id, x_center, y_center, width, height), ...]
    '''
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"Malformed label line in {txt_path}: {line}")
                continue
            cls, x, y, w, h = map(float, parts)
            labels.append((int(cls), x, y, w, h))
    return labels

class Resizer():
    def __init__(self, target_size=640):
        self.target_size = target_size
        # self.to_tensor = to_tensor

    def resize_n_adjust_labels(self, image: Image.Image, labels: list): 
        '''
        Resize the input image to (target_size, target_size) with padding if necessary,
        and adjust the labels accordingly.

        Args:
            image(PIL.Image): The input image.
            labels(list of tuples): List of labels in (class_id, x_center, y_center, width, height) format.

        Returns:
            resized_img(PIL.Image): The resized and padded image.
            adjusted_labels(list of tulples): Adjusted labels after resizing.
        '''

        # Compute padding to make image square
        w, h = image.size
        max_len = max(w, h)
        diff_x = max_len - w
        diff_y = max_len - h
        xpad1 = diff_x // 2
        ypad1 = diff_y // 2

        # Apply padding and resize to target size
        padding = (xpad1, ypad1, diff_x - xpad1, diff_y - ypad1)
        image = ImageOps.expand(image, padding, fill=0)
        target_size = (self.target_size, self.target_size)
        resized_img = image.resize(target_size)

        # Adjust label coordinates to match the resized image
        adjusted_labels = []
        for line in labels:
            # Convert normalized coordinates to padded absolute values
            cls, x, y, bw, bh = line
            x_abs = x * w + xpad1
            y_abs = y * h + ypad1
            bw_abs = bw * w
            bh_abs = bh * h

            # Convert back to normalized coordinates in the resized image
            x_new = x_abs / max_len
            y_new = y_abs / max_len
            bw_new = bw_abs / max_len
            bh_new = bh_abs / max_len

            adjusted_labels.append((cls, x_new, y_new, bw_new, bh_new))

        return resized_img, adjusted_labels

def process_single_pair(jpg_path: str, txt_path: str, resizer, data_dir: str) -> tuple:
        '''        
        Process a single image and label file pair.

        Args:
            jpg_path (str): Relative path to the input image.
            txt_path (str): Relative path to the corresponding YOLO label file.
            resizer (Resizer): An instance of the Resizer class.
            data_dir (str): Root directory containing the 'renamed' subdirectory.

        Returns:
            tuple: (resized_image, adjusted_labels, filename_without_extension)
        '''
        # Build full paths to the image and label files
        filename, _ = os.path.splitext(jpg_path)
        jpg_full_path = os.path.join(data_dir, 'renamed', jpg_path)
        txt_full_path = os.path.join(data_dir, 'renamed', txt_path)
        
        image = Image.open(jpg_full_path)
        labels = load_yolo_label(txt_full_path)
        
        processed_image, processed_label = resizer.resize_n_adjust_labels(image, labels)
        return processed_image, processed_label, filename

def save_data(image: Image.Image, labels: list, filename: str, data_dir: str):
    '''
    Save the processed image and label data using the original filename.

    Args:
        image (PIL.Image): The processed image.
        labels (list of tuples): List of (class_id, x_center, y_center, width, height).
        filename (str): Filename without extension.
        data_dir (str): Root directory where the 'processed' folder will be created.
    '''

    dst_dir = os.path.join(data_dir, 'processed')
    os.makedirs(dst_dir, exist_ok=True)
    processed_jpg_path = os.path.join(dst_dir, f'{filename}.jpg')
    processed_txt_path = os.path.join(dst_dir, f'{filename}.txt')
    
    # Save the image as a JPEG file
    image.save(processed_jpg_path, format='JPEG')

    # Save labels in YOLO format
    with open(processed_txt_path, 'w') as f:
        for lbl in labels:
            cls, x, y, w, h = lbl
            line = f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            f.write(line)