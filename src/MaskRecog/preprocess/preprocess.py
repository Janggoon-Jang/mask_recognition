import os
from pathlib import Path
from tqdm import tqdm
from MaskRecog.preprocess import preprocessor

def main():
    args = preprocessor.load_args()
    resizer = preprocessor.Resizer(target_size=args['image_size'])
    data_dir = Path(args["data_dir_2"]) / "cleaned"
    data_paths = preprocessor.get_data_paths(data_dir)
    for jpg_path, txt_path in tqdm(data_paths):
        try:
            image, labels, image_path = preprocessor.process_single_pair(jpg_path, txt_path, resizer)
        except Exception as e:
            print(f'Failed to process {jpg_path}: {e}')
            continue

        try:
            preprocessor.save_data(image, labels, image_path, Path(args['data_dir_2']))
        except Exception as e:
            print(f'Failed to save {jpg_path}: {e}')
            continue


if __name__ == '__main__':
    main()