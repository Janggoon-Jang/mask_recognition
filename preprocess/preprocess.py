import os
from tqdm import tqdm
import preprocessor

def main():
    args = preprocessor.load_args()
    resizer = preprocessor.Resizer(target_size=args['image_size'])
    data_paths = preprocessor.get_data_paths(os.path.join(args['data_dir'], 'renamed'))
    for jpg_path, txt_path in tqdm(data_paths):
        try:
            image, labels, filename = preprocessor.process_single_pair(jpg_path, txt_path, resizer, args['data_dir'])
        except Exception as e:
            print(f'Failed to process {jpg_path}: {e}')
            continue

        try:
            preprocessor.save_data(image, labels, filename, args['data_dir'])
        except Exception as e:
            print(f'Failed to save {jpg_path}: {e}')
            continue


if __name__ == '__main__':
    main()