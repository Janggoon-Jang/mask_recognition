import random
import shutil
from pathlib import Path
from MaskRecog.preprocess.preprocessor import load_args
from MaskRecog.utils.utils import seed_everything


def data_split(data_dir : Path, ratios : list[tuple[float, float, float]]) -> None:
    '''
    1. ratio sum check
    2. mkdir
    3. paths
    4. random and move
    '''
    # ratios sum check
    tr, va, te = ratios
    if abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("(train + valid + test) must sum to 1.0")
    
    # mkdir
    dst_dir = data_dir / "final"

    for parents in ["images", "labels"]:
        for child in ["train", "valid", "test"]:
            new_dir = dst_dir / parents / child
            new_dir.mkdir(parents=True, exist_ok=True)

    # paths
    jpg_path_gen = (data_dir / "processed" / "images").glob("*.jpg")
    paired_paths = []
    for jpg_path in jpg_path_gen:
        txt_path = jpg_path.parent.parent / "labels" / jpg_path.with_suffix(".txt").name
        paired_paths.append((jpg_path, txt_path))

    # random and move
    for jpg_path, txt_path in paired_paths:
        random_value = random.random()
        # random_value is in (0,tr) -> train
        if random_value < tr: 
            new_jpg_path = dst_dir / "images" / "train"
            new_txt_path = dst_dir / "labels" / "train"
        # random_value is in [tr, tr+va) -> vaild 
        elif tr <= random_value < (tr + va):
            new_jpg_path = dst_dir / "images" / "valid"
            new_txt_path = dst_dir / "labels" / "valid"
        # random_value is in (tr+va, 1) -> test    
        else: 
            new_jpg_path = dst_dir / "images" / "test"
            new_txt_path = dst_dir / "labels" / "test"

        shutil.copy2(jpg_path, new_jpg_path)
        shutil.copy2(txt_path, new_txt_path)


def main():
    args = load_args()
    seed = args.get("seed", 42)
    seed_everything(seed)
    ratios = tuple(args.get("data_split_ratio"))
    data_split(Path(args["data_dir_final"]), ratios)


if __name__ == "__main__":
    main()