from pathlib import Path
import shutil
import random

if __name__ == "__main__":
    BASE_PATH = Path("data")
    old_data_dir = BASE_PATH / "image"
    train_data_dir = BASE_PATH / "image_train"
    valid_data_dir = BASE_PATH / "image_valid"

    train_data_dir.mkdir(exist_ok=True)
    valid_data_dir.mkdir(exist_ok=True)

    with open(BASE_PATH / "train.txt", "r") as f:
        lines = f.readlines()
        file_paths = [l.strip() for l in lines]

    for idx, fp in enumerate(file_paths):
        _, label, _, image_name = fp.split("/")
        split = random.random()
        if split < 0.7:
            label_dir = train_data_dir / label
            label_dir.mkdir(exist_ok=True)
            shutil.copy(old_data_dir / fp, label_dir / image_name)
        else:
            label_dir = valid_data_dir / label
            label_dir.mkdir(exist_ok=True)
            shutil.copy(old_data_dir / fp, label_dir / image_name)