#import ads3 as ads3
import ads3_pt_profile as ads3
import torch
import os

from pathlib import Path
from torch.utils import data as D

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from util_11 import create_pipeline_perspective, create_pipeline_no_perspective

torch.manual_seed(0)
INPUT_SIZE = 224
root = Path("data")
file_train = root / "train.txt"
folder_images = root / "image"
images_train = root / "image_train"
images_valid = root / "image_valid"

class DALIDataset():
    def __init__(self, path, batch_size, num_workers):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

        pipeline = create_pipeline_no_perspective(
            self.batch_size, 
            self.num_workers, 
            self.path
        )
        self.dataset = DALIGenericIterator(
            pipeline,
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.path)])

if __name__ == "__main__":
    """Initialise dataset"""
    labels = ads3.get_labels()

    log_name = "results/11.csv"

    loader_train = DALIDataset(images_train, 80, 1)
    loader_valid = DALIDataset(images_valid, 80, 1)

    ads3.run_experiment(
        loader_train, loader_valid, log_name
    )  # For profiling feel free to lower epoch count via epoch=X