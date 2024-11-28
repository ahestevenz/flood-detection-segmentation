from typing import Tuple, Dict, List

from flood_detector.segmentation_dataset import SegmentationDataset
from flood_detector.plot_helpers import show_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from loguru import logger as logging
from pathlib import Path


class FloodDatasetManagement:
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.data_dir = conf['data']['path']
        self.image_size = conf['data']['image_size']
        self.batch_size = conf['train']['batch_size']
        self.process_data()

    def process_data(self):
        images_path = Path(self.data_dir)/Path('images/all_images')
        files = images_path.glob("*.png")
        images = []
        masks = []
        for file in files:
            images.append(file)
            masks.append(Path(self.data_dir)/Path('masks/all_masks')/file.name)
        self.df = pd.DataFrame({'images': images, 'masks': masks})

    def plot_item(self):
        row = self.df.iloc[int(np.random.rand()*len(self.df))]
        image_path = row.images
        mask_path = row.masks
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE) / 255
        show_image(image, mask)
        plt.show()

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        train_df, valid_df = train_test_split(
            self.df, test_size=0.2, random_state=42)
        if self.conf["data"]["need_augmentation"]:
            trainset = SegmentationDataset(
                self.conf, train_df, "train")
            validset = SegmentationDataset(
                self.conf, valid_df, "valid")
        else:
            trainset = SegmentationDataset(
                self.conf, train_df)
            validset = SegmentationDataset(
                self.conf, valid_df)
        logging.debug(f"Dataset | Trainset size: {len(trainset)}")
        logging.debug(f"Dataset | Validset size: {len(validset)}")
        return trainset, validset

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        trainset, validset = self.get_datasets()
        trainloader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True)
        validloader = DataLoader(validset, batch_size=self.batch_size)
        logging.debug(
            f"Dataset | Total number of batches in trainloader: {len(trainloader)}")
        logging.debug(
            f"Dataset | Total no of batches in validloader: {len(validloader)}")
        image, mask = next(iter(trainloader))
        logging.debug(f'Dataset | One batch image shape: {image.shape}')
        logging.debug(f'Dataset | One batch mask shape: {mask.shape}')
        return trainloader, validloader
