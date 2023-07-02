from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import cv2
import torch


class SegmentationDataset(Dataset):

    def __init__(self, conf, df, data_aug=None) -> None:
        super().__init__()
        self.df = df
        self.conf = conf
        if data_aug == "train":
            self.augmentations = self.get_train_augs()
        elif data_aug == "valid":
            self.augmentations = self.get_valid_augs()
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.images
        mask_path = row.masks
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        # (h, w,c) -> (c, h, w)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask

    def get_train_augs(self):
        return A.Compose([
            A.Resize(self.conf['data']['image_size'],
                     self.conf['data']['image_size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

    def get_valid_augs(self):
        return A.Compose([
            A.Resize(self.conf['data']['image_size'],
                     self.conf['data']['image_size'])
        ])
