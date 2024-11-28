from typing import Tuple, Dict, List
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from loguru import logger as logging


class FloodDetector(nn.Module):
    def __init__(self, conf) -> None:
        super(FloodDetector, self).__init__()
        self.arc = smp.Unet(
            encoder_name=conf['model']['encoder'],
            encoder_weights=conf['model']['weights'],
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logics = self.arc(images)
        if (masks != None):
            loss1 = DiceLoss(mode='binary')(logics, masks)
            loss2 = nn.BCEWithLogitsLoss()(logics, masks)
            return logics, loss1 + loss2
        return logics
