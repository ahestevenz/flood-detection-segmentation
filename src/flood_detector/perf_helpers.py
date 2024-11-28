from typing import Tuple, Dict, List
from loguru import logger as logging
from torch.utils.data import Dataset
from torch import nn
import torch
import numpy as np


def run(validset: Dataset,
        model: nn.Module,
        device: str) -> List:
    results = []
    for idx in range(len(validset)):
        image, mask = validset[idx]
        # c h w -> 1, c h w
        logits_mask = model(image.to(device).unsqueeze(0))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5)*1.0
        results.append((image, mask, pred_mask.detach().cpu().squeeze(0)))

    return results


def get_perf_metrics(results: List) -> List:
    metrics = []
    for result in results:
        mask = result[1]
        pred_mask = result[2]
        diff = np.abs(mask-pred_mask).numpy()
        metric = (diff.flatten().shape[0]-diff[diff >
                  0.5].shape[0])/diff.flatten().shape[0]
        metrics.append(metric)

    return metrics
