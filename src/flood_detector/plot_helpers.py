from typing import Tuple, Dict, List
from loguru import logger as logging
from pathlib import Path
import numpy as np

import os
from matplotlib import pyplot as plt

plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')
MATPLOTLIB_FONT_DIR = os.path.join(
    os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")


def show_image(image: np.array, mask: np.array, pred_image: np.array = None, artifact_path: Path = None):
    if pred_image == None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title('Image')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')
    elif pred_image != None:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.set_title('Image')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')
        ax3.set_title('Model Output')
        ax3.imshow(pred_image.permute(1, 2, 0).squeeze(), cmap='gray')

    if (artifact_path):
        fig_path = artifact_path/"imgs"
        fig_path.mkdir(parents=False, exist_ok=True)
        test_file = fig_path/f"test_result.png"
        plt.savefig(test_file)
        logging.debug(f"Test result has been saved in {test_file}")


def plot_metrics(metrics: np.array, artifact_path: Path = None) -> Tuple[float, float]:
    import seaborn as sns
    mean = np.mean(metrics)
    std = np.std(metrics)
    ax = sns.histplot(metrics)
    ax.set(xlabel='similarity metric', ylabel='number of images',
           title=f"Segmentation Perfomance | mean: {mean:.3f} / std: {std:.3f}")

    if (artifact_path):
        fig_path = artifact_path/"perf"
        fig_path.mkdir(parents=False, exist_ok=True)
        test_file = fig_path/f"perf_result.png"
        plt.savefig(test_file)
        logging.debug(f"Test result has been saved in {test_file}")
    return mean, std


def plot_results(results: List, artifact_path: Path, save_to_gif: bool = True):
    results_path = artifact_path/"results/imgs"
    results_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        image, mask, pred_mask = result
        show_image(image, mask, pred_mask)
        plt.savefig(results_path/f'{i}.png')

    if (save_to_gif):
        save_gif(artifact_path/"results")


def save_gif(results_path: Path):
    from PIL import Image
    frames = []
    images_path = results_path/'imgs'
    imgs = images_path.glob("*.png")

    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(results_path/'valid_results.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=800, loop=0)
