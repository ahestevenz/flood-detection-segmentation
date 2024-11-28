"""Command line script to upload a whole directory tree."""
from __future__ import print_function

from builtins import input
import argparse
import cProfile as profile
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
from loguru import logger as logging
import numpy as np
import torch

# local
from flood_detector.utils import load_conf, save_conf
from flood_detector.plot_helpers import show_image
from flood_detector.flood_detector import FloodDetector
from flood_detector.flood_dataset_management import FloodDatasetManagement

__author__ = ["Ariel Hernandez <ahestevenz@bleiben.ar>"]
__copyright__ = "Copyright 2023 Bleiben."
__license__ = """General Public License"""


def train_function(data_loader, model, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(data_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(data_loader)


def eval_function(data_loader, model, device) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            logits, loss = model(images, masks)
            total_loss += loss.item()

    return total_loss/len(data_loader)


def _main(args):
    """Actual program (without command line parsing). This is so we can profile.
    Parameters
    ----------
    args: namespace object as returned by ArgumentParser.parse_args()
    """

    # 1. Load datasets
    logging.info(f"Loading configuration file from {args['json_file']}")
    if not Path(args['json_file']).exists():
        logging.error(
            f'{args["json_file"]} does not exist. Please check config.json path and try again')
        return -1
    conf = load_conf(args['json_file'])

    i = 0
    while Path(conf['main']['artefacts']+f'/run_{str(i)}').is_dir():
        i += 1
    experiment_path = Path(conf['main']['artefacts']+f'/run_{str(i)}')
    experiment_path.mkdir(parents=False, exist_ok=True)
    logging.info(f'Experiment directory: {experiment_path}')

    dataset_mgnt = FloodDatasetManagement(conf)
    _, validset = dataset_mgnt.get_datasets()
    trainloader, validloader = dataset_mgnt.get_dataloaders()

    # 2. Build model
    logging.info(f'Building model...')
    model = FloodDetector(conf)
    model.to(conf['main']['device'])
    logging.debug(model)

    # 3. Training
    logging.info(f'Training model...')
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['model']['lr'])
    best_valid_loss = np.Inf

    for i in range(conf['train']['epochs']):
        train_loss = train_function(
            trainloader, model, optimizer, conf['main']['device'])
        valid_loss = eval_function(validloader, model, conf['main']['device'])

        logging.info(
            f'EPOCH {i+1} Train Loss {train_loss:.3f} Valid loss {valid_loss:.3f}')
        if valid_loss < best_valid_loss:
            model_name = f"flood_detector_{conf['model']['encoder']}_ep_{conf['train']['epochs']}_bs_{conf['train']['batch_size']}"
            model_name = f'{model_name}_augmented_data.pt' if conf[
                'data']['need_augmentation'] else f'{model_name}.pt'
            torch.save(model.state_dict(), experiment_path/Path(model_name))
            best_valid_loss = valid_loss

    # 4. Testing
    logging.info(f'Testing model...')
    idx = int(np.random.rand()*len(validset))
    image, mask = validset[idx]

    # c h w -> 1, c h w
    logits_mask = model(image.to(conf['main']['device']).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5)*1.0

    show_image(
        image, mask, pred_mask.detach().cpu().squeeze(0), experiment_path)

    # 5. Final tasks
    conf['main']['experiment'] = experiment_path.as_posix()
    conf['data']['len_train_dataset'] = len(trainloader)
    conf['data']['len_valid_dataset'] = len(validloader)
    save_conf(conf, experiment_path/Path("config.json"))
    return 0


def main():
    """CLI for upload the encripted files"""

    # Module specific
    argparser = argparse.ArgumentParser(
        description='Welcome to the Flood Detector training script')
    argparser.add_argument('-j', '--json_file', help='JSON configuration (default: "%(default)s")', required=False,
                           default='/Users/ahestevenz/Desktop/tech-projects/1_code/flood-detection-segmentation/config.json')

    # Default Args
    argparser.add_argument('-v', '--verbose', help='Increase logging output  (default: INFO)'
                           '(can be specified several times)', action='count', default=0)
    argparser.add_argument('-p', '--profile', help='Run with profiling and store '
                           'output in given file', metavar='output.prof')
    args = vars(argparser.parse_args())

    _V_LEVELS = ["INFO", "DEBUG"]
    loglevel = min(len(_V_LEVELS)-1, args['verbose'])
    logging.remove()
    logging.add(sys.stdout, level=_V_LEVELS[loglevel])

    if args['profile'] is not None:
        logging.info("Start profiling")
        r = 1
        profile.runctx("r = _main(args)", globals(),
                       locals(), filename=args['profile'])
        logging.info("Done profiling")
    else:
        logging.info("Running without profiling")
        r = _main(args)
    return r


if __name__ == '__main__':
    exit(main())
