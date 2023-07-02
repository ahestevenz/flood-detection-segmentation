"""Command line script to upload a whole directory tree."""
from __future__ import print_function
from bnFloodDetector import utils
from bnFloodDetector import perf_helpers
from bnFloodDetector import plot_helpers
from bnFloodDetector import FloodDetector
from bnFloodDetector import FloodDatasetManagement

from builtins import input
import argparse
import cProfile as profile
import sys
from pathlib import Path
from loguru import logger as logging
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# local

__author__ = ["Ariel Hernandez <ahestevenz@bleiben.ar>"]
__copyright__ = "Copyright 2023 Bleiben."
__license__ = """General Public License"""


def _main(args):
    """Actual program (without command line parsing). This is so we can profile.
    Parameters
    ----------
    args: namespace object as returned by ArgumentParser.parse_args()
    """

    # 1. Load configuration
    logging.info(f"Loading configuration file from {args['json_file']}")
    if not Path(args['json_file']).exists():
        logging.error(
            f'{args["json_file"]} does not exist. Please check config.json path and try again')
        return -1
    conf = utils.load_conf(args['json_file'])
    experiment_path = Path(
        conf['main']['artefacts']+f'/run_{str(conf["test"]["run"])}')
    if not experiment_path.exists():
        logging.error(
            f'{experiment_path} does not exist. Please check config.json file')
        return -1
    logging.info(f'Experiment directory: {experiment_path}')

    # 2. Load model
    exp_conf = utils.load_conf(
        experiment_path/Path("config.json"), from_string=True)
    model_name = f"flood_detector_{exp_conf['model']['encoder']}_ep_{exp_conf['train']['epochs']}_bs_{exp_conf['train']['batch_size']}"
    model_name = f'{model_name}_augmented_data.pt' if exp_conf[
        'data']['need_augmentation'] else f'{model_name}.pt'
    logging.info(f'Loading model: {model_name}')

    logging.info(f'Building model...')
    device = torch.device(conf['main']['device'])
    model = FloodDetector.FloodDetector(exp_conf)
    model.to(device)
    model.load_state_dict(torch.load(
        experiment_path/Path(model_name), map_location=device))
    logging.debug(model)

    # 3. Testing
    dataset_mgnt = FloodDatasetManagement.FloodDatasetManagement(exp_conf)
    _, validset = dataset_mgnt.get_datasets()
    results = perf_helpers.run(
        validset, model, device)
    metrics = perf_helpers.get_perf_metrics(results)
    mean, std = plot_helpers.plot_metrics(metrics, experiment_path)
    logging.info(f'Validation dataset | mean: {mean:5f} / std: {std:5f}')
    plot_helpers.plot_results(results, experiment_path,
                              save_to_gif=conf['test']['save_gif'])

    return 0


def main():
    """CLI for upload the encripted files"""

    # Module specific
    argparser = argparse.ArgumentParser(
        description='Welcome to the Flood Detector testing script')
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
