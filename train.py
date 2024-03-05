import argparse
import os
import importlib.util

import torch
from models.utils.builder import build_detector
from data.builder import build_dataset

def get_config_from_file(filename, mode):
    spec = importlib.util.spec_from_file_location(mode, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Create a dictionary from module attributes
    config_dict = {key: getattr(module, key) for key in dir(module) if not key.startswith('__')}
    return config_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--phase', default = 'reconstruction', help='the dir to save logs and models')
    parser.add_argument('--config', default = 'configs/train_reconstruction_conf.py', help='train config file path')
    parser.add_argument('--logs-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-checkpoint', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--cuda_devices', default='0', help='training gpu ids')
    

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    cfg = get_config_from_file(args.config, args.phase)
    model = build_detector(cfg.get('model'))
    dataset = build_dataset(cfg.get('data')['train'])


if __name__ == '__main__':
    main()
