"""
main.py
"""

import argparse
import yaml
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from MultiLabelIncremental import MultiLabelIncremental


# Globel Parameters
parser = argparse.ArgumentParser(description='Multi-Label Incremental Training')
parser.add_argument('--config_name', default="./configs/MSCOCO.yaml", type=str, help="Config Name")
parser.add_argument('--output_name', default="MSCOCO", type=str, help="Results Path")
parser.add_argument('--seed', default=3407, type=int)


def load_config(args, config_file):
    # Load Config
    with open(config_file) as f:
        config = yaml.safe_load(f)
        for k, v in config.items():
            if k not in args:
                setattr(args, k, v)
    return args


def random_seed(args):
    # Set Random Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark=True
    torch.backends.cudnn.benchmark = True


def main():
    args = parser.parse_args()
    config_file = args.config_name
    args = load_config(args, config_file)
    random_seed(args)

    args.output_name += f'_B{args.base_classes}C{args.task_size}' + f'_{args.num_protos}Protos'
    args.logger_dir = './logs/' + args.output_name
    args.tensorboard_dir = './tensorboard/' + args.output_name 
    args.model_save_path = './saved_models/' + args.output_name
    args.excel_path = f'{args.output_name}.xlsx'

    multi_incremental = MultiLabelIncremental(args)
    multi_incremental.train()

    del multi_incremental


if __name__ == "__main__":
    main()
