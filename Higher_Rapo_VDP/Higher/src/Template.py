import argparse
import wandb
import random
import numpy as np
import torch


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ---------------------------------------- Utility Functions  ----------------------------------------#
def get_compute_device():
    device = torch.device('cpu')
    if torch.cuda.device_count():
        device = torch.device('cuda')
    return device


def fix_seeds(seed=101):
    # No randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
    return seed


# ---------------------------------------- Utility Functions  ----------------------------------------#

def main():
    print(args)


parser = argparse.ArgumentParser('MAML with Higher')
# General
general_params = parser.add_argument_group('General')
general_params.add_argument('--data', type=str, default="./data",
                            help='Path to the folder the data is downloaded to.')
general_params.add_argument('--dataset', type=str,
                            choices=["omniglot", "mini-imagenet", "fc100",
                                     "cifarfs", "tiered-imagenet"], default='mini-imagenet',
                            help='Name of the dataset (default: omniglot).')
# Meta Learning Params
meta_params = parser.add_argument_group('Meta Learning Parameters')
meta_params.add_argument('--ways', type=int, default=5,
                         help='Number of classes per task (N in "N-way", default: 5).')
meta_params.add_argument('--shots', type=int, default=1,
                         help='Number of training example per class (k in "k-shot", default: 5).')
meta_params.add_argument('--adaptation-steps', type=int, default=1,
                         help='Number of adaptation steps on meta-train datasets.')
meta_params.add_argument('--num-tasks', type=int, default=32,
                         help='Number of tasks to sample from task distribution. (Meta batch size)')
meta_params.add_argument('--total-num-tasks', type=int, default=-1,
                         help='Total number of tasks in task distribution. Always keep it to -1.')
# meta_params.add_argument('--first-order', action='store_true',
#                          help='Use the first order approximation, do not use highers-order '
#                               'derivatives during meta-optimization.')
meta_params.add_argument('--meta-lr', type=float, default=0.001,
                         help='Learning rate for the meta-optimizer (optimization of the outer '
                              'loss). The default optimizer is Adam (default: 1e-3).')
meta_params.add_argument('--fast-lr', type=float, default=0.01,
                         help='Learning rate for the meta-optimizer (optimization of the outer '
                              'loss). The default optimizer is Adam (default: 1e-3).')
meta_params.add_argument('--meta-learn', type=int, default=1,
                         help='Set this to False only for debugging purpose to'
                              'verify that meta learning is happening')

# Model
model_params = parser.add_argument_group('Model')
model_params.add_argument('--input', type=int, default=84,
                          help='Input to dimension')
model_params.add_argument('--channels', type=int, default=3,
                          help='Input to dimension')
model_params.add_argument('--fine-tune', type=int, default=1,
                          help='Only meta learn the FC layer')

# Optimization
optim_params = parser.add_argument_group('Optimization')
optim_params.add_argument('--num-epochs', type=int, default=-1,
                          help='Number of epochs of meta-training (default: 50000).')
optim_params.add_argument('--seed', type=int, default=fix_seeds(),
                          help='Number of epochs of meta-training (default: 101).')

# Misc
misc = parser.add_argument_group('Misc')
misc.add_argument('--num-workers', type=int, default=4,
                  help='Number of workers to use for data-loading (default: 1).')
misc.add_argument('--device', type=str, default=get_compute_device(),
                  help="Compute device information")

# Visualization
viz = parser.add_argument_group('Wandb')
viz.add_argument('--wand-project', type=str, default="Github_Rapo",
                 help='Wandb project name should go here')
viz.add_argument('--username', type=str, default="hikmatkhan",
                 help='Wandb username should go here')
viz.add_argument('--wandb-log', type=int, default=1,
                 help='If True, Logs will be reported on wandb.')

args = parser.parse_args()

if __name__ == '__main__':
    main()
