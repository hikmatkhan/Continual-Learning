import argparse
import importlib
import os
import time
import higher
import wandb


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Setup save
startTime = time.time()
start_date_str = time.strftime("%m_%d_%y", time.localtime(startTime))
start_time_str = time.strftime("%H_%M", time.localtime(startTime))

parser = argparse.ArgumentParser(prog="VDP")

parser.add_argument('--network', type=str, default='MNIST_CONV',
                    choices=['VDP', 'FMNIST_CONV', 'MNIST_FC',
                             'MNIST_CONV', 'CIFAR10_CONV','MNIST_FC_BBB'], help='Dataset Experiment')
# WandB Arguments
parser.add_argument('--wandb', type=bool, default=True, help='Run With WandB')
parser.add_argument('--project', type=str, default='cifar10_vdp', help='WandB Project')
parser.add_argument('--account', type=str, default='angelinic0', help='WandB Account')
# Load Model
parser.add_argument('--load_model', type=str, default='', help='Path to a previously trained model checkpoint')
parser.add_argument('--continue_train', type=bool, default=False, help='Continue to train this checkpoint ')
# Print Parameters
parser.add_argument('--lightning', type=bool, default=False, help='Run with PyTorch Lightning')
parser.add_argument('--kdes', type=bool, default=False, help='Plot KDE Plots')
parser.add_argument('--base_save', type=str, default='../models/network/' + start_date_str + '/' + start_time_str + '/',
                    help='Where to Save model, network gets replaced with the experiment you run')
# Code Parameters
parser.add_argument('--num_workers', type=int, default=6, help='Number of CPU works to process dataset')
parser.add_argument('--data_path', type=str, default='../../data/', help='Path to save dataset data')
# Training Parameters
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=120, help='Batch size')
# Loss Function Parameters
parser.add_argument('--tau', type=float, default=0.002, help='KL Weight Term')
parser.add_argument('--clamp', type=float, default=1000, help='Clamping')
parser.add_argument('--var_sup', type=float, default=0.001, help='Loss Variance Bias')
# Learning Rate Parameters
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--lr_sched', type=float, default=[100, 150], help='Learning Rate Scheduler Milestones')
parser.add_argument('--gamma', type=float, default=0.1, help='Learning Rate Gamma')
# SWA
parser.add_argument('--swa_start', type=int, default=0, help='SWA Starting Epoch')
parser.add_argument('--swa_lr', type=int, default=0.01, help='SWA Learning Rate')
parser.add_argument('--swa_anneal', type=int, default=20, help='SWA Learning Rate')
# BBB
parser.add_argument('--samples', type=int, default=10, help='Samples')
parser.add_argument('--rho', default='-3', type=float, help='Initial rho')
parser.add_argument('--sig1', default='0.0', type=float, help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sig2', default='6.0', type=float, help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi', default='0.25', type=float, help='weighting factor for prior')
# Layer Weight Initialization Parameters
parser.add_argument('--conv_input_mean_mu', type=float, default=0, help='Input Mean Weight Normal Distr. Mean')
parser.add_argument('--conv_input_mean_sigma', type=float, default=0.1, help='Input Mean Weight Normal Distr. Sigma')
parser.add_argument('--conv_input_mean_bias', type=float, default=0.0001, help='Input Mean Bias')
parser.add_argument('--conv_input_sigma_min', type=float, default=-12, help='Input Sigma Weight Uniform Distr. Min')
parser.add_argument('--conv_input_sigma_max', type=float, default=-2.2, help='Input Sigma Weight Uniform Distr. Max')
parser.add_argument('--conv_input_sigma_bias', type=float, default=0.0001, help='Input Sigma Bias')

parser.add_argument('--fc_input_mean_mu', type=float, default=0, help='Input Mean Weight Normal Distr. Mean')
parser.add_argument('--fc_input_mean_sigma', type=float, default=0.1, help='Input Mean Weight Normal Distr. Sigma')
parser.add_argument('--fc_input_mean_bias', type=float, default=0.0001, help='Input Mean Bias')
parser.add_argument('--fc_input_sigma_min', type=float, default=-12, help='Input Sigma Weight Uniform Distr. Min')
parser.add_argument('--fc_input_sigma_max', type=float, default=-2.2, help='Input Sigma Weight Uniform Distr. Max')
parser.add_argument('--fc_input_sigma_bias', type=float, default=0.0001, help='Input Sigma Bias')

parser.add_argument('--drop_out', type=float, default=0.1, help='Network Droput')

parser.add_argument('--out_shape', type=int, default=None, help='will autofill')

parser.add_argument('--model', type=str, choices=['maml'], default='maml',
                            help='Name of the Algorithm (default: maml).')

# parser.set_defaults(test=False)

args = parser.parse_args()

from inspect import getmembers, isfunction

if __name__ == '__main__':
    print("Argument:", args)
    args.model = "Hello"
    print("Argument:", args)

    if 0:
        print("True")
    else:
        print("False")

    # netmodload = importlib.import_module('Networks.' + args.network + '.VDPNet', package='Net')
    # model = netmodload.Net(args)
    # print(model)
    import torch
    A = torch.tensor([[0, 1], [1, 0]])
    B = torch.tensor([[3, 4, 5], [6, 7, 8]])
    C = torch.tensor(7)
    print("Torch_V:", torch.__version__)
    # torch.block_diag(A, B, C)



    # ResNets = getmembers(resnet, isfunction)
    # for name, func in ResNets:
    #     print(name)
    #     if "resnet" in name.lower():
    #         print(name)

