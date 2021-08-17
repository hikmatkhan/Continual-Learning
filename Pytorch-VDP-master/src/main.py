# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini


"""
import re
import os
import time
import wandb
import torch
import argparse
import importlib
from datetime import datetime
from torchvision import transforms
from utils import print_arguments
from loaders import LoadData
import higher
from Approach.PyTorchBase import run_sanity

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Setup save
startTime = time.time()
start_date_str = time.strftime("%m_%d_%y", time.localtime(startTime))
start_time_str = time.strftime("%H_%M", time.localtime(startTime))

parser = argparse.ArgumentParser(prog="VDP")

parser.add_argument('--network', type=str, default='MNIST_FC',
                    choices=['MNIST_FC', 'FMNIST_CONV', 'MNIST_CONV', 'CIFAR10_CONV','MNIST_FC_BBB'], help='Dataset Experiment')
# WandB Arguments
parser.add_argument('--wandb', type=bool, default=True, help='Run With WandB')
parser.add_argument('--project', type=str, default='MNIST_FC', help='WandB Project')
parser.add_argument('--account', type=str, default='hikmatkhan-', help='WandB Account')
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
parser.add_argument('--batch_size', type=int, default=124, help='Batch size')
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


def main():
    args = parser.parse_args()
    if args.wandb and not args.lightning:
        wandb.init(project=args.project, entity=args.account, name='Run_' + start_date_str + '_' + start_time_str)
        wandb.config.update(args)

    print_arguments(args)

    if not args.lightning and torch.cuda.device_count() >= 1:
        args.devices = torch.device('cuda:0')
        print('Using device:', args.devices)

    print()
    print('Load data...')
    print()

    transform_in = transforms.Compose([transforms.ToTensor()])

    dset = re.split('_', args.network)[0]

    data = LoadData(dset, transform_in, args)
    train_loader, test_loader = data.get_loaders()
    # train_set, test_set = data.get_datasets()

    netmodload = importlib.import_module('Networks.' + args.network + '.VDPNet', package='Net')
    model = netmodload.Net(args)
    # print(model)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim_meta = torch.optim.Adam(model.parameters(), lr=args.lr)



    # meta_optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    #sched = torch.optim.lr_scheduler.MultiStepLR(optim, [], gamma=0.1, last_epoch=-1, verbose=False)
    swa_model = None
    swa_scheduler = None
    if args.swa_start > 0:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optim, swa_lr=args.swa_lr, anneal_epochs=args.swa_anneal)

    args.base_save = re.sub(r"/network/", '/' + args.network + '/', args.base_save)
    if not os.path.exists(args.base_save):
        os.makedirs(args.base_save)

    print("Starting this run on :")
    print(datetime.now().strftime("%Y-%m-%d %H:%M"))

    print('-' * 101)
    print('-' * 101)
    print('-' * 101)
    if args.lightning:
        from Networks.LightningNet import Lightning_Net
        from Approach.PyTorchLightning import run

        l_model = Lightning_Net(model, optim, sched, args)
        run(args, l_model, train_loader, test_loader)

    else:
        if len(re.split('_', args.network)) == 3 and re.split('_', args.network)[2] == 'BBB':
            from Approach.PyTorchBaseBBB import run
        else:
            from Approach.PyTorchBase import run
        if torch.cuda.device_count() >= 1:
            model.to(args.devices)
            if args.swa_start > 0:
                swa_model.to(args.devices)

        if args.wandb:
            wandb.watch(model)

        run_sanity(args, model, train_loader, test_loader,
                   optim, optim_meta, sched, swa_model,
                   swa_scheduler)
        # run(args, model, train_loader, test_loader,
        # optim, sched, swa_model, swa_scheduler)


    complete_base_save = os.path.split(args.base_save)[0] + '_complete/'

    os.rename(args.base_save, complete_base_save)


if __name__ == '__main__':
    main()
