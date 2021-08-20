import argparse
import importlib
import time
import higher
import wandb
from torchsummary import torchsummary

from train import run_inner_loop
from test import run_val_loop
import torch.optim as optim
import utils
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#-----------------------------VDP Params ------------------------------#

# Setup save
startTime = time.time()
start_date_str = time.strftime("%m_%d_%y", time.localtime(startTime))
start_time_str = time.strftime("%H_%M", time.localtime(startTime))

parser = argparse.ArgumentParser(prog="VDP")

parser.add_argument('--network', type=str, default='MNIST_CONV',
                    choices=['VDP', 'FMNIST_CONV', 'MNIST_FC',
                             'MNIST_CONV', 'CIFAR10_CONV','MNIST_FC_BBB'], help='Dataset Experiment')

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

#-----------------------------VDP Params ------------------------------#

# parser = argparse.ArgumentParser('MAML using highers library')

# General
general_params = parser.add_argument_group('General Params')
general_params.add_argument('--data', type=str, default="./data",
                            help='Path to the folder the data is downloaded to.')
general_params.add_argument('--dataset', type=str,
                            choices=["omniglot", "mini-imagenet", "fc100",
                                     "cifarfs", "tiered-imagenet"], default='omniglot',
                            help='Name of the dataset (default: omniglot).')
general_params.add_argument('--ways', type=int, default=5,
                            help='Number of classes per task (N in "N-way", default: 5).')
general_params.add_argument('--shots', type=int, default=5,
                            help='Number of training example per class (k in "k-shot", default: 5).')
general_params.add_argument('--adaptation-steps', type=int, default=1,
                            help='Number of adaptation steps on meta-train datasets.')
# general_params.add_argument('--num-shots-test', type=int, default=15,
#                             help='Number of test example per class. If negative, same as the number '
#                                  'of training examples `--num-shots` (default: 15).')
# 2750932
# Model
model_params = parser.add_argument_group('Model')
model_params.add_argument('--model-name', type=str, choices=['vanilla',
                                                             'lenet', 'mlp', 'senet', 'resnet'], default='vanilla',
                          help='Name of the Algorithm (default: maml).')
model_params.add_argument('--fine-tune', type=int, default=0, help='Only meta learn the FC layer')
model_params.add_argument('--input-dim', type=int, default=28, help='Input dimension')
model_params.add_argument('--channels', type=int, default=1, help='Input channels')
general_params.add_argument('--model', type=str, choices=['maml'], default='maml',
                            help='Name of the Algorithm (default: maml).')

# Optimization
optim_params = parser.add_argument_group('Optimization')
# optim_params.add_argument('--batch-size', type=int, default=25,
#                           help='Number of tasks in a batch of tasks (default: 25).')
optim_params.add_argument('--seed', type=int, default=101,
                          help='Default seed to start experiments.')
optim_params.add_argument('--num-steps', type=int, default=1,
                          help='Number of fast adaptation steps, ie. gradient descent '
                               'updates (default: 1).')
optim_params.add_argument('--num-epochs', type=int, default=5000,
                          help='Number of epochs of meta-training (default: 50).')
# optim_params.add_argument('--num-batches', type=int, default=100,
#                           help='Number of batch of tasks per epoch (default: 100).')
optim_params.add_argument('--num-tasks', type=int, default=32,
                          help='Meta-Batch Size: Number of tasks to sample from task distribution.')
# optim_params.add_argument('--step-size', type=float, default=0.1,
#                           help='Size of the fast adaptation step, ie. learning rate in the '
#                                'gradient descent update (default: 0.1).')
optim_params.add_argument('--first-order', action='store_true',
                          help='Use the first order approximation, do not use highers-order '
                               'derivatives during meta-optimization.')
optim_params.add_argument('--meta-lr', type=float, default=0.001,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')
optim_params.add_argument('--meta-learn', type=int, default=1,
                          help='Set this to False only for debugging purpose to'
                               'verify that meta learning is happening')
optim_params.add_argument('--fast-lr', type=float, default=0.01,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')
optim_params.add_argument('--gpu', type=str, default="2",
                          help='Gpu index')

# Misc
misc = parser.add_argument_group('Misc')
misc.add_argument('--num-workers', type=int, default=1,
                  help='Number of workers to use for data-loading (default: 1).')
# Logging
# WandB Arguments
# parser.add_argument('--wandb', type=bool, default=False, help='Run With WandB')
# parser.add_argument('--project', type=str, default='cifar10_vdp', help='WandB Project')
# parser.add_argument('--account', type=str, default='angelinic0', help='WandB Account')
misc.add_argument('--wand-project', type=str, default="VDP+MAML",
                  help='Wandb project name should go here')
misc.add_argument('--wand-note', type=str, default="Test Run Note",
                  help='To identify run')

misc.add_argument('--username', type=str, default="hikmatkhan",
                  help='Wandb username should go here')
misc.add_argument('--wandb-logging', type=int, default=1,
                  help='If True, Logs will be reported on wandb.')
misc.add_argument('--verbose', action='store_true')
misc.add_argument('--use-cuda', action='store_true')
misc.add_argument('--device', type=str, default=utils.get_compute_device(), help="Compute device information")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == '__main__':
    utils.fix_seeds(seed=args.seed)
    print(args)

    if args.wandb_logging:
        wandb.init(project="{0}_{1}".format(args.wand_project, args.network), entity="hikmatkhan-")
        # wandb.config.update(args)
    # ----------------------------------- Outer loop -----------------------------------#
    # tasksets = utils.get_torch_ds(ways=args.ways, shots=args.shots, num_tasks=args.num_tasks)
    tasksets = utils.get_l2l_ds(args.dataset, data_path=args.data, ways=args.ways,
                                shots=args.shots)

    # if args.model_name == 'vanilla':
    #     meta_theta = VanillaNet(args=args).to(args.device)
    #     # meta_theta = ResNet18S(out_dim=args.ways).to(args.device)
    # elif args.model_name == 'resnet':
    #     meta_theta = ResNet18S(out_dim=args.ways).to(args.device)
    netmodload = importlib.import_module('Networks.' + args.network + '.VDPNet', package='Net')
    meta_theta = netmodload.Net(args).to(args.device)
    print(meta_theta)
    meta_optim = optim.Adam(meta_theta.parameters(),
                            lr=args.meta_lr)

    for epoch in range(0, args.num_epochs):

        meta_optim.zero_grad()
        # ----------------------------------- Inner loop -----------------------------------#
        meta_train_acc, meta_train_loss = run_inner_loop(
            meta_theta, tasksets, args)
        # ----------------------------------- Inner loop -----------------------------------#
        if (args.meta_learn):
            meta_optim.step()
            if args.wandb_logging:
                wandb.log({"Meta Step": epoch})


        # ----------------------------------- Outer loop -----------------------------------#
        # # # Meta Evaluation
        meta_val_acc, meta_val_loss = run_val_loop(
                meta_theta, tasksets, args, eval_on_testset=False)
        # # Meta Adaptation
        meta_test_acc, meta_test_loss = run_val_loop(
            meta_theta, tasksets, args, eval_on_testset=True)

        if args.wandb_logging:
            wandb.log({"meta_train_acc":meta_train_acc,
                       "meta_train_loss":meta_train_loss,
                       "meta_val_acc":meta_val_acc,
                       "meta_val_loss":meta_val_loss,
                       "meta_test_acc":meta_test_acc,
                       "meta_test_loss":meta_test_loss,
                       })
        print("meta_train_acc:", round(meta_train_acc, 3), " meta_val_acc:", round(meta_val_acc, 3), " meta_test_acc:", round(meta_test_acc, 3))
