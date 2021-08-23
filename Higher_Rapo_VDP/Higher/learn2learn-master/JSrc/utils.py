# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini

"""
import os
import torch
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def print_arguments(args):
    print('=' * 101)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 101)


class conv_weight_args:
    def __init__(self, args):
        self.conv_input_mean_mu = args.conv_input_mean_mu
        self.conv_input_mean_sigma = args.conv_input_mean_sigma
        self.conv_input_mean_bias = args.conv_input_mean_bias
        self.conv_input_sigma_min = args.conv_input_sigma_min
        self.conv_input_sigma_max = args.conv_input_sigma_max
        self.conv_input_sigma_bias = args.conv_input_sigma_bias


class fc_weight_args:
    def __init__(self, args):
        self.fc_input_mean_mu = args.fc_input_mean_mu
        self.fc_input_mean_sigma = args.fc_input_mean_sigma
        self.fc_input_mean_bias = args.fc_input_mean_bias
        self.fc_input_sigma_min = args.fc_input_sigma_min
        self.fc_input_sigma_max = args.fc_input_sigma_max
        self.fc_input_sigma_bias = args.fc_input_sigma_bias


def find_modules_names(model, with_classifier=False):
    modules_names = []
    for name, p in model.named_parameters():
        if with_classifier is False:
            if not name.startswith('classifier'):
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))
        else:
            n = name.split('.')[:-1]
            modules_names.append('.'.join(n))

    modules_names = list(set(modules_names))

    layer_names = []
    for name in modules_names:
        n = name.split('.')
        if len(n) == 1:
            layer_names.append(n[0])
    layer_names.sort()
    return layer_names


def kde_plot_layers(args, model, epoch, save_path):
    column_name = f"Epoch_{epoch:04d}_End"
    for layer in find_modules_names(model):
        layer_kde_path = save_path + layer + '/'
        mean_layer_kde_path = layer_kde_path + 'mean/'
        sigma_layer_kde_path = layer_kde_path + 'sigma/'
        ss_layer_kde_path = layer_kde_path + 'sigma_sp/'
        if not os.path.exists(mean_layer_kde_path):
            os.makedirs(mean_layer_kde_path)
        if not os.path.exists(sigma_layer_kde_path):
            os.makedirs(sigma_layer_kde_path)
        if not os.path.exists(ss_layer_kde_path):
            os.makedirs(ss_layer_kde_path)

        if 'fc' or 'ful' in layer:
            mean_df = pd.DataFrame(getattr(model, layer).mean.weight.view(-1, 1).detach().cpu().numpy(), columns=[column_name])
            fig = plt.figure()
            sns.kdeplot(mean_df[column_name], shade=True, label=column_name)
            plt.title(layer + ' Mean ' + column_name)
            plt.xlabel('Mean Value')
            plt.ylabel('Density')
            fig.savefig(mean_layer_kde_path + f"{layer}_mean_kde_{column_name}.png")
            if args.wandb:
                wandb.log({f"{layer}_mean_kde": plt})
            plt.close()

            sigma_df = pd.DataFrame(getattr(model, layer).sigma_weight.view(-1, 1).detach().cpu().numpy(), columns=[column_name])
            fig = plt.figure()
            sns.kdeplot(sigma_df[column_name], shade=True, label=column_name)
            plt.title(layer + ' Sigma ' + column_name)
            plt.xlabel('Variance Value')
            plt.ylabel('Density')
            fig.savefig(sigma_layer_kde_path + f"{layer}_sigma_kde_{column_name}.png")
            if args.wandb:
                wandb.log({f"{layer}_sigma_kde": plt})
            plt.close()

            ss_df = pd.DataFrame(torch.log1p(torch.exp(getattr(model, layer).sigma_weight.view(-1, 1).detach().cpu())).numpy(), columns=[column_name])
            fig = plt.figure()
            sns.kdeplot(ss_df[column_name], shade=True, label=column_name)
            plt.title(layer + ' SoftPlus Sigma ' + column_name)
            plt.xlabel('Variance Value')
            plt.ylabel('Density')
            fig.savefig(ss_layer_kde_path + f"{layer}_sigma_sp_kde_{column_name}.png")
            if args.wandb:
                wandb.log({f"{layer}_sigma_sp_kde": plt})
            plt.close()
        else:
            getattr(model, layer).mean.weight.shape


#-------------------------------- VDP Maml ----------------------------------_#
import random

import learn2learn
import numpy as np
import torch
import torchvision
from learn2learn.data import TaskDataset
from learn2learn.data.transforms import NWays, KShots, LoadData
from torchvision.transforms import transforms


def fix_seeds(seed=101):
    # No randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)


def get_compute_device():
    device = torch.device('cpu')
    if torch.cuda.device_count():
        device = torch.device('cuda')
    return device


def get_torch_ds(dataset =
    #              torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose(
    # [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                 torchvision.datasets.MNIST(root="data", train=True,
                        download=True, transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])),
                 ways=10, shots=1, num_tasks=32):
    '''
    # tasksets = utils.get_torch_ds(ways=args.ways, shots=args.shots, num_tasks=args.num_tasks)
    # for task in tasksets:
    #     X, y = task
    #     X, y = X.to(args.device), y.to(args.device)
    #     print("Y:", y)
    '''
    # MetaDataset MNIST or Custom Dataset
    dataset = learn2learn.data.MetaDataset(dataset)
    l2l_transforms = [
        NWays(dataset, n=ways),
        KShots(dataset, k=shots * 2),
        LoadData(dataset),
    ]
    tasksets = TaskDataset(dataset, l2l_transforms, num_tasks=num_tasks)

    return tasksets


def get_l2l_ds(dataset_name, data_path="./data", ways=1, shots=5):
    tasksets = learn2learn.vision.benchmarks.get_tasksets(dataset_name,
                                                          train_samples=shots*2,  # 2*shots,
                                                          train_ways=ways,
                                                          test_samples=shots*2,  # 2*shots,
                                                          test_ways=ways,
                                                          root=data_path,
                                                          num_tasks=-1)
    return tasksets


def get_indices(X, args):
    # Separate data into Meta-Train/Meta-Test sets
    meta_train_indices = np.zeros(X.size(0), dtype=bool)
    # print("X:", X.size())
    meta_train_indices[np.arange(args.shots * args.ways) * 2] = True
    meta_test_indices = torch.from_numpy(~meta_train_indices)
    meta_train_indices = torch.from_numpy(meta_train_indices)
    return meta_train_indices, meta_test_indices


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

# def copy_models_weights(model):
#     deep_copy =
#     for name, param in model_1_params:
#         if name in model_2_params_dict:
#             model_2_params_dict[name].data.copy_(param.data)
# from models.vanilla_net import VanillaNet
# from models.lenet import LeNetwork
# from models.resnet import ResNet18
# from models.resnet import ResNet18S


# def load_meta_theta(args):
#
#     if args.dataset == "omniglot":
#         if args.model_name == 'vanilla':
#             meta_theta = VanillaNet(args=args).to(args.device)
#         elif args.model_name == 'lenet':
#             meta_theta = LeNetwork(out_dim=args.ways,
#                                    in_channel=args.channels,
#                                    img_sz=args.input_dim).to(args.device)
#         elif args.model_name == 'resnet':
#             meta_theta = ResNet18S(out_dim=args.ways).to(args.device)
#
#         else:
#             print("Couldn't found appropriate model for Omniglot")
#
#     elif args.model_name == 'mini-imagenet':
#         if args.model_name == 'lenet':
#             meta_theta = LeNetwork(out_dim=args.ways,
#                                    in_channel=args.channels,
#                                    img_sz=args.input_dim).to(args.device)
#         elif args.model_name == 'resnet':
#             meta_theta = ResNet18(out_dim=args.ways).to(args.device)
#         else:
#             print("Couldn't found appropriate model for Omniglot")
#
#     elif args.model_name == 'cifarfs':
#         if args.model_name == 'lenet':
#             meta_theta = LeNetwork(out_dim=args.ways,
#                                    in_channel=args.channels,
#                                    img_sz=args.input_dim).to(args.device)
#         elif args.model_name == 'resnet':
#             meta_theta = ResNet18(out_dim=args.ways).to(args.device)
#         else:
#             print("Couldn't found appropriate model for Omniglot")
#
#     elif args.model_name == 'tiered-imagenet':
#         if args.model_name == 'lenet':
#             meta_theta = LeNetwork(out_dim=args.ways,
#                                    in_channel=args.channels,
#                                    img_sz=args.input_dim).to(args.device)
#         elif args.model_name == 'resnet':
#             meta_theta = ResNet18(out_dim=args.ways).to(args.device)
#         else:
#             print("Couldn't found appropriate model for Omniglot")
#
#     else:
#         print("Dataset not found.")
#
#     return meta_theta
#-------------------------------- VDP Maml ----------------------------------_#
