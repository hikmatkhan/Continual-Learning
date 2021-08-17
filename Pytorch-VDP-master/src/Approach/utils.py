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