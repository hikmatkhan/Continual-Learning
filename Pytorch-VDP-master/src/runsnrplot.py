# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini


"""
import re
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import importlib

from loaders import LoadData
from transforms import addNoise
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from adjustText import adjust_text
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(prog="VDP")
parser.add_argument('--network', type=str, default='VDP',
                    choices=['VDP', 'FMNIST_CONV', 'MNIST_CONV', 'CIFAR10_CONV','MNIST_FC_BBB'], help='Dataset Experiment')
# Load Model
parser.add_argument('--load_model', type=str,
                    default='F:/PycharmProjects/VDP/models/VDP/06_02_21/17_29_complete/network_MNIST_FC_valid_acc=95.9.pkl',
                    help='Path to a previously trained model checkpoint')
#parser.add_argument('--load_model', type=str,
#                    default='/home/angelinic0/projects/VDP/models/MNIST_CONV/06_02_21/16_48_complete/network_MNIST_CONV_valid_acc=98.0.pkl',
#                    help='Path to a previously trained model checkpoint')
#parser.add_argument('--load_model', type=str,
#                    default='F:/PycharmProjects/VDP/models/MNIST_FC_BBB/05_26_21/18_09_complete/network_MNIST_FC_BBB_valid_acc= 97.9.pkl',
#                    help='Path to a previously trained model checkpoint')

parser.add_argument('--continue_train', type=bool, default=False, help='Continue to train this checkpoint ')

parser.add_argument('--lightning', type=bool, default=False, help='Run with PyTorch Lightning')
# Code Parameters
parser.add_argument('--num_workers', type=int, default=16, help='Number of CPU works to process dataset')
parser.add_argument('--data_path', type=str, default='../../data/', help='Path to save dataset data')
# Training Parameters
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
# Loss Function Parameters
parser.add_argument('--tau', type=float, default=0.002, help='KL Weight Term')
parser.add_argument('--clamp', type=float, default=10, help='Clamping')
parser.add_argument('--var_sup', type=float, default=0.001, help='Loss Variance Bias')
# Learning Rate Parameters
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--lr_sched', type=float, default=[100, 150], help='Learning Rate Scheduler Milestones')
parser.add_argument('--gamma', type=float, default=0.1, help='Learning Rate Gamma')
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


def print_arguments(args):
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)


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

    modules_names = set(modules_names)

    return modules_names


def logs(model):

    lp, lvp = 0.0, 0.0
    for name in find_modules_names(model):
        n = name.split('.')
        if len(n) == 1:
            m = model._modules[n[0]]
        elif len(n) == 3:
            m = model._modules[n[0]]._modules[n[1]]._modules[n[2]]
        elif len(n) == 4:
            m = model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

        lp += m.log_prior
        lvp += m.log_variational_posterior

    return lp, lvp


def main():
    args = parser.parse_args()

    netmodload = importlib.import_module('Networks.' + args.network + '.VDPNet', package='Net')
    model = netmodload.Net(args)

    if not args.lightning and torch.cuda.device_count() >= 1:
        args.devices = torch.device('cuda:0')
        print('Using device:', args.devices)

    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    devices = torch.device('cuda:0')

    model.to(devices)

    snrs = np.arange(-10,30,2)

    var_class0 = list()
    var_class1 = list()
    var_class2 = list()
    var_class3 = list()
    var_class4 = list()
    var_class5 = list()
    var_class6 = list()
    var_class7 = list()
    var_class8 = list()
    var_class9 = list()

    var_correct = list()
    var_incorrt = list()

    var_pred = list()
    var = list()
    accuracy = list()
    for i, snr in enumerate(snrs):
        print('SNR: ' + str(snr))
        transform_in = transforms.Compose([transforms.ToTensor(), addNoise(snr)])

        dset = re.split('_', args.network)[0]
        data = LoadData(dset, transform_in, args)
        _, test_loader = data.get_loaders()

        model.train()
        total_loss = 0
        total_acc = 0
        total_num = 0

        var_batch = np.empty((len(test_loader), args.batch_size, 10), dtype=float)

        var_batch_pred = np.empty((len(test_loader), args.batch_size), dtype=float)

        var_batch_correct = []
        var_batch_incorrect = []

        var_batch_class0 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class1 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class2 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class3 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class4 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class5 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class6 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class7 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class8 = np.empty((len(test_loader), args.batch_size), dtype=float)
        var_batch_class9 = np.empty((len(test_loader), args.batch_size), dtype=float)

        for idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(devices), targets.to(devices)

            #if idx == 1:
            #    plt.imshow(data[0,0,:].detach().cpu().numpy())
            #    plt.title(f'SNR: {snr}')
            #    plt.show()

            labels = nn.functional.one_hot(targets, len(torch.unique(test_loader.dataset.targets)))

            if 'BBB' in args.network:
                for_var = torch.empty(args.samples, len(targets), 10, device=devices)
                lps, lvps, predictions = [], [], []
                for i in range(args.samples):
                    preds = model(data)
                    predictions.append(preds)
                    for_var[i] = preds
                    lp, lv = logs(model)
                    lps.append(lp)
                    lvps.append(lv)

                mu_y_out = torch.mean(for_var, dim=0)
                _, pred = mu_y_out.max(1, keepdim=True)
                var_only = torch.var(for_var, dim=0).detach().cpu().numpy()

                loss = model.batch_loss(predictions, lps, lvps, len(test_loader), targets)
            else:
                mu_y_out, sigma_y_out = model.forward(data)
                loss = model.batch_loss(mu_y_out, sigma_y_out, labels)

                # Get Variance Only
                _, pred = mu_y_out.max(1, keepdim=True)
                var_only = torch.diagonal(sigma_y_out, dim1=1, dim2=2).detach().cpu().numpy()

            total_loss += loss.detach().cpu().numpy() * len(targets)
            total_acc += pred.eq(targets.view_as(pred)).sum().item()
            total_num += len(targets)

            var_batch[idx] = var_only
            # Get Variance of predicted value
            pred_var = var_only[np.arange(len(var_only)), pred.T[0].detach().cpu().numpy()]
            var_batch_pred[idx] = pred_var

            for j in range(len(targets)):
                if pred.T[0, j] == targets[j]:
                    var_batch_correct.append(pred_var[j])
                else:
                    var_batch_incorrect.append(pred_var[j])

            var_batch_class0[idx] = var_only[:, 0]
            var_batch_class1[idx] = var_only[:, 1]
            var_batch_class2[idx] = var_only[:, 2]
            var_batch_class3[idx] = var_only[:, 3]
            var_batch_class4[idx] = var_only[:, 4]
            var_batch_class5[idx] = var_only[:, 5]
            var_batch_class6[idx] = var_only[:, 6]
            var_batch_class7[idx] = var_only[:, 7]
            var_batch_class8[idx] = var_only[:, 8]
            var_batch_class9[idx] = var_only[:, 9]

        var.append(np.mean(var_batch))
        var_pred.append(np.mean(var_batch_pred))
        var_correct.append(sum(var_batch_correct) / len(var_batch_correct))
        var_incorrt.append(sum(var_batch_incorrect) / len(var_batch_incorrect))
        var_class0.append(np.mean(var_batch_class0))
        var_class1.append(np.mean(var_batch_class1))
        var_class2.append(np.mean(var_batch_class2))
        var_class3.append(np.mean(var_batch_class3))
        var_class4.append(np.mean(var_batch_class4))
        var_class5.append(np.mean(var_batch_class5))
        var_class6.append(np.mean(var_batch_class6))
        var_class7.append(np.mean(var_batch_class7))
        var_class8.append(np.mean(var_batch_class8))
        var_class9.append(np.mean(var_batch_class9))

        loss = total_loss / total_num
        acc = total_acc / total_num

        accuracy.append(acc)

        print('Loss: ' + str(loss))
        print('Acc: ' + str(acc))

    fig, ax1 = plt.subplots()

    ax1.plot(snrs, var_class0, label='Mean Variance\n of Classes', color='gray')
    ax1.plot(snrs, var_class1, color='gray')
    ax1.plot(snrs, var_class2, color='gray')
    ax1.plot(snrs, var_class3, color='gray')
    ax1.plot(snrs, var_class4, color='gray')
    ax1.plot(snrs, var_class5, color='gray')
    ax1.plot(snrs, var_class6, color='gray')
    ax1.plot(snrs, var_class7, color='gray')
    ax1.plot(snrs, var_class8, color='gray')
    ax1.plot(snrs, var_class9, color='gray')


    ax1.plot(snrs, var_pred, label='Mean Variance\n of Predicted Class', color='red')
    ax1.plot(snrs, var_correct, label='Mean Variance\n of Correct Pred', color='cyan')
    ax1.plot(snrs, var_incorrt, label='Mean Variance\n of Incorrect Pred', color='Green')
    ax1.plot(snrs, var, label='Mean Variance\n of output', color='blue')

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('VDP Test Variance')
    ax1.set_yscale('linear')

    ax2 = ax1.twinx()

    ax2.plot(snrs, accuracy, label='Test Accuracy', color='orange')

    texts = []

    texts.append(ax1.text(snrs[0], var_class0[0], '0', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class1[0], '1', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class2[0], '2', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class3[0], '3', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class4[0], '4', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class5[0], '5', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class6[0], '6', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class7[0], '7', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class8[0], '8', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))
    texts.append(ax1.text(snrs[0], var_class9[0], '9', verticalalignment='center', horizontalalignment='right', fontsize='x-small', color='gray'))

    adjust_text(texts, force_text=(0.25, 0.25), only_move={'text':'xy'})
    for te in texts:
        te.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                           path_effects.Normal()])

    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Model Accuracy')
    ax2.set_yscale('linear')
    ax2.set_ylim(0,1)
    loaded_path = os.path.split(args.load_model)
    ax1.set_title(loaded_path[1])
    fig.legend(loc='lower left',  bbox_to_anchor=(0.55, 0.3))
    plt.savefig(loaded_path[0] + '/SNR_Variance_plot.png', dpi=300)
    ax1.set_ylim(-1, var[0]*2)
    plt.savefig(loaded_path[0] + '/SNR_Variance_plot_high_var_insp.png', dpi=300)



if __name__ == '__main__':
    main()