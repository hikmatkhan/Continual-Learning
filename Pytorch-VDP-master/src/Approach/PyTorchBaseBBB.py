# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini


"""
import os
import time
import wandb
import numpy as np
import torch.nn as nn
import torch.utils.data
from utils import kde_plot_layers



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


def eval(args, model, loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(loader):
        data = data.to(args.devices)
        targets = targets.to(args.devices)

        output = model(data, sample=False)
        _, pred = output.max(1, keepdim=True)

        lps, lvps, predictions = [], [], []
        for i in range(args.samples):
            predictions.append(model(data))
            lp, lv = logs(model)
            lps.append(lp)
            lvps.append(lv)

        loss = model.batch_loss(predictions, lps, lvps, len(loader), targets)

        total_loss += loss.detach().cpu().numpy()
        total_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)

    loss = total_loss / total_num
    acc = total_acc / total_num

    return loss, acc


def run(args, model, train_loader, test_loader, optim, meta_optim, sched, swa_model=None, swa_scheduler=None):
    prev_epoch = 0

    if args.load_model != '':
        # When resuming training
        loaded_path = os.path.split(args.load_model)[0]
        args.base_save = loaded_path + '/continue/'

        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict']).to(model.device)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        prev_epoch = checkpoint['epoch']

    if args.load_model == '' or args.continue_train:
        if args.continue_train:
            print(str(prev_epoch) + ' previous epochs were trained.')
            args.epochs = args.epochs - prev_epoch
            print('After this training session the model will have trained for '+str(args.epochs)+' epochs')
        prev_valid_acc = 0
        for i_ep in range(args.epochs):
            total_loss = 0
            total_acc = 0
            total_num = 0
            model.train()


            print('| Epoch {:3d}, '.format(i_ep+prev_epoch), end='')
            for idx, (data, targets) in enumerate(train_loader):
                optim.zero_grad()
                data, targets = data.to(args.devices), targets.to(args.devices)

                output = model(data, sample=False)
                _, pred = output.max(1, keepdim=True)

                lps, lvps, predictions = [], [], []
                for i in range(args.samples):
                    predictions.append(model(data))
                    lp, lv = logs(model)
                    lps.append(lp)
                    lvps.append(lv)

                loss = model.batch_loss(predictions, lps, lvps, len(train_loader), targets)

                total_loss += loss.detach().cpu().numpy()
                total_acc += pred.eq(targets.view_as(pred)).sum().item()
                total_num += len(targets)

                loss.backward(retain_graph=True)
                optim.step()

            train_loss = total_loss / total_num
            train_acc = total_acc / total_num

            print(' | Train: loss={:8.4f}, acc={:5.1f}% | '.format(np.round(train_loss, 4), 100 * train_acc), end='')

            valid_loss, valid_acc = eval(args, model, test_loader)
            print('Valid: acc={:5.1f}% | '.format(100 * valid_acc), end='')

            print('')

            # Save Every Tenth Epoch
            if valid_acc > prev_valid_acc:
                prev_valid_acc = valid_acc
                model_file = args.base_save + 'network_' + args.network + '_valid_acc={:5.1f}'.format(100 * valid_acc) + '.pkl'
                torch.save({
                    'epoch': i_ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': train_loss}, model_file)

    else:
        print('*'*40+' Validation Only Run '+'*'*40)
        valid_loss, valid_acc = eval(args, model, test_loader)
        print(' Valid: loss={:8.2f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
        print('')