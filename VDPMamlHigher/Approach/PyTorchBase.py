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


def eval(args, model, loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(loader):
        data, targets = data.to(args.devices), targets.to(args.devices)

        mu_y_out, sigma_y_out = model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)

        total_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)

    loss = total_loss / total_num
    acc = total_acc / total_num

    return loss, acc


def run(args, model, train_loader, test_loader, optim, sched, swa_model=None, swa_scheduler=None):
    prev_epoch = 0

    if args.load_model != '':
        loaded_path = os.path.split(args.load_model)[0]
        args.base_save = loaded_path + '/continue/'

        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict']).to(model.device)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']

    if args.kdes:
        kde_path = args.base_save + 'KDEs/'
        if not os.path.exists(kde_path):
            os.makedirs(kde_path)
        kde_plot_layers(model, 0, kde_path)

    if args.load_model == '' or args.continue_train:
        if args.continue_train:
            print(str(prev_epoch) + ' previous epochs were trained.')
            args.epochs = args.epochs - prev_epoch
            print('After this training session the model will have trained for '+str(args.epochs)+' epochs')
        prev_valid_acc = 0
        for i_ep in range(args.epochs):

            clock0 = time.time()
            total_loss = 0
            total_acc = 0
            total_num = 0
            model.train()
            print('| Epoch {:3d}, '.format(i_ep+prev_epoch), end='')
            for idx, (data, targets) in enumerate(train_loader):
                optim.zero_grad()
                data, targets = data.to(args.devices), targets.to(args.devices)

                mu_y_out, sigma_y_out = model.forward(data)
                labels = nn.functional.one_hot(targets, list(model.children())[-2].out_features)
                loss = model.batch_loss(mu_y_out, sigma_y_out, labels)

                _, pred = mu_y_out.max(1, keepdim=True)

                total_loss += loss.detach().cpu().numpy()
                total_acc += pred.eq(targets.view_as(pred)).sum().item()
                total_num += len(targets)

                loss.backward()
                optim.step()

            if 0 != args.swa_start and \
                    ((args.swa_start <= i_ep and i_ep % args.swa_anneal == 0) or i_ep == args.epochs-1):
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                sched.step()

            train_loss = total_loss / total_num
            train_acc = total_acc / total_num

            print(' | Train: loss={:8.4f}, acc={:5.1f}% | '.format(np.round(train_loss, 4), 100 * train_acc), end='')

            optim.zero_grad()
            del data, targets, mu_y_out, sigma_y_out, loss, pred

            valid_loss, valid_acc = eval(args, model, test_loader)
            print('Valid: acc={:5.1f}% | '.format(100 * valid_acc), end='')

            if 0 != args.swa_start and \
                    ((args.swa_start <= i_ep and i_ep % args.swa_anneal == 0) or i_ep == args.epochs-1):
                torch.optim.swa_utils.update_bn(train_loader, swa_model)
                valid_loss, valid_acc = eval(args, swa_model, test_loader)
                print('SWA Valid: acc={:5.1f}% | '.format(100 * valid_acc), end='')

           # for param_group in optim.param_group:
           #     lr = param_group['lr']
           #     print(f'LR: {lr}', end='')
            print('')

            if args.kdes:
                if i_ep > 0 and i_ep % 2 == 0:
                    kde_plot_layers(model, i_ep, kde_path)

            if args.wandb:
                wandb.log({'Validation Accuracy': valid_acc, 'epoch': i_ep})
                wandb.log({'Validation Loss': valid_loss, 'epoch': i_ep})
                wandb.log({'Training Accuracy': train_acc, 'epoch': i_ep})
                wandb.log({'Training Loss': train_loss, 'epoch': i_ep})

            # Save Every Tenth Epoch
            if valid_acc > prev_valid_acc:
                prev_valid_acc = valid_acc
                model_file = args.base_save + 'network_' + args.network + '_valid_acc={:3.1f}'.format(100 * valid_acc) + '.pkl'
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