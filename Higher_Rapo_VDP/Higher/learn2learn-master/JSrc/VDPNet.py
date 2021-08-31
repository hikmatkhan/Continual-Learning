# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini

"""

import torch
import torch.nn as nn
import torch.utils.data
from Comps.VDP_Layers import VDP_Flatten, VDP_Conv2D, VDP_Relu, VDP_Maxpool, VDP_FullyConnected, VDP_Softmax
from utils import conv_weight_args, fc_weight_args


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        # Hypers
        # self.lr = args.lr
        # self.gamma = args.gamma
        # self.lr_sched = args.lr_sched
        self.tau = args.tau
        self.clamp = args.clamp
        self.var_sup = args.var_sup

        self.fc_init = fc_weight_args(args)
        # Omniglot 256, 128, 64, 64
        self.fullyCon1 = VDP_FullyConnected(784, 256, input_flag=True, weight_args=self.fc_init)
        # self.fullyCon1 = VDP_FullyConnected(1024, 512, input_flag=True, weight_args=self.fc_init)
        self.relu = VDP_Relu()
        # Added
        self.fullyCon3 = VDP_FullyConnected(256, 128, weight_args=self.fc_init)
        self.fullyCon4 = VDP_FullyConnected(128, 64, weight_args=self.fc_init)
        self.fullyCon5 = VDP_FullyConnected(64, 64, weight_args=self.fc_init)
        # self.dropout = nn.Dropout(0.25)

        self.fullyCon2 = VDP_FullyConnected(64, args.ways, weight_args=self.fc_init)
        self.softmax = VDP_Softmax(1)

    def forward(self, x):
        mu_flat = torch.flatten(x.permute([0, 2, 3, 1]), start_dim=1)
        mu, sigma = self.fullyCon1.forward(mu_flat)
        mu, sigma = self.relu.forward(mu, sigma)

        # # Added
        mu, sigma = self.fullyCon3.forward(mu, sigma)
        mu, sigma = self.relu.forward(mu, sigma)
        mu, sigma = self.fullyCon4.forward(mu, sigma)
        mu, sigma = self.relu.forward(mu, sigma)


        mu, sigma = self.fullyCon5.forward(mu, sigma)
        mu, sigma = self.relu.forward(mu, sigma)

        mu, sigma = self.fullyCon2.forward(mu, sigma)
        mu, sigma = self.softmax.forward(mu, sigma)

        return mu, sigma

    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        NS = torch.diag(
            torch.ones(list(self.children())[-2].out_features, device=y_pred_sd.device) * torch.tensor(self.var_sup,
                                                                                                       device=y_pred_sd.device))
        y_pred_sd_inv = torch.inverse(y_pred_sd + NS)
        mu_ = y_pred_mean - y_test
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = (torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) +
              (torch.slogdet(y_pred_sd + NS)[1]).unsqueeze(1)).mean()
        return ms

    def batch_loss(self, output_mean, output_sigma, label):
        output_sigma_clamp = torch.clamp(output_sigma, -self.clamp, self.clamp)
        neg_log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, label)
        loss_value = neg_log_likelihood + (self.tau * self.fullyCon1.kl_loss_term() +
                                           self.tau * self.fullyCon2.kl_loss_term())
        return loss_value

