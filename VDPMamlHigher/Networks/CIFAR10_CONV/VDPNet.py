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
        self.lr = args.lr
        self.gamma = args.gamma
        self.lr_sched = args.lr_sched
        self.tau = args.tau
        self.clamp = args.clamp
        self.var_sup = args.var_sup

        self.loss = nn.GaussianNLLLoss(full=False, eps=1e-06, reduction='mean')

        self.conv_init = conv_weight_args(args)
        self.fc_init = fc_weight_args(args)

        self.conv2d1 = VDP_Conv2D(3, 32, 3, input_flag=True)
        self.conv2d2 = VDP_Conv2D(32, 32, 3)
        self.conv2d3 = VDP_Conv2D(32, 64, 3)
        self.conv2d4 = VDP_Conv2D(64, 64, 3)
        self.conv2d5 = VDP_Conv2D(64, 128, 3)
        self.conv2d6 = VDP_Conv2D(128, 128, 3)

        self.drop = nn.Dropout(0.2)

        self.relu = VDP_Relu()
        self.maxpool = VDP_Maxpool(2, 2, padding=1)
        self.theflattening = VDP_Flatten()
        self.fullyCon1 = VDP_FullyConnected(512, 10)
        self.softmax = VDP_Softmax(1)

    def forward(self, x_input):
        mu, sigma = self.conv2d1(x_input)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.conv2d2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.maxpool(mu, sigma)
        mu = self.drop(mu)

        mu, sigma = self.conv2d3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.conv2d4(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.maxpool(mu, sigma)
        mu = self.drop(mu)

        mu, sigma = self.conv2d5(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.conv2d6(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.maxpool(mu, sigma)
        mu = self.drop(mu)

        mu_flat, sigma_flat = self.theflattening(mu, sigma)

        mu, sigma = self.fullyCon1(mu_flat, sigma_flat)
        mu, sigma = self.softmax(mu, sigma)

        return mu, sigma

    #def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
    #    thing = torch.diag(torch.ones(list(self.children())[-2].out_features, device=y_pred_sd.device) * torch.tensor(self.var_sup, device=y_pred_sd.device))
    #    y_pred_sd_inv = torch.inverse(y_pred_sd + thing)
    #    mu_ = y_pred_mean - y_test
    #    mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
    #    ms = 0.5 * torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) + 0.5 * torch.log(torch.det(y_pred_sd + thing)).unsqueeze(1)
    #    ms = ms.mean()
    #    return ms

    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        NS = torch.diag(torch.ones(list(self.children())[-2].out_features, device=y_pred_sd.device) * torch.tensor(self.var_sup, device=y_pred_sd.device))
        y_pred_sd_inv = torch.inverse(y_pred_sd + NS)
        mu_ = y_pred_mean - y_test
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = (torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) +
              (torch.slogdet(y_pred_sd + NS)[1]).unsqueeze(1)).mean()
        return ms

    def batch_loss(self, output_mean, output_sigma, label):
        output_sigma_clamp = torch.clamp(output_sigma, -self.clamp, self.clamp)
        neg_log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, label)
        loss_value = neg_log_likelihood + (self.tau * self.conv2d1.kl_loss_term() +
                                           self.tau * self.conv2d2.kl_loss_term() +
                                           self.tau * self.conv2d3.kl_loss_term() +
                                           self.tau * self.conv2d4.kl_loss_term() +
                                           self.tau * self.conv2d5.kl_loss_term() +
                                           self.tau * self.conv2d2.kl_loss_term() +
                                           self.tau * self.fullyCon1.kl_loss_term())
        return loss_value
