# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini


"""
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import grad


class default_weight_args:
    def __init__(self):
        super(default_weight_args, self).__init__()
        self.fc_input_mean_mu = 0
        self.fc_input_mean_sigma = 0.1
        self.fc_input_mean_bias = 0.0001
        self.fc_input_sigma_min = -12
        self.fc_input_sigma_max = -2.2
        self.fc_input_sigma_bias = 0.0001

        self.conv_input_mean_mu = 0
        self.conv_input_mean_sigma = 0.1
        self.conv_input_mean_bias = 0.0001
        self.conv_input_sigma_min = -12
        self.conv_input_sigma_max = -2.2
        self.conv_input_sigma_bias = 0.0001


global_layer_default = default_weight_args()


class VDP_Flatten(nn.Module):
    def __init__(self):
        super(VDP_Flatten, self).__init__()

    def forward(self, mu, sigma):
        mu_flat = torch.flatten(mu.permute([0, 2, 3, 1]), start_dim=1)

        sigma_flat = torch.zeros(mu_flat.shape[0], mu_flat.shape[1], mu_flat.shape[1], device=sigma.device)

        batch = sigma.shape[0]
        for i in range(batch):
            sigma_flat[i, :] = torch.block_diag(*sigma[i, :])

        return mu_flat, sigma_flat


class VDP_Conv2D(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation 2D Convolutional
    Layer. This class contains the function :func:`__init__` for the initialization of the instance,
    the function :func:`forward` for the forward propagation through the layer when called, and the
    last function :func:`kl_loss_term` which is called in the loss function as part of the
    regularization of the network.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=(5, 5), stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', weight_args=global_layer_default,
                 input_flag=False):

        super(VDP_Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_flag = input_flag

        self.bias = bias

        # Standard Conv Layer
        self.mean = nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation,
                                   groups, bias, padding_mode)
        # Conv Layer Initializing
        nn.init.normal_(self.mean.weight, mean=weight_args.conv_input_mean_mu, std=weight_args.conv_input_mean_sigma)
        self.mean.bias.data.fill_(weight_args.conv_input_mean_bias)
        # Sigma Conv Layer
        self.sigma_weight = nn.Parameter(torch.zeros([1, out_channels]), requires_grad=True)
        self.sigma_bias = nn.Parameter(torch.tensor(weight_args.conv_input_sigma_bias), requires_grad=True)

        nn.init.uniform_(self.sigma_weight, a=weight_args.conv_input_sigma_min, b=weight_args.conv_input_sigma_max)

        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, mu, sigma=0):
        """
        Forward pass over the VDP 2D Convolutional Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required only with input_flag=False)
        :type  sigma:   Float
        """
        # Input Version
        mu_z = self.mean(mu)
        x_patches = self.unfold(mu).permute(0, 2, 1)
        x_matrix = torch.bmm(x_patches, x_patches.permute(0, 2, 1))
        x_matrix_tile = x_matrix.unsqueeze(1).repeat(1, self.out_channels, 1, 1)
        if self.bias:
            sigma_z = torch.mul(torch.log1p(torch.exp(self.sigma_weight)), x_matrix_tile.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + self.sigma_bias
        else:
            sigma_z = torch.mul(torch.log1p(torch.exp(self.sigma_weight)), x_matrix_tile.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if not self.input_flag:
            # Non-Input Version

            diag_sigma = torch.diagonal(sigma, dim1=2, dim2=3)
            sigma_patches = self.unfold(diag_sigma.reshape([mu.shape[0], mu.shape[1], mu.shape[2], mu.shape[2]]))

            mu_cov = torch.reshape(self.mean.weight, [-1, self.kernel_size * self.kernel_size * mu.shape[1]])
            mu_cov_square = mu_cov * mu_cov
            trace = torch.sum(sigma_patches, 1).unsqueeze(2).repeat(1, 1, self.out_channels)
            mu_wT_sigma_mu_w1 = torch.matmul(mu_cov_square, sigma_patches) + torch.mul(torch.log1p(torch.exp(self.sigma_weight)), trace).permute(0, 2, 1)

            sigma_1 = torch.diag_embed(mu_wT_sigma_mu_w1, dim1=2)

            sigma_z = sigma_1 + sigma_z

        return mu_z, sigma_z

    def kl_loss_term(self):
        """
        KL Loss term for the loss function
        """

        c_s = torch.log1p(torch.exp(self.sigma_weight))

        kl_loss = -0.5 * torch.mean((self.kernel_size * self.kernel_size) * torch.log(c_s) +
                                    (self.kernel_size * self.kernel_size) -
                                    torch.norm(self.mean.weight) ** 2 -
                                    (self.kernel_size * self.kernel_size) * c_s)

        return kl_loss


class VDP_Relu(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation RELU. This
    class contains the function :func:`__init__` for the initialization of the instance and
    the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, inplace=False):
        """
        :param inplace:         (Default  False)
        """
        super(VDP_Relu, self).__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP Relu Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_g = self.relu(mu)

        activation_gradient = grad(mu_g.sum(), mu, retain_graph=True)[0]

        if len(mu_g.shape) == 2:

            grad_square = torch.bmm(activation_gradient.unsqueeze(2), activation_gradient.unsqueeze(1))

            sigma_g = torch.mul(sigma, grad_square)
        else:
            gradient_matrix = activation_gradient.permute([0, 2, 3, 1]).view(activation_gradient.shape[0], -1, mu_g.shape[1]).unsqueeze(3)

            grad1 = gradient_matrix.permute([0, 2, 1, 3])
            grad2 = grad1.permute([0, 1, 3, 2])

            grad_square = torch.matmul(grad1, grad2)

            sigma_g = torch.mul(sigma, grad_square)  # shape =[image_size*image_size,image_size*image_size, num_filters[0]]
        return mu_g, sigma_g


class VDP_Maxpool(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation 2D Maxpool. This
    class contains the function :func:`__init__` for the initialization of the instance and
    the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False, padding_mode='zeros'):
        """
        :param kernel_size:     Kernel Size     (Default  2)
        :param stride:          Stride Length   (Default  2)
        :param padding:         Padding T/F     (Default  0)
        :param dilation:        Dilation        (Default  1)
        :param return_indices:  Return Indices  (Default  True)
        :param ceil_mode:       Ceiling Mode    (Default  False)
        """
        super(VDP_Maxpool, self).__init__()
        self.maxPooling = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP 2D Maxpool Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_p, argmax1 = self.maxPooling(mu)

        argmax = argmax1.reshape(argmax1.shape[0], argmax1.shape[1], argmax1.shape[2] ** 2)

        argmax_indices = argmax.repeat(1, 1, argmax.shape[-1]).reshape(argmax.shape[0], argmax.shape[1], argmax.shape[-1], argmax.shape[-1])

        index_fix = argmax.unsqueeze(3) * sigma.shape[-1]
        sigma_indexes = argmax_indices + index_fix

        sigma_p = torch.gather(sigma.reshape(argmax.shape[0], argmax.shape[1], -1),
                               dim=2,
                               index=sigma_indexes.reshape(argmax.shape[0], argmax.shape[1], -1)).reshape(argmax.shape[0], argmax.shape[1],
                                                                                                          argmax.shape[2], argmax.shape[2])
        return mu_p, sigma_p


class VDP_Full_FullyConnected(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation Fully Connected
    Layer. This class contains the function :func:`__init__` for the initialization of the instance,
    the function :func:`forward` for the forward propagation through the layer when called, and
    the last function :func:`kl_loss_term` which is called in the loss function as part of the
    regularization of the network.
    """

    def __init__(self, in_features, out_features, bias=True, input_flag=False,
                 weight_args=global_layer_default):
        """
        :param in_features:     (Required)
        :param out_features:    (Required)
        :param bias:            (Default  True)
        :param mean_mu:         (Default  0)
        :param mean_sigma:      (Default  0.1)
        :param sigma_min:       (Default -12)
        :param sigma_max:       (Default -2.2)
        :param mean_bias:       (Default  0.001)
        :param sigma_bias:      (Default  0.001)
        :param input_flag:      (Default  False)
        """
        super(VDP_Full_FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.input_flag = input_flag

        self.mean = nn.Linear(in_features, out_features, bias)

        nn.init.normal_(self.mean.weight, mean=weight_args.fc_input_mean_mu, std=weight_args.fc_input_mean_sigma)
        self.mean.bias.data.fill_(weight_args.fc_input_mean_bias)

        self.sigma_weight = nn.Parameter(torch.zeros([out_features, in_features]), requires_grad=True)

        nn.init.uniform_(self.sigma_weight, a=weight_args.fc_input_sigma_min, b=weight_args.fc_input_sigma_max)
        self.sigma_bias = nn.Parameter(torch.ones([out_features])*self.bias, requires_grad=True)

    def forward(self, mu, sigma=0):
        """
        Forward pass over the VDP 2D Fully Connected Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required if input_flag=False)
        :type  sigma:   Float
        """
        if self.input_flag:
            # Input Version
            mu_f = self.mean(mu)

            # Mu in ^2 * Sigma Weights
            mu_in = torch.square(mu)
            if self.bias:
                sigma_out = torch.matmul(torch.log1p(torch.exp(self.sigma_weight)).unsqueeze(0), mu_in.unsqueeze(2)).squeeze(2) + torch.diag(self.sigma_bias)
            else:
                sigma_out = torch.matmul(torch.log1p(torch.exp(self.sigma_weight)).unsqueeze(0), mu_in.unsqueeze(2)).squeeze(2)

            sigma_out = torch.diag_embed(sigma_out, dim1=1)

        else:
            # Mu Weight * Mu in
            mu_f = self.mean(mu)
            flag = False

            if len(sigma.shape) == 4:
                flag = True
                # tr(Sigma Weights * Sigma In)
                diag_w_sigma = torch.diag_embed(torch.log1p(torch.exp(self.sigma_weight)))
                sigma = torch.diagonal(sigma, dim1=2, dim2=3).reshape(sigma.shape[0], -1).unsqueeze(1)
                sigma_in = sigma.unsqueeze(1).repeat(1, self.out_features, 1, 1)
                sigma_1 = torch.diag_embed(torch.diagonal(torch.matmul(sigma_in, diag_w_sigma), dim1=2, dim2=3).sum(2))
            else:
                # tr(Sigma Weights * Sigma In)
                diag_w_sigma = torch.diag_embed(torch.log1p(torch.exp(self.sigma_weight)))
                sigma_in = sigma.unsqueeze(1).repeat(1, self.out_features, 1, 1)
                sigma_1 = torch.diag_embed(torch.diagonal(torch.matmul(sigma_in, diag_w_sigma), dim1=2, dim2=3).sum(2))

            # Mean Weight^2 * sigma in
            w_mu = self.mean.weight.unsqueeze(0)

            sigma_2 = torch.matmul(torch.matmul(w_mu, sigma), w_mu.permute([0, 2, 1]))

            # Mu in ^2 * Sigma Weights
            mu_in = torch.square(mu)
            if self.bias:
                sigma_3 = torch.matmul(torch.log1p(torch.exp(self.sigma_weight)).unsqueeze(0), mu_in.unsqueeze(2)).squeeze(2) + torch.diag(self.sigma_bias)
            else:
                sigma_3 = torch.matmul(torch.log1p(torch.exp(self.sigma_weight)).unsqueeze(0), mu_in.unsqueeze(2)).squeeze(2)

            diag_sig_3 = torch.diag_embed(sigma_3, dim1=1)

            sigma_out = sigma_1 + sigma_2 + diag_sig_3
        return mu_f, sigma_out

    def kl_loss_term(self):
        """
        KL Loss term for the loss function
        """
        f_s = torch.log1p(torch.exp(self.sigma_weight))

        kl_loss = -0.5 * torch.mean((self.in_features * torch.log(f_s)) +
                                    self.in_features -
                                    torch.norm(self.mean.weight) ** 2 -
                                    (self.in_features * f_s))
        return kl_loss


class VDP_FullyConnected(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation Fully Connected
    Layer. This class contains the function :func:`__init__` for the initialization of the instance,
    the function :func:`forward` for the forward propagation through the layer when called, and
    the last function :func:`kl_loss_term` which is called in the loss function as part of the
    regularization of the network.
    """

    def __init__(self, in_features, out_features, bias=True, weight_args=global_layer_default,
                 input_flag=False):
        """
        :param in_features:     (Required)
        :param out_features:    (Required)
        :param bias:            (Default  True)
        :param mean_mu:         (Default  0)
        :param mean_sigma:      (Default  0.1)
        :param sigma_min:       (Default -12)
        :param sigma_max:       (Default -2.2)
        :param mean_bias:       (Default  0.001)
        :param sigma_bias:      (Default  0.001)
        :param input_flag:      (Default  False)
        """
        super(VDP_FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.input_flag = input_flag

        self.mean = nn.Linear(in_features, out_features, bias)
        nn.init.normal_(self.mean.weight, mean=weight_args.fc_input_mean_mu, std=weight_args.fc_input_mean_sigma)
        self.mean.bias.data.fill_(weight_args.fc_input_mean_bias)

        self.sigma_weight = nn.Parameter(torch.zeros([1, out_features]), requires_grad=True)
        nn.init.uniform_(self.sigma_weight, a=weight_args.fc_input_sigma_min, b=weight_args.fc_input_sigma_max)
        self.sigma_bias = nn.Parameter(torch.ones([out_features]) * self.bias, requires_grad=True)

    def forward(self, mu, sigma=0):
        """
        Forward pass over the VDP 2D Fully Connected Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required if input_flag=False)
        :type  sigma:   Float
        """
        if self.input_flag:
            # Input Version
            mu_f = self.mean(mu)

            mu_pT_mu_p = torch.matmul(mu.unsqueeze(1), mu.unsqueeze(2)).squeeze(1)
            if self.bias:
                sigma_1 = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * mu_pT_mu_p.repeat(1, self.out_features)
            else:
                sigma_1 = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * mu_pT_mu_p.repeat(1, self.out_features)

            sigma_out = torch.diag_embed(sigma_1, dim1=1) + torch.diag(self.sigma_bias)

        else:
            # Mu Weight * Mu in
            mu_f = self.mean(mu)

            # Mean Weight^2 * sigma in
            muh_w = self.mean.weight.unsqueeze(0)
            muhT_sigmab_mu = torch.matmul(torch.matmul(muh_w, sigma), muh_w.permute([0, 2, 1]))

            tr_diag_sigma = torch.diagonal(sigma, dim1=1, dim2=2).sum(1).unsqueeze(1)

            mu_pT_mu_p = torch.matmul(mu.unsqueeze(1), mu.unsqueeze(2)).squeeze(1)

            # tr(Sigma Weights * Sigma In)
            # Mu in ^2 * Sigma Weights
            if self.bias:
                sigma_weight_out = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * (tr_diag_sigma.repeat(1, self.out_features) + mu_pT_mu_p.repeat(1, self.out_features))
            else:
                sigma_weight_out = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * (tr_diag_sigma.repeat(1, self.out_features) + mu_pT_mu_p.repeat(1, self.out_features))

            diag_sigma_weight_out = torch.diag_embed(sigma_weight_out, dim1=1)

            # tr(Sigma Weights * Sigma In) + Mean Weight^2 * sigma in + Mu in ^2 * Sigma Weights
            sigma_out = muhT_sigmab_mu + diag_sigma_weight_out + torch.diag(self.sigma_bias)

        return mu_f, sigma_out

    def kl_loss_term(self):
        """
        KL Loss term for the loss function
        """
        f_s = torch.log1p(torch.exp(self.sigma_weight))

        kl_loss = -0.5 * torch.mean((self.in_features * torch.log(f_s)) +
                                      self.in_features -
                                      torch.norm(self.mean.weight)**2 -
                                      (self.in_features * f_s))
        return kl_loss


class VDP_Softmax(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation Softmax Activation.
    This class contains the function :func:`__init__` for the initialization of the instance
    and the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, dim=1):
        """
        :param dim:     (Default  1)
        """
        super(VDP_Softmax, self).__init__()
        self.softmax = nn.Softmax(dim)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP 2D Softmax Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_y = self.softmax(mu)

        grad_f1 = torch.bmm(mu_y.unsqueeze(2), mu_y.unsqueeze(1))
        diag_f = torch.diag_embed(mu_y, dim1=1)
        grad_soft = diag_f - grad_f1
        sigma_y = torch.matmul(grad_soft, torch.matmul(sigma, grad_soft.permute(0, 2, 1)))
        return mu_y, sigma_y


class VDP_BatchNorm(nn.Module):
    """
    This class is for the instance creation of an Variational Density Propagation Softmax Activation.
    This class contains the function :func:`__init__` for the initialization of the instance
    and the function :func:`forward` for the forward propagation through the layer when called
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        """
        :param dim:     (Default  1)
        """
        super(VDP_BatchNorm, self).__init__()
        self.eps = eps
        self.batchNorm = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, mu, sigma):
        """
        Forward pass over the VDP 2D Softmax Layer

        :param mu:      Data Mean     (Required)
        :type  mu:      Float
        :param sigma:   Data Sigma    (Required)
        :type  sigma:   Float
        """
        mu_bn = self.batchNorm(mu)

        sigma_bn = torch.mul(sigma, 1 / (self.batchNorm.running_var + self.eps)) * self.batchNorm.weight + self.batchNorm.bias

        return mu_bn, sigma_bn

# class VDP_Skip(nn.Module):
#    """
#    This class is for the instance creation of an Variational Density Propagation Softmax Activation.
#    This class contains the function :func:`__init__` for the initialization of the instance
#    and the function :func:`forward` for the forward propagation through the layer when called
#    """
#
#    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
#        """
#        :param dim:     (Default  1)
#        """
#        super(VDP_BatchNorm, self).__init__()
#        self.eps = eps
#        self.batchNorm = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
#
#    def forward(self, mu, sigma):
#        """
#        Forward pass over the VDP 2D Softmax Layer
#
#        :param mu:      Data Mean     (Required)
#        :type  mu:      Float
#        :param sigma:   Data Sigma    (Required)
#        :type  sigma:   Float
#        """
#        mu_bn = self.batchNorm(mu)
#
#        sigma_bn = torch.mul(sigma, 1 / (self.batchNorm.running_var + self.eps)) * self.batchNorm.weight + self.batchNorm.bias
#
#        return mu_bn, sigma_bn
