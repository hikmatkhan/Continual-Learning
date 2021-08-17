import torch
import numpy as np
import torch.nn as nn
from Comps.FC import BayesianLinear


class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        self.samples = args.samples
        self.devices = args.devices
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.fc1 = BayesianLinear(784, 1200, args)
        self.fc2 = BayesianLinear(1200, 10, args)

    def forward(self, x, sample=False):
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x, sample))
        y = torch.nn.functional.log_softmax(self.fc2(x, sample), dim=1)
        return y

    def batch_loss(self, predictions, lps, lvps, num_batches, target):
        # hack
        w1 = 1.e-3
        w2 = 1.e-3
        w3 = 5.e-2

        samp = torch.randint(low=0, high=10, size=(1,))

        outputs = torch.stack([predictions[samp]], dim=0).to(device=self.devices)
        log_var = w1 * torch.as_tensor(lvps, device=self.devices).mean()
        log_p = w2 * torch.as_tensor(lps, device=self.devices).mean()
        nll = w3 * torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.devices)

        return (log_var - log_p) / num_batches + nll