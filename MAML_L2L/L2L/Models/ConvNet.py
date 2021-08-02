from typing import Any

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD


class ConvNet(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 kernel_size=3,
                 ways=5):
        super().__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=kernel_size, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(out_channels * out_channels, ways)
        )

    def linear_head(self):
        return self.linear_layers

    def conv(self):
        return self.cnn_layers

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

if __name__ == '__main__':
    convNet = ConvNet()
    print(convNet)