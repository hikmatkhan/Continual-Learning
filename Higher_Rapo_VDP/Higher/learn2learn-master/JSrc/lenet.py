import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz // 4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(500, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class Yann_LeNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=3):
        super(Yann_LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=6,
                               kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, out_dim)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        # print("X", x.shape)
        x = self.linear1(x)
        x = self.tanh(x)
        # print("X", x.shape)
        x = self.linear2(x)
        return x


def LeNetwork(out_dim=10, in_channel=3, img_sz=32):  # LeNet with color input
    return LeNet(out_dim=out_dim, in_channel=in_channel, img_sz=img_sz)


def Get_Yann_LeNet(out_dim=10, in_channel=3 ):
    return Yann_LeNet(out_dim=out_dim, in_channel=in_channel)
