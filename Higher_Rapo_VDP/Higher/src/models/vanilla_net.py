import torch.nn as nn


# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def VanillaNet(args):
    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by highers later.
    # Before highers, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    net = nn.Sequential(
        nn.Conv2d(args.channels, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        Flatten(),
        nn.Linear(64, args.ways)).to(args.device)

    return net