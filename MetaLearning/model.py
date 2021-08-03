import torch
import torchsummary
from torch import nn
from models import mlp
from torch import optim
import copy
from torchvision.models import resnet18


# class FV(nn.Module):
#     def __init__(self):
#         super(FV, self).__init__()
#
#     def forward(self, x):
#         return x


class model(nn.Module):

    def __init__(self, args):
        super(model, self).__init__()
        # self.module = mlp.MLP100().to(args.device)
        # torchsummary.summary(
        #     self.module, input_size=(1, 32 * 32))
        self.module = resnet18(pretrained=True).to(args.device)
        # print(self.module)
        # self.module.fc = FV()

        # torchsummary.summary(self.module, input_size=(3, 64, 64))
        # print(self.module)
        if args.fine_tune:
            for param in self.module.parameters():
                param.requires_grad = False

        self.module.fc = nn.Linear(self.module.fc.in_features, args.ways).to(args.device)
        # torchsummary.summary(self.module, input_size=(3, 64, 64))
        # print(self.module)
        self.optm = optim.Adam(self.module.parameters(), lr=args.fast_lr)
        self.loss = nn.CrossEntropyLoss()

    def init_meta_params(self, meta_params):
        return

    def forward(self, x):
        return self.module(x)

    def adapt(self, loss):
        print("Adapt parameters")
        self.optm.zero_grad()
        loss.backward()
        self.optm.step()

    def clone(self):
        # print("clone parameters")
        return copy.deepcopy(self.module)


# if __name__ == '__main__':
#     maml = MAML()
#     maml.adapt()
#     maml.clone()
#     print(maml)
