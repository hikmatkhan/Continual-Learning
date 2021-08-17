from torch import nn
from torchvision.models import resnet18


def ResNet18(output_classes, fine_tune=False):
    model = resnet18(pretrained=True)
    # print(self.module)
    # self.module.fc = FV()

    # torchsummary.summary(self.module, input_size=(3, 64, 64))
    # print(self.module)
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, output_classes)
    return model
