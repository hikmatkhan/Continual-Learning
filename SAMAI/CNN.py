import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
print(torch.utils)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(57600, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)



    def forward(self, x):
        print("X:", x.shape)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        x = x.view(-1, self.num_flat_features(x))
        print("X:::", x.shape)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#Define the network
model = CNN()
model.zero_grad()

x = torch.randn(1, 1, 64, 64)
predictions = model(x)

#Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])


# Dataset
trainset = torchvision.datasets.MNIST("./data", train=True, transform=transforms, download=True)
testset = torchvision.datasets.MNIST("./data", train=False, transform=transforms, download=True)

trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

print("TrainLoader:", len(trainloader))
print("TestLoader:", len(testloader))
# dataiter = iter(trainloader)
# dataiter.next()

for i, (data, target) in enumerate(trainloader):
    print(data.shape)


# trainloader = torchvision.utils.data.Dataloader()

print("Loss:", F.mse_loss(predictions, predictions))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
    
    def __getitem__(self, item):
        super(CustomDataset, self).__getitem__()
        
    def __len__(self):
        super(CustomDataset, self).__len__()