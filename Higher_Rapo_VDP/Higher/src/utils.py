import random
import learn2learn
import numpy as np
import torch
import torchvision
from learn2learn.data import TaskDataset
from learn2learn.data.transforms import NWays, KShots, LoadData

from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import transforms


def fix_seeds(seed=101):
    # No randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
    return seed


def get_compute_device():
    device = torch.device('cpu')
    if torch.cuda.device_count():
        device = torch.device('cuda')
    return device


def get_torch_ds(dataset= torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                 # torchvision.datasets.MNIST(root="data", train=True,
                 #        download=True, transform=transforms.Compose([transforms.ToTensor(),
                 #        transforms.Normalize((0.5,), (0.5,))])),
                 ways=10, shots=1, num_tasks=-1):
    '''
    # tasksets = utils.get_torch_ds(ways=args.ways, shots=args.shots, num_tasks=args.num_tasks)
    # for task in tasksets:
    #     X, y = task
    #     X, y = X.to(args.device), y.to(args.device)
    #     print("Y:", y)
    '''
    # MetaDataset MNIST or Custom Dataset
    dataset = learn2learn.data.MetaDataset(dataset)
    l2l_transforms = [
        NWays(dataset, n=ways),
        KShots(dataset, k=shots),
        LoadData(dataset),
    ]
    tasksets = TaskDataset(dataset, l2l_transforms, num_tasks=num_tasks)


    return tasksets


def get_l2l_ds(args):
    tasksets = learn2learn.vision.benchmarks.get_tasksets(args.dataset,
                                                          train_samples=args.shots*2,  # 2*shots,
                                                          train_ways=args.ways,
                                                          test_samples=args.shots*2,  # 2*shots,
                                                          test_ways=args.ways,
                                                          root=args.data,
                                                          num_tasks=args.total_num_tasks)
    return tasksets


def get_indices(X, args):
    # Separate data into Meta-Train/Meta-Test sets
    meta_train_indices = np.zeros(X.size(0), dtype=bool)
    # print("X:", X.size())
    meta_train_indices[np.arange(args.shots * args.ways) * 2] = True
    meta_test_indices = torch.from_numpy(~meta_train_indices)
    meta_train_indices = torch.from_numpy(meta_train_indices)
    return meta_train_indices, meta_test_indices


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

# def copy_models_weights(model):
#     deep_copy =
#     for name, param in model_1_params:
#         if name in model_2_params_dict:
#             model_2_params_dict[name].data.copy_(param.data)
from models.vanilla_net import VanillaNet
from models.lenet import LeNetwork
from models.resnet import ResNet18
from models.resnet import ResNet18S


def get_meta_theta(args):
    print(args.dataset)
    if args.dataset == "omniglot":
        if args.model_name == 'vanilla':
            return VanillaNet(args=args).to(args.device)
        elif args.model_name == 'lenet':
            return LeNetwork(out_dim=args.ways,
                                   in_channel=args.channels,
                                   img_sz=args.input_dim).to(args.device)
        elif args.model_name == 'resnet':
            return ResNet18S(out_dim=args.ways).to(args.device)

        else:
            print("Couldn't found appropriate model for Omniglot")
            return None

    elif args.dataset == 'mini-imagenet':
        if args.fine_tune:
            model = resnet18(pretrained=True).to(args.device)
            # if args.fine_tune:
            #     for param in model.parameters():
            #         param.requires_grad = False
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.ways).to(args.device)
            return model
        else:
            if args.model_name == 'lenet':
                return LeNetwork(out_dim=args.ways,
                                       in_channel=args.channels,
                                       img_sz=args.input_dim).to(args.device)
            elif args.model_name == 'resnet':
                 return ResNet18(out_dim=args.ways).to(args.device)
            else:
                print("Couldn't found appropriate model for Omniglot")

    elif args.dataset == 'cifarfs':
        if args.model_name == 'lenet':
            return LeNetwork(out_dim=args.ways,
                                   in_channel=args.channels,
                                   img_sz=args.input_dim).to(args.device)
        elif args.model_name == 'resnet':
            return ResNet18(out_dim=args.ways).to(args.device)
        else:
            print("Couldn't found appropriate model for Omniglot")
            return None

    elif args.dataset == 'tiered-imagenet':
        if args.model_name == 'lenet':
            return LeNetwork(out_dim=args.ways,
                                   in_channel=args.channels,
                                   img_sz=args.input_dim).to(args.device)
        elif args.model_name == 'resnet':
            return ResNet18(out_dim=args.ways).to(args.device)
        else:
            print("Couldn't found appropriate model for Omniglot")
            return None

    else:
        print("Dataset not found.")
        return None

    return None
