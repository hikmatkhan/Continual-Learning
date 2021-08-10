import random

import learn2learn
import numpy as np
import torch
import torchvision
from learn2learn.data import TaskDataset
from learn2learn.data.transforms import NWays, KShots, LoadData
from torchvision.transforms import transforms


def fix_seeds(seed=101):
    # No randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)


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
                 ways=10, shots=1, num_tasks=32):
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


def get_l2l_ds(dataset_name, data_path="./data", ways=1, shots=5, num_tasks=32):
    tasksets = learn2learn.vision.benchmarks.get_tasksets(dataset_name,
                                                          train_samples=shots*2,  # 2*shots,
                                                          train_ways=ways,
                                                          test_samples=shots*2,  # 2*shots,
                                                          test_ways=ways,
                                                          root=data_path,
                                                          num_tasks=num_tasks)
    return tasksets


def get_indices(X, args):
    # Separate data into Meta-Train/Meta-Test sets
    meta_train_indices = np.zeros(X.size(0), dtype=bool)
    # print("X:", X.size())
    meta_train_indices[np.arange(args.shots * args.ways) * 2] = True
    meta_test_indices = torch.from_numpy(~meta_train_indices)
    meta_train_indices = torch.from_numpy(meta_train_indices)
    return meta_train_indices, meta_test_indices

# def copy_models_weights(model):
#     deep_copy =
#     for name, param in model_1_params:
#         if name in model_2_params_dict:
#             model_2_params_dict[name].data.copy_(param.data)
