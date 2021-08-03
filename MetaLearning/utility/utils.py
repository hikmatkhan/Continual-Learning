import random

import learn2learn
import numpy as np
import torch
from learn2learn.data import TaskDataset
from learn2learn.data.transforms import NWays, KShots, LoadData


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


def get_torch_ds(dataset, ways=1, shots=5, num_tasks=5):
    # MetaDataset MNIST or Custom Dataset
    dataset = learn2learn.data.MetaDataset(dataset)
    l2l_transforms = [
        NWays(dataset, n=ways),
        KShots(dataset, k=shots),
        LoadData(dataset),
    ]
    tasksets = TaskDataset(dataset, l2l_transforms, num_tasks=num_tasks)

    # # MetaDataset Sampling
    # for task in tasksets:
    #     X, y = task
    #     print(y)

    return tasksets


def get_l2l_ds(dataset_name, ways=1, shots=5, data_path="./data"):
    tasksets = learn2learn.vision.benchmarks.get_tasksets(dataset_name,
                                                          train_samples=shots,  # 2*shots,
                                                          train_ways=ways,
                                                          test_samples=shots,  # 2*shots,
                                                          test_ways=ways,
                                                          root=data_path)
    return tasksets

# def copy_models_weights(model):
#     deep_copy =
#     for name, param in model_1_params:
#         if name in model_2_params_dict:
#             model_2_params_dict[name].data.copy_(param.data)
