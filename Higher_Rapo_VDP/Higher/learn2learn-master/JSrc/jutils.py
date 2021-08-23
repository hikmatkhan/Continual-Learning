import random
import numpy as np
import torch
import wandb


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


def init_wandb(args, model=None):
    wandb.init(project=args.wand_project, entity=args.username, reinit=True)
    if model != None:
        wandb.watch(model, log_freq=10)

