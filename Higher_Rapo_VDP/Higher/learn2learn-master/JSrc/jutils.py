import os
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


def save_best_model(model, epoch,
                    meta_train_accuracy, meta_valid_accuracy, meta_test_accuracy,
                    meta_train_error, meta_valid_error, meta_test_error, optimizer, path):

    if not os.path.exists('best_models'):
        os.makedirs('best_models')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "meta_train_accuracy": meta_train_accuracy,
        "meta_train_loss": meta_train_error,
        "meta_valid_accuracy": meta_valid_accuracy,
        "meta_valid_loss": meta_valid_error,
        "meta_test_accuracy": meta_test_accuracy,
        "meta_test_loss": meta_test_error},
        "best_models/{0}".format(path))  # "model.pt")
    print("Best model saved.")

# def load_best_model(path):
#     loaded_model = torch.load(path)
#     return (loaded_model["model"], loaded_model["epoch"], loaded_model["val_loss"],
#             loaded_model["val_acc"], loaded_model["optimizer"], loaded_model["path"])
