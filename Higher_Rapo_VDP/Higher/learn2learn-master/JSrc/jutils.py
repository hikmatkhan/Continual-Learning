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


def add_gaussian_noise(X, mu=0, std=0.01):
    # noise = np.random.normal(loc=mu, scale=std, size=X.shape)  # (5, 1, 28,28))
    # print("Noise:", noise.shape)
    # noisy_x = X + noise
    # print("Noisy X:", noisy_x.shape)
    return X + np.random.normal(loc=mu, scale=std, size=X.shape)


def init_wandb(args, model=None):
    wandb.init(project=args.wand_project, entity=args.username, reinit=True)
    if model != None:
        wandb.watch(model, log_freq=10)


def save_best_model(model, epoch,
                    meta_train_accuracy, meta_valid_accuracy, meta_test_accuracy,
                    meta_train_error, meta_valid_error, meta_test_error, optimizer, path, root_folder):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

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
        "{0}/{1}".format(root_folder, path))  # "model.pt")
    print("Best model saved.")

# def load_best_model(path):
#     loaded_model = torch.load(path)
#     return (loaded_model["model"], loaded_model["epoch"], loaded_model["val_loss"],
#             loaded_model["val_acc"], loaded_model["optimizer"], loaded_model["path"])
