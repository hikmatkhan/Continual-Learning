import argparse
import time

import wandb
import random
import numpy as np
import torch
import jutils

import argparse
import wandb
import random
import numpy as np
import torch
import random
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# parser = argparse.ArgumentParser('MAML with Higher')

# Setup save
startTime = time.time()
start_date_str = time.strftime("%m_%d_%y", time.localtime(startTime))
start_time_str = time.strftime("%H_%M", time.localtime(startTime))

parser = argparse.ArgumentParser(prog="VDP")

parser.add_argument('--network', type=str, default='MNIST_FC',
                    choices=['VDP', 'FMNIST_CONV', 'MNIST_FC',
                             'MNIST_CONV', 'CIFAR10_CONV', 'MNIST_FC_BBB'], help='Dataset Experiment')

# Load Model
parser.add_argument('--load_model', type=str, default='', help='Path to a previously trained model checkpoint')
parser.add_argument('--continue_train', type=bool, default=False, help='Continue to train this checkpoint ')
# Print Parameters
parser.add_argument('--lightning', type=bool, default=False, help='Run with PyTorch Lightning')
parser.add_argument('--kdes', type=bool, default=False, help='Plot KDE Plots')
# parser.add_argument('--base_save', type=str, default='../models/network/' + start_date_str + '/' + start_time_str + '/',
#                     help='Where to Save model, network gets replaced with the experiment you run')
# Code Parameters
parser.add_argument('--num_workers', type=int, default=6, help='Number of CPU works to process dataset')
parser.add_argument('--data_path', type=str, default='../../data/', help='Path to save dataset data')
# Training Parameters
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=120, help='Batch size')
# Loss Function Parameters
parser.add_argument('--tau', type=float, default=0.002, help='KL Weight Term')
parser.add_argument('--clamp', type=float, default=1000, help='Clamping')
parser.add_argument('--var-sup', type=float, default=0.001, help='Loss Variance Bias')
# Learning Rate Parameters
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
# parser.add_argument('--lr_sched', type=float, default=[100, 150], help='Learning Rate Scheduler Milestones')
# parser.add_argument('--gamma', type=float, default=0.1, help='Learning Rate Gamma')
# SWA
# parser.add_argument('--swa_start', type=int, default=0, help='SWA Starting Epoch')
# parser.add_argument('--swa_lr', type=int, default=0.01, help='SWA Learning Rate')
# parser.add_argument('--swa_anneal', type=int, default=20, help='SWA Learning Rate')
# BBB
parser.add_argument('--samples', type=int, default=10, help='Samples')
parser.add_argument('--rho', default='-3', type=float, help='Initial rho')
parser.add_argument('--sig1', default='0.0', type=float, help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sig2', default='6.0', type=float, help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi', default='0.25', type=float, help='weighting factor for prior')
# Layer Weight Initialization Parameters
parser.add_argument('--conv_input_mean_mu', type=float, default=0, help='Input Mean Weight Normal Distr. Mean')
parser.add_argument('--conv_input_mean_sigma', type=float, default=0.1, help='Input Mean Weight Normal Distr. Sigma')
parser.add_argument('--conv_input_mean_bias', type=float, default=0.0001, help='Input Mean Bias')
parser.add_argument('--conv_input_sigma_min', type=float, default=-12, help='Input Sigma Weight Uniform Distr. Min')
parser.add_argument('--conv_input_sigma_max', type=float, default=-2.2, help='Input Sigma Weight Uniform Distr. Max')
parser.add_argument('--conv_input_sigma_bias', type=float, default=0.0001, help='Input Sigma Bias')

parser.add_argument('--fc_input_mean_mu', type=float, default=0, help='Input Mean Weight Normal Distr. Mean')
parser.add_argument('--fc_input_mean_sigma', type=float, default=0.1, help='Input Mean Weight Normal Distr. Sigma')
parser.add_argument('--fc_input_mean_bias', type=float, default=0.0001, help='Input Mean Bias')
parser.add_argument('--fc_input_sigma_min', type=float, default=-12, help='Input Sigma Weight Uniform Distr. Min')
parser.add_argument('--fc_input_sigma_max', type=float, default=-2.2, help='Input Sigma Weight Uniform Distr. Max')
parser.add_argument('--fc_input_sigma_bias', type=float, default=0.0001, help='Input Sigma Bias')

parser.add_argument('--drop_out', type=float, default=0.1, help='Network Droput')

parser.add_argument('--out_shape', type=int, default=None, help='will autofill')

# General
general_params = parser.add_argument_group('General')
general_params.add_argument('--data', type=str, default="~/data",
                            help='Path to the folder the data is downloaded to.')
general_params.add_argument('--dataset', type=str,
                            choices=["omniglot", "mini-imagenet", "fc100",
                                     "cifarfs", "tiered-imagenet"], default='omniglot',
                            help='Name of the dataset (default: omniglot).')
# Meta Learning Params
meta_params = parser.add_argument_group('Meta Learning Parameters')
meta_params.add_argument('--ways', type=int, default=5,
                         help='Number of classes per task (N in "N-way", default: 5).')
meta_params.add_argument('--shots', type=int, default=1,
                         help='Number of training example per class (k in "k-shot", default: 5).')
meta_params.add_argument('--adaptation-steps', type=int, default=1,
                         help='Number of adaptation steps on meta-train datasets.')
meta_params.add_argument('--num-tasks', type=int, default=32,
                         help='Number of tasks to sample from task distribution. (Meta batch size)')
meta_params.add_argument('--total-num-tasks', type=int, default=20000,
                         help='Total number of tasks in task distribution. Always keep it to -1.')
# meta_params.add_argument('--first-order', action='store_true',
#                          help='Use the first order approximation, do not use highers-order '
#                               'derivatives during meta-optimization.')
meta_params.add_argument('--meta-lr', type=float, default=0.001,
                         help='Learning rate for the meta-optimizer (optimization of the outer '
                              'loss). The default optimizer is Adam (default: 1e-3).')
meta_params.add_argument('--fast-lr', type=float, default=0.01,
                         help='Learning rate for the meta-optimizer (optimization of the outer '
                              'loss). The default optimizer is Adam (default: 1e-3).')
meta_params.add_argument('--meta-learn', type=int, default=1,
                         help='Set this to False only for debugging purpose to'
                              'verify that meta learning is happening')

# Model
model_params = parser.add_argument_group('Model')
model_params.add_argument('--input', type=int, default=28,
                          help='Input to dimension')
model_params.add_argument('--channels', type=int, default=1,
                          help='Input to dimension')
model_params.add_argument('--fine-tune', type=int, default=0,
                          help='Only meta learn the FC layer')
import datetime

model_params.add_argument('--timestamp', type=str, default="{0}.pt".format(datetime.datetime.now()),
                          help='Save best model with this timestamp.')

model_params.add_argument('--snr-min-std', type=int, default=0.1,
                          help='Std maximium.')
model_params.add_argument('--snr-max-std', type=int, default=2,
                          help='Std minimium.')
model_params.add_argument('--snr-steps', type=int, default=0.1,
                          help='Different values for SNR')

# Optimization
optim_params = parser.add_argument_group('Optimization')
optim_params.add_argument('--epochs', type=int, default=10000,
                          help='Number of epochs of meta-training (default: 50000).')
optim_params.add_argument('--seed', type=int, default=jutils.fix_seeds(),
                          help='Number of epochs of meta-training (default: 101).')

# Misc
misc = parser.add_argument_group('Misc')
misc.add_argument('--workers', type=int, default=4,
                  help='Number of workers to use for data-loading (default: 1).')
misc.add_argument('--device', type=str, default=jutils.get_compute_device(),
                  help="Compute device information")
misc.add_argument('--cuda', type=int, default=1,
                  help="Use GPU if 1 otherwise not")

# Visualization
viz = parser.add_argument_group('Wandb')
viz.add_argument('--wand-project', type=str, default="V2+VDP+L2L+FC+Omniglot",
                 help='Wandb project name should go here')
viz.add_argument('--username', type=str, default="hikmatkhan-",
                 help='Wandb username should go here')
viz.add_argument('--wandb-log', type=int, default=1,
                 help='If True, Logs will be reported on wandb.')

viz.add_argument('--vdp', type=str, default="V2+FC+VDP+Omniglot",
                 help='VDP + Dataset information')

viz.add_argument('--gpu', type=str, default="2",
                 help='VDP + Dataset information')

args = parser.parse_args()


# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, is_eval_on_noisy=False, mu=0, std=0.01):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        # train_error = loss(learner(adaptation_data), adaptation_labels)
        mu_y_out, sigma_y_out = learner(adaptation_data)
        prd_loss = loss(mu_y_out, adaptation_labels)
        labels = nn.functional.one_hot(adaptation_labels, list(learner.module.children())[-2].out_features)
        # loss = model.batch_loss(mu_y_out, sigma_y_out, labels)
        b_loss = learner.batch_loss(mu_y_out, sigma_y_out, labels)
        learner.adapt(prd_loss + b_loss)

    valid_accuracy = -1
    valid_error = -1
    if is_eval_on_noisy is False:
        # Evaluate the adapted model
        # predictions = learner(evaluation_data)
        mu_y_out, sigma_y_out = learner(evaluation_data)
        # valid_error = loss(mu_y_out, evaluation_labels)
        prd_loss = loss(mu_y_out, evaluation_labels)
        labels = nn.functional.one_hot(evaluation_labels, list(learner.module.children())[-2].out_features)
        b_loss = learner.batch_loss(mu_y_out, sigma_y_out, labels)
        valid_accuracy = accuracy(mu_y_out, evaluation_labels)
        valid_error = (b_loss + prd_loss)

    noisy_valid_error = -1
    noisy_valid_accuracy = -1
    if is_eval_on_noisy is True:
        noisy_evaluation_data = evaluation_data + torch.normal(mean=mu, std=std, size=evaluation_data.shape).to(
            args.device)
        # np.random.normal(loc=mu, scale=std, size=evaluation_data.shape)
        noisy_evaluation_data = noisy_evaluation_data.to(args.device)
        mu_y_out, sigma_y_out = learner(noisy_evaluation_data)
        # valid_error = loss(mu_y_out, evaluation_labels)
        noisy_prd_loss = loss(mu_y_out, evaluation_labels)
        labels = nn.functional.one_hot(evaluation_labels, list(learner.module.children())[-2].out_features)
        b_loss = learner.batch_loss(mu_y_out, sigma_y_out, labels)
        noisy_valid_accuracy = accuracy(mu_y_out, evaluation_labels)
        noisy_valid_error = (b_loss + noisy_prd_loss)

    return valid_error, valid_accuracy, noisy_valid_error, noisy_valid_accuracy


def main(args):
    jutils.init_wandb(args)
    jutils.fix_seeds()

    # Load train/validation/test tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(args.dataset,
                                                  train_ways=args.ways,
                                                  train_samples=2 * args.shots,
                                                  test_ways=args.ways,
                                                  test_samples=2 * args.shots,
                                                  num_tasks=args.total_num_tasks,
                                                  root=args.data,
                                                  )

    ##### Create model
    ##### model = l2l.vision.models.OmniglotFC(28 ** 2, args.ways)

    from VDPNet import Net
    model = Net(args).to(args.device)
    print("model:", model)
    # model.load_state_dict(torch.load("./best_models/2021-08-30 11:19:03.139908.pt")["model_state_dict"])
    # print("Model loaded.")
    # model.to(args.device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # jutils.save_best_model(model=model, optimizer=opt, epoch=0, val_loss=0, val_acc=0, path=args.timestamp)
    # model.load_state_dict(torch.load("model.pt")["model_state_dict"])
    best_meta_valid_accuracy = 0
    best_meta_test_accuracy = 0
    best_meta_train_accuracy = 0
    if args.wandb_log:
        wandb.log({"best_meta_valid_accuracy": best_meta_valid_accuracy,
                   "best_meta_train_accuracy": best_meta_train_accuracy,
                   "best_model_dict": args.timestamp,
                   "best_meta_test_accuracy": best_meta_test_accuracy})

    # print("Model loaded.")
    for iteration in range(args.epochs):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(args.num_tasks):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy, _, _ = fast_adapt(batch,
                                                                     learner,
                                                                     loss,
                                                                     args.adaptation_steps,
                                                                     args.shots,
                                                                     args.ways,
                                                                     args.device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy, _, _ = fast_adapt(batch,
                                                                     learner,
                                                                     loss,
                                                                     args.adaptation_steps,
                                                                     args.shots,
                                                                     args.ways,
                                                                     args.device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        meta_train_error = meta_train_error / args.num_tasks
        meta_train_accuracy = meta_train_accuracy / args.num_tasks
        meta_valid_error = meta_valid_error / args.num_tasks
        meta_valid_accuracy = meta_valid_accuracy / args.num_tasks


        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / args.num_tasks)
        opt.step()

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        # noisy_meta_test_error = 0.0
        # noisy_meta_test_accuracy = 0.0
        noisy_meta_test_errors = {}
        noisy_meta_test_accuracies = {}
        for std in np.arange(args.snr_min_std, args.snr_max_std, args.snr_steps):
            noisy_meta_test_accuracies["{0}".format(round(std, 3))] = 0
            noisy_meta_test_errors["{0}".format(round(std, 3))] = 0
            # print("Key:", round(std, 3))
        for task in range(args.num_tasks):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = tasksets.test.sample()

            evaluation_error, evaluation_accuracy, _, _ = fast_adapt(batch,
                                                                     learner,
                                                                     loss,
                                                                     args.adaptation_steps,
                                                                     args.shots,
                                                                     args.ways,
                                                                     args.device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

            for std in np.arange(args.snr_min_std, args.snr_max_std, args.snr_steps):
                # print("******", round(std, 3), "******")
                _, _, noisy_evaluation_error, noisy_evaluation_accuracy = fast_adapt(batch,
                                                                                     learner,
                                                                                     loss,
                                                                                     args.adaptation_steps,
                                                                                     args.shots,
                                                                                     args.ways,
                                                                                     args.device,
                                                                                     is_eval_on_noisy=True, mu=0,
                                                                                     std=std)

                noisy_meta_test_accuracies["{0}".format(round(std, 3))] += noisy_evaluation_accuracy.item()
                noisy_meta_test_errors["{0}".format(round(std, 3))] += noisy_evaluation_error.item()

        meta_test_error = meta_test_error / args.num_tasks
        meta_test_accuracy = meta_test_accuracy / args.num_tasks
        for std in np.arange(args.snr_min_std, args.snr_max_std, args.snr_steps):
            # print("K:", std)
            noisy_meta_test_errors["{0}".format(round(std, 3))] /= args.num_tasks
            noisy_meta_test_accuracies["{0}".format(round(std, 3))] /= args.num_tasks

        # noisy_meta_test_error = noisy_meta_test_error / args.num_tasks
        # noisy_meta_test_accuracy = noisy_meta_test_accuracy / args.num_tasks

        print(
            "E|{0}| Acc| Train: {1} Val:{2} Test:{3} R-Test:{7}|   Loss| Train:{4} Val:{5} Test:{6} R-Test:{8} |".format(
                iteration,
                round(meta_train_accuracy, 2), round(meta_valid_accuracy, 2), round(meta_test_accuracy, 2),
                round(meta_train_error, 2), round(meta_valid_error, 2), round(meta_test_error, 2),
                round(noisy_meta_test_accuracies["{0}".format(round(args.snr_min_std, 3))], 2),
                round(noisy_meta_test_errors["{0}".format(round(args.snr_min_std, 3))], 2)))
        # print("noisy_meta_test_accuracies:", noisy_meta_test_accuracies)

        if best_meta_valid_accuracy < meta_valid_accuracy and meta_valid_accuracy > 0.6:
            print("Best Val Acc:", best_meta_valid_accuracy, " Obtained Val Acc:", meta_valid_accuracy)
            best_meta_valid_accuracy = meta_valid_accuracy
            jutils.save_best_model(model=maml.module,
                                   epoch=iteration,
                                   meta_train_accuracy=meta_train_accuracy,
                                   meta_valid_accuracy=meta_valid_accuracy,
                                   meta_test_accuracy=meta_test_accuracy,
                                   meta_train_error=meta_train_error,
                                   meta_valid_error=meta_valid_error,
                                   meta_test_error=meta_test_error,
                                   optimizer=opt, path=args.timestamp,
                                   root_folder="vdp_models")
        if best_meta_test_accuracy < meta_test_accuracy:
            best_meta_test_accuracy = meta_test_accuracy
        if best_meta_train_accuracy < meta_train_accuracy:
            best_meta_train_accuracy = meta_train_accuracy
        if ((100 <= iteration <= 150) or (2100 <= iteration <= 2150)) and meta_valid_accuracy < 0.21:
            print("Val-Acc Not improving:", meta_valid_accuracy)
            if args.wandb_log:
                wandb.log({"meta_train_accuracy": meta_train_accuracy * -1,
                           "meta_train_loss": meta_train_error * -1,
                           "meta_valid_accuracy": meta_valid_accuracy * -1,
                           "meta_valid_loss": meta_valid_error * -1,
                           "meta_test_accuracy": meta_test_accuracy * -1,
                           "meta_test_loss": meta_test_error * -1,
                           "best_meta_valid_accuracy": best_meta_valid_accuracy * -1,
                           "best_meta_test_accuracy": best_meta_test_accuracy * -1,
                           "best_meta_train_accuracy": best_meta_train_accuracy * -1,
                           # "noisy_meta_test_accuracy": noisy_meta_test_accuracies["0.1"] * -1,
                           # "noisy_meta_test_error": noisy_meta_test_errors["0.1"] * -1

                           })
            return

        if args.wandb_log:
            wandb.log({"meta_train_accuracy": meta_train_accuracy,
                       "meta_train_loss": meta_train_error,
                       "meta_valid_accuracy": meta_valid_accuracy,
                       "meta_valid_loss": meta_valid_error,
                       "meta_test_accuracy": meta_test_accuracy,
                       "meta_test_loss": meta_test_error,
                       "best_meta_valid_accuracy": best_meta_valid_accuracy,
                       "best_meta_test_accuracy": best_meta_test_accuracy,
                       "best_meta_train_accuracy": best_meta_train_accuracy,

                       })
            for std in np.arange(args.snr_min_std, args.snr_max_std, args.snr_steps):
                wandb.log({
                "noisy_meta_test_accuracy_std_{0}".format(round(std, 3)): noisy_meta_test_accuracies["{0}".format(round(std, 3))],
                "noisy_meta_test_error_std_{0}".format(round(std, 3)): noisy_meta_test_errors["{0}".format(round(std, 3))]})


if __name__ == '__main__':
    print("CMD Arguments:", args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
