import argparse

import higher
import numpy as np
import torch

from MetaLearning.maml import Maml
from MetaLearning.MetaWithHigher.models import LeNetwork
from MetaLearning.MetaWithHigher import utils
import torch.optim as optim
import torch.nn.functional as F
import wandb

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser('MAML')

# General
general_params = parser.add_argument_group('General')
general_params.add_argument('--data', type=str, default="./data",
                            help='Path to the folder the data is downloaded to.')
general_params.add_argument('--dataset', type=str,
                            choices=["omniglot", "mini-imagenet", "fc100",
                                     "cifarfs", "tiered-imagenet"], default='omniglot',
                            help='Name of the dataset (default: omniglot).')

general_params.add_argument('--ways', type=int, default=5,
                            help='Number of classes per task (N in "N-way", default: 5).')
general_params.add_argument('--shots', type=int, default=1,
                            help='Number of training example per class (k in "k-shot", default: 5).')
general_params.add_argument('--adaptation-steps', type=int, default=5,
                            help='Number of adaptation steps on meta-train datasets.')

# general_params.add_argument('--num-shots-test', type=int, default=15,
#                             help='Number of test example per class. If negative, same as the number '
#                                  'of training examples `--num-shots` (default: 15).')

# Model
model_params = parser.add_argument_group('Model')
model_params.add_argument('--fine-tune', type=bool, default=False,
                          help='Only meta learn the FC layer')
model_params.add_argument('--input', type=int, default=32,
                          help='Input to dimension')

general_params.add_argument('--model', type=str,
                            choices=['maml'], default='maml',
                            help='Name of the Algorithm (default: maml).')

# Optimization
optim_params = parser.add_argument_group('Optimization')

# optim_params.add_argument('--batch-size', type=int, default=25,
#                           help='Number of tasks in a batch of tasks (default: 25).')
optim_params.add_argument('--num-steps', type=int, default=1,
                          help='Number of fast adaptation steps, ie. gradient descent '
                               'updates (default: 1).')
optim_params.add_argument('--num-epochs', type=int, default=50000,
                          help='Number of epochs of meta-training (default: 50).')
# optim_params.add_argument('--num-batches', type=int, default=100,
#                           help='Number of batch of tasks per epoch (default: 100).')
optim_params.add_argument('--num-tasks', type=int, default=32,
                          help='Number of tasks to sample from task distribution.')
# optim_params.add_argument('--step-size', type=float, default=0.1,
#                           help='Size of the fast adaptation step, ie. learning rate in the '
#                                'gradient descent update (default: 0.1).')
optim_params.add_argument('--first-order', action='store_true',
                          help='Use the first order approximation, do not use highers-order '
                               'derivatives during meta-optimization.')
optim_params.add_argument('--meta-lr', type=float, default=0.001,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')
optim_params.add_argument('--fast-lr', type=float, default=0.1,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')


# Misc
misc = parser.add_argument_group('Misc')
misc.add_argument('--num-workers', type=int, default=1,
                  help='Number of workers to use for data-loading (default: 1).')
misc.add_argument('--wand_project', type=str, default="Github_Rapo",
                  help='Wandb project name should go here')
misc.add_argument('--username', type=str, default="hikmatkhan",
                  help='Wandb username should go here')
misc.add_argument('--wandb_logging', type=bool, default=True,
                  help='If True, Logs will be reported on wandb.')
misc.add_argument('--verbose', action='store_true')
misc.add_argument('--use-cuda', action='store_true')
misc.add_argument('--device', type=str, default=utils.get_compute_device(), help="Compute device information")
args = parser.parse_args()


# TODO
# 1. Load datasets
# 2. Define adapted NN
# 3. Define MetaLearner

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def train_on_task(theta_pi, diff_optim,  tasksets):
    # Meta training set
    X, y = tasksets.train.sample()
    X, y = X.to(args.device), y.to(args.device)
    # adapt_optimizer = optim.SGD(meta_theta.parameters(),
    #                             lr=0.1, momentum=0.9)#Adam(meta_theta.parameters(), lr=args.fast_lr)

    meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
    # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
    # with highers.innerloop_ctx(meta_theta, adapt_optimizer,
    #                           copy_initial_weights=False,
    #                           override={'lr': torch.tensor([args.fast_lr],
    #                                                        requires_grad=True).to(args.device)}) as (theta_pi, diff_optim):
    #     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    spt_mean_loss = 0
    spt_mean_acc = 0
    for step in range(args.adaptation_steps):
        y_spt = theta_pi(X[meta_train_indices])
        spt_mean_acc += accuracy(y_spt, y[meta_train_indices])
        spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
        spt_mean_loss += spt_loss.detach().item()
        diff_optim.step(spt_loss)  # After this call. There will be next version of the theta

    spt_mean_loss /= args.adaptation_steps
    spt_mean_acc /= args.adaptation_steps
    y_qry = theta_pi(X[meta_test_indices])
    qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
    qry_loss.backward()

    return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, accuracy(y_qry, y[meta_test_indices])


def val_on_task(theta_pi, diff_optim, tasksets):
    # Meta training set
    X, y = tasksets.validation.sample()
    X, y = X.to(args.device), y.to(args.device)
    # adapt_optimizer = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)

    meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
    # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
    # with highers.innerloop_ctx(meta_theta, adapt_optimizer,
    #                           track_higher_grads=False, override={'lr': torch.tensor([args.fast_lr],
    #                                                                                  requires_grad=True).to(args.device)}
    #                           ) as (theta_pi, diff_optim):
        # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    spt_mean_loss = 0
    spt_mean_acc = 0
    for step in range(args.adaptation_steps):
        y_spt = theta_pi(X[meta_train_indices])
        spt_mean_acc += accuracy(y_spt, y[meta_train_indices])
        spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
        spt_mean_loss += spt_loss.detach().item()
        diff_optim.step(spt_loss)  # After this call. There will be next version of the theta

    spt_mean_loss /= args.adaptation_steps
    spt_mean_acc /= args.adaptation_steps
    y_qry = theta_pi(X[meta_test_indices])
    qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
        # qry_loss.backward()
    # meta_optimizer.step()
    return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, accuracy(y_qry, y[meta_test_indices])


def adapt_on_task(theta_pi, diff_optim, tasksets):
    # Meta test set (Adaptation Phase)
    X, y = tasksets.test.sample()
    X, y = X.to(args.device), y.to(args.device)
    # adapt_optimizer = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)

    meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
    # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
    # with highers.innerloop_ctx(meta_theta, adapt_optimizer,
    #                           track_higher_grads=False, override={'lr': torch.tensor([args.fast_lr],
    #                                                                                  requires_grad=True).to(args.device)}) as (
    #         theta_pi, diff_optim):
    # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    spt_mean_loss = 0
    spt_mean_acc = 0
    for step in range(args.adaptation_steps):
        y_spt = theta_pi(X[meta_train_indices])
        spt_mean_acc += accuracy(y_spt, y[meta_train_indices])
        spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
        spt_mean_loss += spt_loss.detach().item()
        diff_optim.step(spt_loss)  # After this call. There will be next version of the theta

    spt_mean_loss /= args.adaptation_steps
    spt_mean_acc /= args.adaptation_steps
    y_qry = theta_pi(X[meta_test_indices])
    qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
    # qry_loss.backward()
    # meta_optimizer.step()
    return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, accuracy(y_qry, y[meta_test_indices])


'''
Meta Learning Loop
'''
if __name__ == '__main__':

    print(args)
    if args.wandb_logging:
        wandb.init(project="No_Meta_Omni_5_Way_5_Shots", entity="hikmatkhan-")

    # meta_theta = maml(args=args).to(args.device)
    train_indices = np.zeros(args.ways * args.shots, dtype=bool)
    train_indices[np.arange(args.ways * args.shots)] = True
    train_indices = torch.from_numpy(train_indices)

    val_indices = np.zeros(args.ways * args.shots, dtype=bool)
    val_indices[np.arange(args.ways * args.shots)] = True
    val_indices = torch.from_numpy(val_indices)

    test_indices = np.zeros(args.ways * args.shots, dtype=bool)
    test_indices[np.arange(args.ways * args.shots)] = True
    test_indices = torch.from_numpy(test_indices)

    # tasksets = utils.get_torch_ds(ways=args.ways, shots=args.shots, num_tasks=args.num_tasks)
    tasksets = utils.get_l2l_ds(args.dataset, data_path=args.data, ways=args.ways,
                                shots=args.shots, num_tasks=args.num_tasks)

    # Outer Optimization
    # meta_theta = Maml(model=ResNet18(output_classes=args.ways), args=args).to(args.device)
    meta_theta = Maml(model=LeNetwork(out_dim=args.ways, in_channel=1, img_sz=28), args=args).to(args.device)
    meta_optimizer = optim.Adam(meta_theta.parameters(), lr=args.meta_lr)
    # meta_train_loss = []
    # adaptation_loss = []
    for epoch in range(0, args.num_epochs):
        meta_train_spt_loss = []
        meta_train_qry_loss = []
        meta_eval_spt_loss = []
        meta_eval_qry_loss = []
        meta_test_spt_loss = []
        meta_test_qry_loss = []

        meta_train_spt_acc = []
        meta_train_qry_acc = []
        meta_eval_spt_acc = []
        meta_eval_qry_acc = []
        meta_test_spt_acc = []
        meta_test_qry_acc = []
        # Inner Optimization
        # adapt_optimizer = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
        meta_optimizer.zero_grad()
        for _ in range(0, args.num_tasks):
            # adapt_optimizer.zero_grad()
            adapt_optimizer = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
            with higher.innerloop_ctx(meta_theta, adapt_optimizer,
                                      copy_initial_weights=False,
                                      override={'lr': torch.tensor([args.fast_lr],
                                                                   requires_grad=True).to(args.device)}
                                      ) as (theta_pi, diff_optim):
                # theta_pi = Maml(model=meta_theta.clone(), args=args).to(args.device)
                train_spt_loss, train_qry_loss, train_spt_acc, train_qry_acc = train_on_task(theta_pi=theta_pi,
                                                                                             tasksets=tasksets,
                                                                                             diff_optim=diff_optim)
                meta_train_spt_loss.append(train_spt_loss)
                meta_train_qry_loss.append(train_qry_loss)
                meta_train_spt_acc.append(train_spt_acc)
                meta_train_qry_acc.append(train_qry_acc)

                # val_spt_loss, val_qry_loss, val_spt_acc, val_qry_acc = val_on_task(meta_theta=meta_theta,
                #                                                                    tasksets=tasksets)
                # meta_eval_spt_loss.append(val_spt_loss)
                # meta_eval_qry_loss.append(val_qry_loss)
                # meta_eval_spt_acc.append(val_spt_acc)
                # meta_eval_qry_acc.append(val_qry_acc)
                #
                # test_spt_loss, test_qry_loss, test_spt_acc, test_qry_acc = adapt_on_task(meta_theta=meta_theta,
                #                                                                          tasksets=tasksets)
                # meta_test_spt_loss.append(test_spt_loss)
                # meta_test_qry_loss.append(test_qry_loss)
                # meta_test_spt_acc.append(test_spt_acc)
                # meta_test_qry_acc.append(test_qry_acc)
        # meta_optimizer.step()

        for _ in range(0, args.num_tasks):
            with higher.innerloop_ctx(meta_theta, adapt_optimizer,
                                      track_higher_grads=False, override={'lr': torch.tensor([args.fast_lr],
                                                                                             requires_grad=True).to(
                        args.device)}
                                      ) as (theta_pi, diff_optim):

                val_spt_loss, val_qry_loss, val_spt_acc, val_qry_acc = val_on_task(theta_pi=theta_pi,
                                                                                   diff_optim=diff_optim,
                                                                                   tasksets=tasksets)
                meta_eval_spt_loss.append(val_spt_loss)
                meta_eval_qry_loss.append(val_qry_loss)
                meta_eval_spt_acc.append(val_spt_acc)
                meta_eval_qry_acc.append(val_qry_acc)

                test_spt_loss, test_qry_loss, test_spt_acc, test_qry_acc = adapt_on_task(
                                                                                    theta_pi=theta_pi,
                                                                                    diff_optim=diff_optim,
                                                                                    tasksets=tasksets)
                meta_test_spt_loss.append(test_spt_loss)
                meta_test_qry_loss.append(test_qry_loss)
                meta_test_spt_acc.append(test_spt_acc)
                meta_test_qry_acc.append(test_qry_acc)



        avg_meta_train_spt_loss = sum(meta_train_spt_loss) / args.num_tasks
        avg_meta_train_qry_loss = sum(meta_train_qry_loss) / args.num_tasks
        avg_meta_train_spt_acc = sum(meta_train_spt_acc) / args.num_tasks
        avg_meta_train_qry_acc = sum(meta_train_qry_acc) / args.num_tasks

        avg_meta_eval_spt_loss = sum(meta_eval_spt_loss) / args.num_tasks
        avg_meta_eval_qry_loss = sum(meta_eval_qry_loss) / args.num_tasks
        avg_meta_eval_spt_acc = sum(meta_eval_spt_acc) / args.num_tasks
        avg_meta_eval_qry_acc = sum(meta_eval_qry_acc) / args.num_tasks

        avg_meta_test_spt_loss = sum(meta_test_spt_loss) / args.num_tasks
        avg_meta_test_qry_loss = sum(meta_test_qry_loss) / args.num_tasks
        avg_meta_test_spt_acc = sum(meta_test_spt_acc) / args.num_tasks
        avg_meta_test_qry_acc = sum(meta_test_qry_acc) / args.num_tasks
        if args.wandb_logging:
            wandb.log({"avg_meta_train_spt_loss:": avg_meta_train_spt_loss,
                       "avg_meta_train_qry_loss": avg_meta_train_qry_loss,
                       "avg_meta_train_spt_acc": avg_meta_train_spt_acc,
                       "avg_meta_train_qry_acc": avg_meta_train_qry_acc,

                       "avg_meta_eval_spt_loss:": avg_meta_eval_spt_loss,
                       "avg_meta_eval_qry_loss": avg_meta_eval_qry_loss,
                       "avg_meta_eval_spt_acc": avg_meta_eval_spt_acc,
                       "avg_meta_eval_qry_acc": avg_meta_eval_qry_acc,

                       "avg_meta_test_spt_loss:": avg_meta_test_spt_loss,
                       "avg_meta_test_qry_loss": avg_meta_test_qry_loss,
                       "avg_meta_test_spt_acc": avg_meta_test_spt_acc,
                       "avg_meta_test_qry_acc": avg_meta_test_qry_acc,
                       })

        # if epoch % 50 == 0:
        #     print(
        #         f'[Epoch {epoch:.2f}] Spt Train Loss: {avg_meta_train_spt_loss:.2f}  Qry Train Loss:{avg_meta_train_qry_loss:.2f}'
        #         f' Spt Val Loss:{avg_meta_eval_spt_loss:.2f}  Qry Val Loss:{avg_meta_eval_qry_loss:.2f}'
        #         f' Spt Test Loss:{avg_meta_test_spt_loss:.2f}  Qry Test Loss:{avg_meta_test_qry_loss:.2f}'
        #         # f'| Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
        #     )
            # qry_acc = (qry_logits.argmax(
        #     dim=1) == y_qry[i]).sum().item() / querysz
        # qry_accs.append(qry_acc)

        # Meta test set
        # X, y = tasksets.validation.sample()

        # print("Val:", y)

        # Adaptation step
        # X, y = tasksets.test.sample()
        # print("Test:", y)
        #     break
        # break

    # #Adapted NN
    # if(args.model == "maml"):
    #     model_pi = model()
    #     print(model_pi.clone())
    #     print(model_pi.adapt())

    # maml = MAML()
    # maml.adapt()
    # maml.clone()
    # print(maml)
