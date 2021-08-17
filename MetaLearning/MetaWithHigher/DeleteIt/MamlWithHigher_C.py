import argparse

import wandb

from MetaLearning.MetaWithHigher.train import run_inner_loop
from MetaLearning.MetaWithHigher.models import vanilla_net
from MetaLearning.MetaWithHigher import utils
import torch.optim as optim

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser('MAML using highers library')

# General
general_params = parser.add_argument_group('General Params')
general_params.add_argument('--data', type=str, default="./data",
                            help='Path to the folder the data is downloaded to.')
general_params.add_argument('--dataset', type=str,
                            choices=["omniglot", "mini-imagenet", "fc100",
                                     "cifarfs", "tiered-imagenet"], default='omniglot',
                            help='Name of the dataset (default: omniglot).')
general_params.add_argument('--ways', type=int, default=5,
                            help='Number of classes per task (N in "N-way", default: 5).')
general_params.add_argument('--shots', type=int, default=5,
                            help='Number of training example per class (k in "k-shot", default: 5).')
general_params.add_argument('--adaptation-steps', type=int, default=5,
                            help='Number of adaptation steps on meta-train datasets.')
# general_params.add_argument('--num-shots-test', type=int, default=15,
#                             help='Number of test example per class. If negative, same as the number '
#                                  'of training examples `--num-shots` (default: 15).')

# Model
model_params = parser.add_argument_group('Model')
model_params.add_argument('--fine-tune', type=bool, default=False, help='Only meta learn the FC layer')
model_params.add_argument('--input-dim', type=int, default=28, help='Input dimension')
model_params.add_argument('--channels', type=int, default=1, help='Input channels')
general_params.add_argument('--model', type=str, choices=['maml'], default='maml',
                            help='Name of the Algorithm (default: maml).')

# Optimization
optim_params = parser.add_argument_group('Optimization')
# optim_params.add_argument('--batch-size', type=int, default=25,
#                           help='Number of tasks in a batch of tasks (default: 25).')
optim_params.add_argument('--seed', type=int, default=101,
                          help='Default seed to start experiments.')
optim_params.add_argument('--num-steps', type=int, default=1,
                          help='Number of fast adaptation steps, ie. gradient descent '
                               'updates (default: 1).')
optim_params.add_argument('--num-epochs', type=int, default=50000,
                          help='Number of epochs of meta-training (default: 50).')
# optim_params.add_argument('--num-batches', type=int, default=100,
#                           help='Number of batch of tasks per epoch (default: 100).')
optim_params.add_argument('--num-tasks', type=int, default=32,
                          help='Meta-Batch Size: Number of tasks to sample from task distribution.')
# optim_params.add_argument('--step-size', type=float, default=0.1,
#                           help='Size of the fast adaptation step, ie. learning rate in the '
#                                'gradient descent update (default: 0.1).')
optim_params.add_argument('--first-order', action='store_true',
                          help='Use the first order approximation, do not use highers-order '
                               'derivatives during meta-optimization.')
optim_params.add_argument('--meta-lr', type=float, default=0.001,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')
optim_params.add_argument('--fast-lr', type=float, default=0.01,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')

# Misc
misc = parser.add_argument_group('Misc')
misc.add_argument('--num-workers', type=int, default=1,
                  help='Number of workers to use for data-loading (default: 1).')
# Logging
misc.add_argument('--wand_project', type=str, default="MamlWithHigher",
                  help='Wandb project name should go here')

misc.add_argument('--username', type=str, default="hikmatkhan",
                  help='Wandb username should go here')
misc.add_argument('--wandb_logging', type=bool, default=True,
                  help='If True, Logs will be reported on wandb.')
misc.add_argument('--verbose', action='store_true')
misc.add_argument('--use-cuda', action='store_true')
misc.add_argument('--device', type=str, default=utils.get_compute_device(), help="Compute device information")
args = parser.parse_args()

#
# def train_on_task(theta_pi, diff_optim, tasksets):
#     # Meta training set
#     X, y = tasksets.train.sample()
#     X, y = X.to(args.device), y.to(args.device)
#     # adapt_optimizer = optim.SGD(meta_theta.parameters(),
#     #                             lr=0.1, momentum=0.9)#Adam(meta_theta.parameters(), lr=args.fast_lr)
#
#     meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
#     # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
#     # with highers.innerloop_ctx(meta_theta, adapt_optimizer,
#     #                           copy_initial_weights=False,
#     #                           override={'lr': torch.tensor([args.fast_lr],
#     #                                                        requires_grad=True).to(args.device)}) as (theta_pi, diff_optim):
#     #     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
#     spt_mean_loss = 0
#     spt_mean_acc = 0
#     for step in range(args.adaptation_steps):
#         y_spt = theta_pi(X[meta_train_indices])
#         spt_mean_acc += accuracy(y_spt, y[meta_train_indices])
#         spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
#         spt_mean_loss += spt_loss.detach().item()
#         diff_optim.step(spt_loss)  # After this call. There will be next version of the theta
#
#     spt_mean_loss /= args.adaptation_steps
#     spt_mean_acc /= args.adaptation_steps
#     y_qry = theta_pi(X[meta_test_indices])
#     qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
#     qry_loss.backward()
#
#     return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, accuracy(y_qry, y[meta_test_indices])


# def val_on_task(theta_pi, diff_optim, tasksets):
#     # Meta training set
#     X, y = tasksets.validation.sample()
#     X, y = X.to(args.device), y.to(args.device)
#     # adapt_optimizer = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
#
#     meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
#     # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
#     # with highers.innerloop_ctx(meta_theta, adapt_optimizer,
#     #                           track_higher_grads=False, override={'lr': torch.tensor([args.fast_lr],
#     #                                                                                  requires_grad=True).to(args.device)}
#     #                           ) as (theta_pi, diff_optim):
#     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
#     spt_mean_loss = 0
#     spt_mean_acc = 0
#     for step in range(args.adaptation_steps):
#         y_spt = theta_pi(X[meta_train_indices])
#         spt_mean_acc += accuracy(y_spt, y[meta_train_indices])
#         spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
#         spt_mean_loss += spt_loss.detach().item()
#         diff_optim.step(spt_loss)  # After this call. There will be next version of the theta
#
#     spt_mean_loss /= args.adaptation_steps
#     spt_mean_acc /= args.adaptation_steps
#     y_qry = theta_pi(X[meta_test_indices])
#     qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
#     # qry_loss.backward()
#     # meta_optimizer.step()
#     return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, accuracy(y_qry, y[meta_test_indices])
#
#
# def adapt_on_task(theta_pi, diff_optim, tasksets):
#     # Meta test set (Adaptation Phase)
#     X, y = tasksets.test.sample()
#     X, y = X.to(args.device), y.to(args.device)
#     # adapt_optimizer = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
#
#     meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
#     # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
#     # with highers.innerloop_ctx(meta_theta, adapt_optimizer,
#     #                           track_higher_grads=False, override={'lr': torch.tensor([args.fast_lr],
#     #                                                                                  requires_grad=True).to(args.device)}) as (
#     #         theta_pi, diff_optim):
#     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
#     spt_mean_loss = 0
#     spt_mean_acc = 0
#     for step in range(args.adaptation_steps):
#         y_spt = theta_pi(X[meta_train_indices])
#         spt_mean_acc += accuracy(y_spt, y[meta_train_indices])
#         spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
#         spt_mean_loss += spt_loss.detach().item()
#         diff_optim.step(spt_loss)  # After this call. There will be next version of the theta
#
#     spt_mean_loss /= args.adaptation_steps
#     spt_mean_acc /= args.adaptation_steps
#     y_qry = theta_pi(X[meta_test_indices])
#     qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
#     # qry_loss.backward()
#     # meta_optimizer.step()
#     return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, accuracy(y_qry, y[meta_test_indices])

if __name__ == '__main__':
    utils.fix_seeds(seed=args.seed)
    print(args)
    if args.wandb_logging:
        wandb.init(project=args.wand_project, entity="hikmatkhan-")

    # tasksets = utils.get_torch_ds(ways=args.ways, shots=args.shots, num_tasks=args.num_tasks)
    tasksets = utils.get_l2l_ds(args.dataset, data_path=args.data, ways=args.ways,
                                shots=args.shots, num_tasks=args.num_tasks)

    # Outer Optimization
    # meta_theta = Maml(model=ResNet18(output_classes=args.ways), args=args).to(args.device)
    # meta_theta = Maml(model=LeNetwork(out_dim=args.ways, in_channel=1, img_sz=28), args=args).to(args.device)
    # meta_theta = LeNetwork(out_dim=args.ways, in_channel=args.channels, img_sz=args.input_dim).to(args.device)
    meta_theta = vanilla_net(args=args).to(args.device)
    meta_optim = optim.Adam(meta_theta.parameters(), lr=args.meta_lr)
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
        fast_optim = optim.SGD(meta_theta.parameters(), lr=args.fast_lr)
        meta_optim.zero_grad()
        #----------------------------------- Inner loop -----------------------------------#
        task_level_train_spt_loss, task_level_train_qry_loss, task_level_train_spt_acc, task_level_train_qry_acc = run_inner_loop(
            meta_theta, fast_optim, tasksets, args)

        mean_meta_train_spt_loss = sum(task_level_train_spt_loss) / args.num_tasks
        mean_meta_train_qry_loss = sum(task_level_train_qry_loss) / args.num_tasks
        mean_meta_train_spt_acc = sum(task_level_train_spt_acc) / args.num_tasks
        mean_meta_train_qry_acc = sum(task_level_train_qry_acc) / args.num_tasks
        #----------------------------------- Inner loop -----------------------------------#
        meta_optim.step()






        # # for epoch in range(args.num_epochs):
        # # for _ in range(0, args.num_tasks):
        #     # adapt_optimizer.zero_grad()
        #     # with MetaLearning.Higher.innerloop_ctx(meta_theta, adapt_optimizer,
        #     #                                        copy_initial_weights=False,
        #     #                                        override={'lr': torch.tensor([args.fast_lr],
        #     #                                                        requires_grad=True).to(args.device)}
        #     #                                        ) as (theta_pi, diff_optim):
        #         # theta_pi = Maml(model=meta_theta.clone(), args=args).to(args.device)
        #
        #         # train_spt_loss, train_qry_loss, train_spt_acc, train_qry_acc = train_on_task(theta_pi=theta_pi,
        #         #                                                                              tasksets=tasksets,
        #         #                                                                              diff_optim=diff_optim)
        #
        #     meta_train_spt_loss.append(train_spt_loss)
        #     meta_train_qry_loss.append(train_qry_loss)
        #     meta_train_spt_acc.append(train_spt_acc)
        #     meta_train_qry_acc.append(train_qry_acc)
        #
        #         # val_spt_loss, val_qry_loss, val_spt_acc, val_qry_acc = val_on_task(meta_theta=meta_theta,
        #         #                                                                    tasksets=tasksets)
        #         # meta_eval_spt_loss.append(val_spt_loss)
        #         # meta_eval_qry_loss.append(val_qry_loss)
        #         # meta_eval_spt_acc.append(val_spt_acc)
        #         # meta_eval_qry_acc.append(val_qry_acc)
        #         #
        #         # test_spt_loss, test_qry_loss, test_spt_acc, test_qry_acc = adapt_on_task(meta_theta=meta_theta,
        #         #                                                                          tasksets=tasksets)
        #         # meta_test_spt_loss.append(test_spt_loss)
        #         # meta_test_qry_loss.append(test_qry_loss)
        #         # meta_test_spt_acc.append(test_spt_acc)
        #         # meta_test_qry_acc.append(test_qry_acc)
        # meta_optimizer.step()
        #
        # for _ in range(0, args.num_tasks):
        #     with MetaLearning.Higher.innerloop_ctx(meta_theta, adapt_optimizer,
        #                                            track_higher_grads=False, override={'lr': torch.tensor([args.fast_lr],
        #                                                                                      requires_grad=True).to(
        #                 args.device)}
        #                                            ) as (theta_pi, diff_optim):
        #         val_spt_loss, val_qry_loss, val_spt_acc, val_qry_acc = val_on_task(theta_pi=theta_pi,
        #                                                                            diff_optim=diff_optim,
        #                                                                            tasksets=tasksets)
        #         meta_eval_spt_loss.append(val_spt_loss)
        #         meta_eval_qry_loss.append(val_qry_loss)
        #         meta_eval_spt_acc.append(val_spt_acc)
        #         meta_eval_qry_acc.append(val_qry_acc)
        #
        #         test_spt_loss, test_qry_loss, test_spt_acc, test_qry_acc = adapt_on_task(
        #             theta_pi=theta_pi,
        #             diff_optim=diff_optim,
        #             tasksets=tasksets)
        #         meta_test_spt_loss.append(test_spt_loss)
        #         meta_test_qry_loss.append(test_qry_loss)
        #         meta_test_spt_acc.append(test_spt_acc)
        #         meta_test_qry_acc.append(test_qry_acc)

        # avg_meta_train_spt_loss = sum(meta_train_spt_loss) / args.num_tasks
        # avg_meta_train_qry_loss = sum(meta_train_qry_loss) / args.num_tasks
        # avg_meta_train_spt_acc = sum(meta_train_spt_acc) / args.num_tasks
        # avg_meta_train_qry_acc = sum(meta_train_qry_acc) / args.num_tasks

        # avg_meta_eval_spt_loss = sum(meta_eval_spt_loss) / args.num_tasks
        # avg_meta_eval_qry_loss = sum(meta_eval_qry_loss) / args.num_tasks
        # avg_meta_eval_spt_acc = sum(meta_eval_spt_acc) / args.num_tasks
        # avg_meta_eval_qry_acc = sum(meta_eval_qry_acc) / args.num_tasks
        #
        # avg_meta_test_spt_loss = sum(meta_test_spt_loss) / args.num_tasks
        # avg_meta_test_qry_loss = sum(meta_test_qry_loss) / args.num_tasks
        # avg_meta_test_spt_acc = sum(meta_test_spt_acc) / args.num_tasks
        # avg_meta_test_qry_acc = sum(meta_test_qry_acc) / args.num_tasks
        if args.wandb_logging:
            wandb.log({"M-trn_spt_loss:": mean_meta_train_spt_loss,
                       "M-trn_qry_loss": mean_meta_train_qry_loss,
                       "M-trn_spt_acc": mean_meta_train_spt_acc,
                       "M-trn_qry_acc": mean_meta_train_qry_acc,

                       # "avg_meta_eval_spt_loss:": avg_meta_eval_spt_loss,
                       # "avg_meta_eval_qry_loss": avg_meta_eval_qry_loss,
                       # "avg_meta_eval_spt_acc": avg_meta_eval_spt_acc,
                       # "avg_meta_eval_qry_acc": avg_meta_eval_qry_acc,
                       #
                       # "avg_meta_test_spt_loss:": avg_meta_test_spt_loss,
                       #              "avg_meta_test_qry_loss": avg_meta_test_qry_loss,
                       #              "avg_meta_test_spt_acc": avg_meta_test_spt_acc,
                       #              "avg_meta_test_qry_acc": avg_meta_test_qry_acc,
                                    })

