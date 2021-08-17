import argparse
import wandb
from torchsummary import torchsummary

from train import run_inner_loop
from test import run_val_loop
from models.vanilla_net import VanillaNet
from models.resnet import ResNet18
from models.resnet import ResNet18S
import torch.optim as optim
import utils
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def load_args():
    parser = argparse.ArgumentParser('MAML using highers library')

    # General
    general_params = parser.add_argument_group('General Params')
    general_params.add_argument('--data', type=str, default="./data",
                                help='Path to the folder the data is downloaded to.')
    general_params.add_argument('--dataset', type=str,
                                choices=["cifarfs", "mini-imagenet", "fc100",
                                         "cifarfs", "tiered-imagenet"], default='omniglot',
                                help='Name of the dataset (default: omniglot).')
    general_params.add_argument('--ways', type=int, default=5,
                                help='Number of classes per task (N in "N-way", default: 5).')
    general_params.add_argument('--shots', type=int, default=1,
                                help='Number of training example per class (k in "k-shot", default: 5).')
    general_params.add_argument('--adaptation-steps', type=int, default=1,
                                help='Number of adaptation steps on meta-train datasets.')
    # general_params.add_argument('--num-shots-test', type=int, default=15,
    #                             help='Number of test example per class. If negative, same as the number '
    #                                  'of training examples `--num-shots` (default: 15).')

    # Model
    model_params = parser.add_argument_group('Model')
    model_params.add_argument('--model-name', type=str, choices=['vanilla',
                                                                 'lenet', 'mlp', 'senet', 'resnet'], default='vanilla',
                              help='Name of the Algorithm (default: maml).')
    model_params.add_argument('--fine-tune', type=int, default=0, help='Only meta learn the FC layer')
    model_params.add_argument('--input-dim', type=int, default=32, help='Input dimension')
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
    optim_params.add_argument('--meta-learn', type=int, default=1,
                              help='Set this to False only for debugging purpose to'
                                   'verify that meta learning is happening')
    optim_params.add_argument('--fast-lr', type=float, default=0.01,
                              help='Learning rate for the meta-optimizer (optimization of the outer '
                                   'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    misc = parser.add_argument_group('Misc')
    misc.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers to use for data-loading (default: 1).')
    # Logging
    misc.add_argument('--wand-project', type=str, default="Cifar_Meta_With_Diff_Architectures",
                      help='Wandb project name should go here')
    misc.add_argument('--wand-note', type=str, default="Test Run Note",
                      help='To identify run')

    misc.add_argument('--username', type=str, default="hikmatkhan",
                      help='Wandb username should go here')
    misc.add_argument('--wandb-logging', type=int, default=1,
                      help='If True, Logs will be reported on wandb.')
    misc.add_argument('--verbose', action='store_true')
    misc.add_argument('--use-cuda', action='store_true')
    misc.add_argument('--device', type=str, default=utils.get_compute_device(), help="Compute device information")
    args = parser.parse_args()

    return args


from inspect import getmembers, isfunction
from models import resnet

if __name__ == '__main__':

    args = load_args()
    utils.fix_seeds(seed=args.seed)
    # if args.wandb_logging:
    #     wandb.init(project=args.wand_project, entity="hikmatkhan-")
    #     wandb.config.update(args, allow_val_change=True)

    ResNets = getmembers(resnet, isfunction)
    # Datasets = [
    # "omniglot",
    #         "mini-imagenet",
    # "cifarfs",
    # "tiered-imagenet"
    # ]

    for name, func in ResNets:
        if "cifar" in name.lower():
            # print("Training:", name)
            print(name)

            # args.wand_project = "Cifar_Meta_With_Diff_Architectures"
            args.model_name = name
            # print(func(out_dim=argss.ways))
            meta_theta = func(out_dim=args.ways).to(args.device)
            # print(meta_theta)
            # Higher_Mini_ImgNet_5W_1S_AStep_1_MetaLearn_True
            args.wand_note = "Higher_{0}_{1}W_{2}S_AStep_{3}_MLearn_{4}".format(
                args.dataset, args.ways, args.shots, args.adaptation_steps, args.meta_learn
            )
            # wandb.config.wand_note = args.wand_note
            # wandb.architecture = name

            if args.wandb_logging:
                wandb.init(project=args.wand_project, entity="hikmatkhan-", reinit=True)
                wandb.config.update(args)

            # from time import sleep
            # sleep(15)
            # continue

            # tasksets = utils.get_torch_ds(ways=args.ways, shots=args.shots, num_tasks=args.num_tasks)
            tasksets = utils.get_l2l_ds(args.dataset, data_path=args.data, ways=args.ways,
                                        shots=args.shots, num_tasks=args.num_tasks)

            meta_optim = optim.Adam(meta_theta.parameters(), lr=args.meta_lr)
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
                fast_optim = optim.SGD(meta_theta.parameters(), lr=args.fast_lr, momentum=0.9)
                meta_optim.zero_grad()
                # ----------------------------------- Inner loop -----------------------------------#
                task_level_train_spt_loss, task_level_train_qry_loss, task_level_train_spt_acc, task_level_train_qry_acc = run_inner_loop(
                    meta_theta, fast_optim, tasksets, args)

                mean_meta_train_spt_loss = sum(task_level_train_spt_loss) / args.num_tasks
                mean_meta_train_qry_loss = sum(task_level_train_qry_loss) / args.num_tasks
                mean_meta_train_spt_acc = sum(task_level_train_spt_acc) / args.num_tasks
                mean_meta_train_qry_acc = sum(task_level_train_qry_acc) / args.num_tasks
                # ----------------------------------- Inner loop -----------------------------------#
                if (args.meta_learn):
                    meta_optim.step()

                # Meta Evaluation
                task_level_eval_spt_loss, task_level_eval_qry_loss, task_level_eval_spt_acc, task_level_eval_qry_acc = run_val_loop(
                    eval_on_testset=False, meta_theta=meta_theta,
                    fast_optim=fast_optim, tasksets=tasksets, args=args)

                mean_meta_eval_spt_loss = sum(task_level_eval_spt_loss) / args.num_tasks
                mean_meta_eval_qry_loss = sum(task_level_eval_qry_loss) / args.num_tasks
                mean_meta_eval_spt_acc = sum(task_level_eval_spt_acc) / args.num_tasks
                mean_meta_eval_qry_acc = sum(task_level_eval_qry_acc) / args.num_tasks

                # Meta Adaptation
                task_level_adapt_spt_loss, task_level_adapt_qry_loss, task_level_adapt_spt_acc, task_level_adapt_qry_acc = run_val_loop(
                    eval_on_testset=True, meta_theta=meta_theta,
                    fast_optim=fast_optim, tasksets=tasksets, args=args)

                mean_meta_adapt_spt_loss = sum(task_level_adapt_spt_loss) / args.num_tasks
                mean_meta_adapt_qry_loss = sum(task_level_adapt_qry_loss) / args.num_tasks
                mean_meta_adapt_spt_acc = sum(task_level_adapt_spt_acc) / args.num_tasks
                mean_meta_adapt_qry_acc = sum(task_level_adapt_qry_acc) / args.num_tasks

                if args.wandb_logging:
                    wandb.log({"M-trn_spt_loss": mean_meta_train_spt_loss,
                               "M-trn_qry_loss": mean_meta_train_qry_loss,
                               "M-trn_spt_acc": mean_meta_train_spt_acc,
                               "M-trn_qry_acc": mean_meta_train_qry_acc,

                               "M-eval_spt_loss": mean_meta_eval_spt_loss,
                               "M-eval_qry_loss": mean_meta_eval_qry_loss,
                               "M-eval_spt_acc": mean_meta_eval_spt_acc,
                               "M-eval_qry_acc": mean_meta_eval_qry_acc,

                               "Adpt_spt_loss": mean_meta_adapt_spt_loss,
                               "Adpt_qry_loss": mean_meta_adapt_qry_loss,
                               "Adpt_spt_acc": mean_meta_adapt_spt_acc,
                               "Adpt_qry_acc": mean_meta_adapt_qry_acc,
                               })
