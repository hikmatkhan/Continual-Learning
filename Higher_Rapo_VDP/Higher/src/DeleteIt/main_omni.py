import mlogger
from torch import optim, nn
from torchvision.models import resnet18

import utils
import argparse

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from models import resnet
from models import resnet
from models.resnet import ResNet18S, ResNet18
from models.vanilla_net import VanillaNet
from test import run_val_loop, run_test_loop
from train import run_inner_loop


def argv():
    parser = argparse.ArgumentParser('MAML with Higher')
    # General
    general_params = parser.add_argument_group('General')
    general_params.add_argument('--data', type=str, default="./data",
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
    meta_params.add_argument('--total-num-tasks', type=int, default=-1,
                             help='Total number of tasks in task distribution. Always keep it to -1.')
    meta_params.add_argument('--first-order', action='store_true',
                             help='Use the first order approximation, do not use highers-order '
                                  'derivatives during meta-optimization.')
    meta_params.add_argument('--meta-lr', type=float, default=0.001,
                             help='Learning rate for the meta-optimizer (optimization of the outer '
                                  'loss). The default optimizer is Adam (default: 1e-3).')
    meta_params.add_argument('--fast-lr', type=float, default=0.01,
                             help='Learning rate for the meta-optimizer (optimization of the outer '
                                  'loss). The default optimizer is Adam (default: 1e-3).')
    meta_params.add_argument('--num-test-task', type=int, default=32,
                             help='Number of test task used to calculate meta test and meta val')
    meta_params.add_argument('--meta-learn', type=int, default=1,
                              help='Set this to False only for debugging purpose to'
                                   'verify that meta learning is happening')

    # Model
    model_params = parser.add_argument_group('Model')
    model_params.add_argument('--input', type=int, default=84,
                              help='Input to dimension')
    model_params.add_argument('--channels', type=int, default=3,
                              help='Input to dimension')
    model_params.add_argument('--model-name', type=str, choices=
    ['vanilla', 'lenet', 'mlp', 'senet', 'resnet'], default='resnet',
                              help='Name of the Algorithm (default: maml).')
    model_params.add_argument('--fine-tune', type=int, default=1,
                              help='Only meta learn the FC layer')

    # Optimization
    optim_params = parser.add_argument_group('Optimization')
    optim_params.add_argument('--num-epochs', type=int, default=50000,
                              help='Number of epochs of meta-training (default: 50000).')
    optim_params.add_argument('--seed', type=int, default=utils.fix_seeds(),
                              help='Number of epochs of meta-training (default: 101).')

    # Misc
    misc = parser.add_argument_group('Misc')
    misc.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers to use for data-loading (default: 1).')
    misc.add_argument('--device', type=str, default=utils.get_compute_device(), help="Compute device information")

    # Visualization
    viz = parser.add_argument_group('Misc')
    viz.add_argument('--wand-project', type=str, default="Github_Rapo",
                     help='Wandb project name should go here')
    viz.add_argument('--username', type=str, default="hikmatkhan",
                     help='Wandb username should go here')
    viz.add_argument('--wandb-log', type=int, default=1,
                     help='If True, Logs will be reported on wandb.')
    viz.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    args = argv()
    print(args)

    # Load datasets
    tasksets = utils.get_l2l_ds(args=args)

    # Load model
    # if args.model_name == 'vanilla':
    #     model = VanillaNet(args=args).to(args.device)
    # elif args.model_name == 'resnet':
    #     model = ResNet18(out_dim=args.ways).to(args.device)

    model = utils.get_meta_theta(args=args)

    # model = resnet18(pretrained=True).to(args.device)
    # if args.fine_tune:
    #     for param in model.parameters():
    #         param.requires_grad = False
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, args.ways).to(args.device)

    print(model)

    # Setup optimizer
    optim_meta = optim.Adam(model.parameters(), lr=args.meta_lr)

    if args.wandb_log:
        import wandb

        wandb.init(project="JMaml", entity="hikmatkhan-")
        wandb.watch(model, log_freq=10)
        wandb.config.update(args)

    for e in range(args.num_epochs):

        optim_meta.zero_grad()
        # ----------------------------------- Inner loop -----------------------------------#
        meta_train_acc, meta_train_loss = run_inner_loop(
            meta_theta=model, tasksets=tasksets, args=args)
        # ----------------------------------- Inner loop -----------------------------------#
        meta_step_info = ""
        if args.meta_learn:
            optim_meta.step()
            meta_step_info = "Meta step has taken."
        else:
            meta_step_info = "Meta step has skipped."

        # ----------------------------------- Outer loop -----------------------------------#
        # Meta Evaluation
        meta_val_acc, meta_val_loss = run_val_loop(meta_theta=model,
                                                   tasksets=tasksets, args=args)
        # Meta Adaptation
        meta_test_acc, meta_test_loss = run_test_loop(meta_theta=model,
                                                      tasksets=tasksets, args=args)
        if args.wandb_log:
            wandb.log({"meta_train_acc": meta_train_acc,
                       "meta_train_loss": meta_train_loss,
                       "meta_val_acc": meta_val_acc,
                       "meta_val_loss": meta_val_loss,
                       "meta_test_acc": meta_test_acc,
                       "meta_test_loss": meta_test_loss})
        print("Epoch|", e, "|", "Meta Train Acc:", meta_train_acc, " Meta Train Loss:", meta_train_loss,
              "Meta Val Acc:", meta_val_acc, " Meta Val Loss:", meta_val_loss,
              "Meta Test Acc:", meta_test_acc, " Meta Test Loss:", meta_test_loss, "(", meta_step_info,")")

