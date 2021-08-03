import argparse

from MetaLearning.model import model
from utility import utils

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
                                     "cifarfs", "tiered-imagenet"], default='cifarfs',
                            help='Name of the dataset (default: omniglot).')

general_params.add_argument('--ways', type=int, default=5,
                            help='Number of classes per task (N in "N-way", default: 5).')
general_params.add_argument('--shots', type=int, default=1,
                            help='Number of training example per class (k in "k-shot", default: 5).')
# general_params.add_argument('--num-shots-test', type=int, default=15,
#                             help='Number of test example per class. If negative, same as the number '
#                                  'of training examples `--num-shots` (default: 15).')

# Model
model_params = parser.add_argument_group('Model')
model_params.add_argument('--fine-tune', type=bool, default=True,
                          help='Only meta learn the FC layer')
model_params.add_argument('--input', type=int, default=32,
                          help='Input to dimension')

general_params.add_argument('--model', type=str,
                            choices=['maml'], default='maml',
                            help='Name of the Algorithm (default: maml).')



# Optimization
optim_params = parser.add_argument_group('Optimization')

optim_params.add_argument('--batch-size', type=int, default=25,
                          help='Number of tasks in a batch of tasks (default: 25).')
optim_params.add_argument('--num-steps', type=int, default=1,
                          help='Number of fast adaptation steps, ie. gradient descent '
                               'updates (default: 1).')
optim_params.add_argument('--num-epochs', type=int, default=50,
                          help='Number of epochs of meta-training (default: 50).')
optim_params.add_argument('--num-batches', type=int, default=100,
                          help='Number of batch of tasks per epoch (default: 100).')
optim_params.add_argument('--step-size', type=float, default=0.1,
                          help='Size of the fast adaptation step, ie. learning rate in the '
                               'gradient descent update (default: 0.1).')
optim_params.add_argument('--first-order', action='store_true',
                          help='Use the first order approximation, do not use higher-order '
                               'derivatives during meta-optimization.')
optim_params.add_argument('--meta-lr', type=float, default=0.001,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')
optim_params.add_argument('--fast-lr', type=float, default=0.001,
                          help='Learning rate for the meta-optimizer (optimization of the outer '
                               'loss). The default optimizer is Adam (default: 1e-3).')

# Misc
misc = parser.add_argument_group('Misc')
misc.add_argument('--num-workers', type=int, default=1,
                  help='Number of workers to use for data-loading (default: 1).')
misc.add_argument('--verbose', action='store_true')
misc.add_argument('--use-cuda', action='store_true')
misc.add_argument('--device', type=str, default=utils.get_compute_device(), help="Compute device information")
args = parser.parse_args()
# TODO
#1. Load datasets
#2. Define adapted NN
#3. Define MetaLearner
if __name__ == '__main__':
    print(args)
    print("Meta_Lr:", args.num_epochs)

    meta_theta = model(args=args)
    # Outer Optimization
    for epoch in range(0, args.num_epochs):
        task_counts = 1

        # Inner Optimization
        for t in range(0, task_counts):
            tasksets = utils.get_l2l_ds(args.dataset, data_path=args.data, ways=args.ways,
                                        shots=args.shots)
            X, y = tasksets.train.sample()
            meta_theta(X)
            print(X.size)
            print("Train:", y)
            X, y = tasksets.validation.sample()
            print("Val:", y)
            X, y = tasksets.test.sample()
            print("Test:", y)




    # #Adapted NN
    # if(args.model == "maml"):
    #     model_pi = model()
    #     print(model_pi.clone())
    #     print(model_pi.adapt())

    # maml = MAML()
    # maml.adapt()
    # maml.clone()
    # print(maml)

