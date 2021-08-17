import argparse
import importlib
import higher
parser = argparse.ArgumentParser()


parser.add_argument('--network', type=str, default='MNIST_CONV',
                    choices=['VDP', 'FMNIST_CONV', 'MNIST_CONV', 'CIFAR10_CONV','MNIST_FC_BBB'], help='Dataset Experiment')
parser.add_argument('--test', dest="test", action='store_true')
parser.add_argument('--no_test', dest="test", action='store_false')

parser.add_argument('--model', type=str, choices=['maml'], default='maml',
                            help='Name of the Algorithm (default: maml).')
# parser.set_defaults(test=False)

args = parser.parse_args()

from inspect import getmembers, isfunction

if __name__ == '__main__':
    print("Argument:", args)
    args.model = "Hello"
    print("Argument:", args)

    if 0:
        print("True")
    else:
        print("False")

    # netmodload = importlib.import_module('.VDP.Networks.' + args.network + '.VDPNet', package='Net')
    # model = netmodload.Net(args)
    # ResNets = getmembers(resnet, isfunction)
    # for name, func in ResNets:
    #     print(name)
    #     if "resnet" in name.lower():
    #         print(name)
    print(higher.innerloop_ctx())
    with higher.innerloop_ctx(None, None,
                              copy_initial_weights=False,
                              # override={'lr': torch.tensor([args.fast_lr],
                              # requires_grad=True).to(args.device)}
                              ) as (theta_pi, diff_optim):
        print("Passed")
