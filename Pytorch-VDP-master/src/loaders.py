from torchvision import datasets
from torch.utils.data import DataLoader


class LoadData:
    def __init__(self, dataset, transform_in, args):
        dataset = dataset.upper()
        if dataset == 'MNIST':
            self.train_loader, self.test_loader, self.train_set, self.test_set = self.mnist(transform_in, args)
        elif dataset == 'FMNIST':
            self.train_loader, self.test_loader, self.train_set, self.test_set = self.fmnist(transform_in, args)
        elif dataset == 'CIFAR10':
            self.train_loader, self.test_loader, self.train_set, self.test_set = self.cifar10(transform_in, args)
        elif dataset == 'CIFAR100':
            self.train_loader, self.test_loader, self.train_set, self.test_set = self.cifar100(transform_in, args)
        else:
            print('Must choose a dataset')
        #print(f"Training Input Shape: {self.train_set.data[0].shape}")
        #print(f"Test Input Shape: {self.test_set.data[0].shape}")

    def get_datasets(self):
        return self.train_set, self.test_set

    def get_loaders(self):
        return self.train_loader, self.test_loader

    @staticmethod
    def mnist(transform_in, args):

        train_set = datasets.MNIST(args.data_path, train=True, download=True,
                                   transform=transform_in)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

        test_set = datasets.MNIST(args.data_path, train=False, download=True,
                                  transform=transform_in)

        test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=round(args.num_workers / 2),
                                 drop_last=True)

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def fmnist(transform_in, args):

        train_set = datasets.FashionMNIST(args.data_path, train=True, download=True,
                                          transform=transform_in)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_set = datasets.FashionMNIST(args.data_path, train=False, download=True,
                                         transform=transform_in)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=round(args.num_workers / 2))

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def cifar10(transform_in, args):
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True,
                                     transform=transform_in)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_set = datasets.CIFAR10(args.data_path, train=False, download=True,
                                    transform=transform_in)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=round(args.num_workers / 2))

        return train_loader, test_loader, train_set, test_set

    @staticmethod
    def cifar100(transform_in, args):
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True,
                                      transform=transform_in)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_set = datasets.CIFAR100(args.data_path, train=False, download=True,
                                     transform=transform_in)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=round(args.num_workers / 2))

        return train_loader, test_loader, train_set, test_set