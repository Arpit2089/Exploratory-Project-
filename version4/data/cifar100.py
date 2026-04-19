# data/cifar100.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar100_loaders(batch_size=128, num_workers=0, pin_memory=False,
                         split_test=False, fast_dev_mode=True):
    """
    Returns CIFAR-100 data loaders.

    Parameters
    ----------
    num_workers  : prefetch workers (use 2 on Colab GPU, 0 on CPU-only / Windows)
    pin_memory   : pin CPU tensors for fast GPU transfers (True when CUDA available)
    split_test   : if True, returns (train, val, test); else (train, test)
    fast_dev_mode: if True, uses ~10% of data for quick iteration
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])

    persistent = (num_workers > 0)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            drop_last=False,
        )

    if split_test:
        train_data_full = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train)
        val_data_full = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_test)

        indices = list(range(len(train_data_full)))
        np.random.seed(42)
        np.random.shuffle(indices)

        if fast_dev_mode:
            train_idx, val_idx = indices[:4500], indices[4500:5000]
        else:
            train_idx, val_idx = indices[:45000], indices[45000:]

        trainloader = make_loader(Subset(train_data_full, train_idx), shuffle=True)
        valloader   = make_loader(Subset(val_data_full,   val_idx),   shuffle=False)

        testset    = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test)
        testloader = make_loader(testset, shuffle=False)

        return trainloader, valloader, testloader

    else:
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train)
        testset  = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test)

        if fast_dev_mode:
            n = len(trainset)
            trainset = Subset(trainset, list(range(int(0.1 * n))))

        trainloader = make_loader(trainset, shuffle=True)
        testloader  = make_loader(testset,  shuffle=False)

        return trainloader, testloader