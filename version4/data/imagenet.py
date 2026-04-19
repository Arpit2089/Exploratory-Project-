# data/imagenet.py
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.logger import get_logger

logger = get_logger("imagenet_loader", logfile="logs/imagenet.log")


def get_imagenet_loaders(batch_size=32, num_workers=0, pin_memory=False,
                         split_test=False, fast_dev_mode=True):
    """
    Returns ImageNet data loaders.

    Parameters
    ----------
    num_workers  : prefetch workers
    pin_memory   : pin CPU tensors for fast GPU transfers
    split_test   : if True, returns (train, val, test); else (train, test)
    fast_dev_mode: if True, uses a small subset
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data_dir     = "./data/imagenet"
    use_fake_data = not os.path.exists(os.path.join(data_dir, "train"))
    if use_fake_data:
        logger.warning("ImageNet not found at %s — using FakeData.", data_dir)
        def dataset_class(is_train, transform):
            return torchvision.datasets.FakeData(
                size=5000 if is_train else 1000,
                image_size=(3, 224, 224),
                num_classes=1000,
                transform=transform,
            )
    else:
        def dataset_class(is_train, transform):
            return torchvision.datasets.ImageFolder(
                root=os.path.join(data_dir, "train" if is_train else "val"),
                transform=transform,
            )

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
        train_data_full = dataset_class(is_train=True,  transform=transform_train)
        val_data_full   = dataset_class(is_train=True,  transform=transform_test)

        indices = list(range(len(train_data_full)))
        np.random.seed(42)
        np.random.shuffle(indices)

        if fast_dev_mode:
            train_idx, val_idx = indices[:1000], indices[1000:1200]
        else:
            split = int(0.9 * len(indices))
            train_idx, val_idx = indices[:split], indices[split:]

        trainloader = make_loader(Subset(train_data_full, train_idx), shuffle=True)
        valloader   = make_loader(Subset(val_data_full,   val_idx),   shuffle=False)

        testset    = dataset_class(is_train=False, transform=transform_test)
        testloader = make_loader(testset, shuffle=False)

        return trainloader, valloader, testloader

    else:
        trainset = dataset_class(is_train=True,  transform=transform_train)
        testset  = dataset_class(is_train=False, transform=transform_test)

        if fast_dev_mode:
            n = len(trainset)
            trainset = Subset(trainset, list(range(int(0.1 * n))))

        trainloader = make_loader(trainset, shuffle=True)
        testloader  = make_loader(testset,  shuffle=False)

        return trainloader, testloader