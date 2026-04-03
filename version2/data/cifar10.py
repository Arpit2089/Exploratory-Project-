################################################################################
# FOLDER: data
# FILE:   cifar10.py
# PATH:   .\data\cifar10.py
################################################################################

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar_loaders(batch_size=128, num_workers=0, split_test=False, fast_dev_mode=True):
    # Light augmentation for fast NAS search (Removed heavy AutoAugment/RandomErasing)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    # Only pin memory in the main process to prevent Windows Worker Deadlocks
    pin_mem = torch.cuda.is_available() and num_workers > 0

    if split_test:
        train_data_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_data_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        
        indices = list(range(len(train_data_full)))
        np.random.seed(42)  
        np.random.shuffle(indices)
        
        if fast_dev_mode:
            train_idx, val_idx = indices[:4500], indices[4500:5000]
        else:
            train_idx, val_idx = indices[:45000], indices[45000:]
            
        train_subset = Subset(train_data_full, train_idx)
        val_subset = Subset(val_data_full, val_idx)
        
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        return trainloader, valloader, testloader

    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        if fast_dev_mode:
            num_train = len(trainset)
            indices = list(range(num_train))
            trainset = Subset(trainset, indices[:int(0.1 * num_train)])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

        return trainloader, testloader