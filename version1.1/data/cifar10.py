import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar_loaders(batch_size=128, num_workers=2):

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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return trainloader, testloader


# # data/cifar10.py

# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# def get_cifar_loaders(batch_size=128, num_workers=2):

#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     trainset = torchvision.datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform_train
#     )

#     testset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=transform_test
#     )

#     trainloader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers
#     )

#     testloader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )

#     return trainloader, testloader

# # def get_cifar_loaders(batch_size=128):

# #     transform_train = transforms.Compose([
# #         transforms.RandomCrop(32, padding=4),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.ToTensor(),
# #     ])

# #     transform_test = transforms.Compose([
# #         transforms.ToTensor(),
# #     ])

# #     trainset = torchvision.datasets.CIFAR10(
# #         root='./data', train=True, download=True, transform=transform_train
# #     )

# #     testset = torchvision.datasets.CIFAR10(
# #         root='./data', train=False, download=True, transform=transform_test
# #     )

# #     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
# #     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# #     return trainloader, testloader
