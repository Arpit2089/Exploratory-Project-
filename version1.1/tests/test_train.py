from data.cifar10 import get_cifar_loaders
from train.trainer import train_finetune
import torch.nn as nn

def main():

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*32*32, 10)
    )

    train_loader, val_loader = get_cifar_loaders()

    error = train_finetune(model, train_loader, val_loader, device="cpu", epochs=1)
    print("Validation error:", error)


if __name__ == "__main__":
    main()
