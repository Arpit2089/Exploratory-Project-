from data.cifar10 import get_cifar_loaders

train, test = get_cifar_loaders()

print(len(train))
print(len(test))
