import random
import torch.nn as nn
import torch.nn.functional as F

class DynamicCNN(nn.Module):
    def __init__(self, arch_dict, num_classes=10):
        super(DynamicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, arch_dict['c1'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(arch_dict['c1'], arch_dict['c2'], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(arch_dict['c2'], arch_dict['c3'], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Forces any spatial size down to 4x4 for the linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        
        self.fc1 = nn.Linear(arch_dict['c3'] * 4 * 4, arch_dict['fc'])
        self.fc2 = nn.Linear(arch_dict['fc'], num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x) 
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_random_architecture():
    return {'c1': random.choice([16, 32, 64]), 'c2': random.choice([32, 64, 128]),
            'c3': random.choice([64, 128, 256]), 'fc': random.choice([128, 256, 512])}

def mutate_architecture(parent_arch):
    child = parent_arch.copy()
    key = random.choice(list(child.keys()))
    if key == 'c1': child['c1'] = random.choice([16, 32, 64])
    elif key == 'c2': child['c2'] = random.choice([32, 64, 128])
    elif key == 'c3': child['c3'] = random.choice([64, 128, 256])
    elif key == 'fc': child['fc'] = random.choice([128, 256, 512])
    return child

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6