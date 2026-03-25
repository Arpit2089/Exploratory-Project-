# train/trainer.py (minimal)
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

def train_finetune(model, train_loader, val_loader, device='cuda', epochs=2, lr=1e-3, freeze_prefix=None):
    model.to(device)
    if freeze_prefix:
        for name,param in model.named_parameters():
            if name.startswith(freeze_prefix): param.requires_grad=False
    opt = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for e in range(epochs):
        model.train()
        for x,y in tqdm(train_loader):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward(); opt.step()
    # compute validation error
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)
    val_err = 1.0 - (correct/total)
    return val_err
