# train/evaluate.py
import torch
from utils.logger import get_logger

logger = get_logger("evaluate", logfile="logs/evaluate.log")

def evaluate_accuracy(model, val_loader, device="cpu"):
    """
    Calculates validation error. 
    Returns: float (error rate between 0.0 and 1.0, lower is better)
    """
    model.to(device)
    
    # FIX: Explicit eval mode. Without this, passing a batch through 
    # the network will corrupt BatchNorm running statistics.
    model.eval() 
    
    correct = 0
    total = 0

    # FIX: no_grad() prevents PyTorch from building a computation graph 
    # during validation, saving massive amounts of VRAM.
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if total == 0:
        logger.warning("Validation loader is empty! Returning 1.0 error.")
        return 1.0

    accuracy = correct / total
    val_error = 1.0 - accuracy
    
    logger.info("Evaluation complete. Accuracy: %.2f%%, Val Error: %.4f", accuracy * 100, val_error)
    
    return val_error