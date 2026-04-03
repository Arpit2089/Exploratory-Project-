# objectives/expensive.py
from train.trainer import train_model
from train.evaluate import evaluate_accuracy as calculate_val_error
from utils.logger import get_logger

logger = get_logger("expensive_obj", logfile="logs/expensive.log")

def evaluate_accuracy(model, train_loader, val_loader, device="cpu", epochs=1, lr=0.01):
    logger.info("Starting expensive evaluation pipeline.")
    
    # Strictly standard training for the expensive objective
    train_model(model, train_loader, device=device, epochs=epochs, lr=lr)
    
    val_error = calculate_val_error(model, val_loader, device=device)
    return val_error
