# objectives/expensive.py

import torch
from train.trainer import train_finetune
from utils.logger import get_logger

logger = get_logger("expensive_obj", logfile="logs/expensive.log")


def evaluate_accuracy(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=1,
):
    """
    Expensive objective:
    Trains model for few epochs and returns validation error.
    """

    logger.info("Starting expensive evaluation (epochs=%d)", epochs)

    device = torch.device(device)
    model = model.to(device)

    val_error = train_finetune(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs
    )

    logger.info("Finished expensive evaluation. Val error=%.4f", val_error)

    return val_error
