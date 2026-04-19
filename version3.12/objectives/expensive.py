# objectives/expensive.py
from train.trainer import train_model
from train.evaluate import evaluate_accuracy
from utils.logger import get_logger

logger = get_logger("expensive_obj", logfile="logs/expensive.log")


def evaluate_on_data(model, train_loader, val_loader,
                     device="cpu", epochs=1, lr=0.01,
                     weight_decay=1e-4, optimizer_name="sgd",
                     show_progress=True):
    """
    Train the model then evaluate and return val_error (lower is better).
    This is the ONLY function called for the expensive objective.
    """
    logger.info("Expensive eval: epochs=%d lr=%.4f opt=%s device=%s",
                epochs, lr, optimizer_name, device)

    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        show_progress=show_progress,
    )

    val_error = evaluate_accuracy(model, val_loader, device=device)
    return val_error