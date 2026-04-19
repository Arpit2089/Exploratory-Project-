# train/evaluate.py
import torch
from utils.logger import get_logger

logger = get_logger("evaluate", logfile="logs/evaluate.log")


def evaluate_accuracy(model, val_loader, device="cpu", use_amp=False) -> float:
    """
    Returns validation error in [0, 1].  Lower is better.

    Parameters
    ----------
    use_amp : bool
        Use fp16 for inference — faster on GPU, no accuracy change.
        Automatically disabled if device is not CUDA.
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    use_amp = use_amp and is_cuda

    model.to(device)
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs  = inputs.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if total == 0:
        logger.warning("Empty validation loader — returning error=1.0")
        return 1.0

    acc       = correct / total
    val_error = 1.0 - acc
    logger.debug("Val accuracy=%.2f%%  val_error=%.4f", acc * 100, val_error)
    return val_error