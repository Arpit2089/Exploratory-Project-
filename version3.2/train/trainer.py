# train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import get_logger

logger = get_logger("trainer", logfile="logs/trainer.log")


def _build_optimizer(model, optimizer_name: str, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    name   = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:  # sgd (default)
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def train_model(model, train_loader, device="cpu", epochs=1,
                lr=0.01, weight_decay=1e-4,
                optimizer_name="sgd",
                show_progress=True,
                use_amp=False):
    """
    Standard supervised training loop.

    Parameters
    ----------
    model          : nn.Module — trained IN-PLACE on *device*.
    train_loader   : DataLoader
    device         : "cpu" or "cuda"
    epochs         : number of full passes over train_loader
    lr             : learning rate
    weight_decay   : L2 regularisation
    optimizer_name : "sgd" | "adam" | "adamw"
    show_progress  : print tqdm bar (disable in subprocess workers)
    use_amp        : Automatic Mixed Precision — fp16 forward, fp32 gradients.
                     Gives 1.5-2x GPU speedup.  Automatically disabled on CPU.
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    use_amp = use_amp and is_cuda   # AMP only works on CUDA

    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        logger.warning("No trainable params — skipping training.")
        return

    optimizer  = _build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion  = nn.CrossEntropyLoss()

    # GradScaler for AMP — no-op if use_amp=False
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info("train_model: epochs=%d lr=%.4f opt=%s device=%s amp=%s",
                epochs, lr, optimizer_name, device, use_amp)

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches    = 0

        if show_progress:
            try:
                from tqdm import tqdm
                batch_iter = tqdm(train_loader,
                                  desc=f"Train [{epoch+1}/{epochs}]",
                                  leave=False, unit="batch")
            except ImportError:
                batch_iter = train_loader
        else:
            batch_iter = train_loader

        for inputs, targets in batch_iter:
            # non_blocking=True overlaps CPU→GPU transfer with GPU compute
            inputs  = inputs.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                # fp16 forward pass
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets)
                # fp32 backward (scaler converts grads back)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

            if show_progress and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg = running_loss / max(n_batches, 1)
        logger.debug("Epoch [%d/%d] avg_loss=%.4f", epoch + 1, epochs, avg)

    logger.info("train_model complete.")