# train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import get_logger

logger = get_logger("trainer", logfile="logs/trainer.log")

# ---------------------------------------------------------------------------
# torch.compile — PyTorch 2.x graph-mode compiler.
# Enabled on Linux (Kaggle) where Triton is supported.
# Disabled on Windows where Triton crashes.
# ---------------------------------------------------------------------------
_COMPILE_AVAILABLE = hasattr(torch, "compile")
import platform
_IS_WINDOWS = platform.system() == "Windows"


def _try_compile(model: nn.Module) -> nn.Module:
    """
    Attempt torch.compile() on Linux/Kaggle for 10-30% speedup.
    Silently falls back to original model on failure or Windows.
    Skip for DataParallel models (unstable).
    """
    if _IS_WINDOWS:
        logger.debug("torch.compile disabled: Windows detected (Triton unsupported).")
        return model
    if not _COMPILE_AVAILABLE:
        return model
    if isinstance(model, nn.DataParallel):
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        logger.debug("torch.compile succeeded.")
        return compiled
    except Exception as e:
        logger.debug("torch.compile skipped: %s", e)
        return model


def _build_optimizer(model, optimizer_name: str, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    name = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        # SGD is best for ImageNet-family tasks
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True,
                         weight_decay=weight_decay)


def train_model(model, train_loader, device="cpu", epochs=1, lr=0.05,
                weight_decay=1e-4, optimizer_name="sgd",
                show_progress=True, use_amp=False):
    """
    Standard supervised training loop.
    Supports:
      - Single GPU (default)
      - Multi-GPU via nn.DataParallel (pass a DataParallel-wrapped model)
      - AMP fp16 (use_amp=True)
      - torch.compile on Linux/Kaggle (auto, single GPU only)

    Parameters
    ----------
    model          : nn.Module or nn.DataParallel
    train_loader   : DataLoader
    device         : "cpu" or "cuda"
    epochs         : number of full passes
    lr             : learning rate
    weight_decay   : L2 regularisation
    optimizer_name : "sgd" | "adam" | "adamw"
    show_progress  : print tqdm bar
    use_amp        : Automatic Mixed Precision — auto-disabled on CPU
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    use_amp = use_amp and is_cuda

    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        logger.warning("No trainable params — skipping training.")
        return

    # torch.compile: only for single-GPU raw models on Linux
    if is_cuda and not isinstance(model, nn.DataParallel):
        model = _try_compile(model)

    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    # label_smoothing=0.1 helps on 200-class tasks (Tiny ImageNet)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # GradScaler for AMP — no-op when use_amp=False
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    is_dp = isinstance(model, nn.DataParallel)
    logger.info(
        "train_model: epochs=%d lr=%.4f opt=%s device=%s amp=%s dp=%s",
        epochs, lr, optimizer_name, device, use_amp, is_dp,
    )

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
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # set_to_none=True is faster than zeroing (avoids memset)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets)

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
    # NOTE: No sleep calls here. Kaggle/cloud GPUs have proper cooling.
    # The time.sleep(30)/time.sleep(120) that previously existed were
    # the primary cause of extreme slowness and have been removed.