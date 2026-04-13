# train/distill.py
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.logger import get_logger

logger = get_logger("distill", logfile="logs/distill.log")


def train_with_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader,
    device="cpu",
    epochs=1,
    temperature=3.0,
    alpha=0.0,       # 0.0 = pure KD alignment, 1.0 = pure CE hard labels
    lr=0.001,
    weight_decay=1e-4,
    optimizer_name="sgd",
    show_progress=True,
    use_amp=False,
):
    """
    Knowledge-distillation alignment phase for ANM children.

    Called BEFORE standard training to recover accuracy lost by approximate
    morphisms (prune, sepconv, remove).

    Parameters
    ----------
    use_amp : bool
        Automatic Mixed Precision — gives 1.5-2x GPU speedup.
        Automatically disabled if device is not CUDA.
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    use_amp = use_amp and is_cuda

    student_model.to(device)
    teacher_model.to(device)
    student_model.train()
    teacher_model.eval()

    # Freeze teacher completely
    for param in teacher_model.parameters():
        param.requires_grad = False

    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(reduction="batchmean")

    name = optimizer_name.lower()
    params = [p for p in student_model.parameters() if p.requires_grad]
    if name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9,
                              weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info(
        "Distillation: epochs=%d T=%.1f alpha=%.2f lr=%.4f device=%s amp=%s",
        epochs, temperature, alpha, lr, device, use_amp,
    )

    for epoch in range(epochs):
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iter = tqdm(train_loader,
                                  desc=f"Distill [{epoch+1}/{epochs}]",
                                  leave=False, unit="batch")
            except ImportError:
                batch_iter = train_loader
        else:
            batch_iter = train_loader

        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs  = inputs.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    student_logits = student_model(inputs)
                    with torch.no_grad():
                        teacher_logits = teacher_model(inputs)

                    loss_hard = criterion_hard(student_logits, targets)
                    s_log_p   = F.log_softmax(student_logits / temperature, dim=1)
                    t_p       = F.softmax(teacher_logits    / temperature, dim=1)
                    loss_soft = criterion_soft(s_log_p, t_p) * (temperature ** 2)
                    loss      = alpha * loss_hard + (1.0 - alpha) * loss_soft

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                               max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_logits = student_model(inputs)
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)

                loss_hard = criterion_hard(student_logits, targets)
                s_log_p   = F.log_softmax(student_logits / temperature, dim=1)
                t_p       = F.softmax(teacher_logits    / temperature, dim=1)
                loss_soft = criterion_soft(s_log_p, t_p) * (temperature ** 2)
                loss      = alpha * loss_hard + (1.0 - alpha) * loss_soft

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                               max_norm=5.0)
                optimizer.step()

            if show_progress and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            # Periodic cleanup to avoid accumulating graph memory
            if batch_idx % 20 == 0:
                gc.collect()

        scheduler.step()

    logger.info("Distillation complete.")