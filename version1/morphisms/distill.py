# morphisms/distill.py
import torch
import torch.nn as nn
from utils.logger import get_logger
logger = get_logger("distill", logfile="logs/distill.log")

def train_student_with_distillation(parent_model, child_model, dataloader, device='cpu', epochs=3, lr=1e-3, temp=1.0, alpha=0.9):
    """
    Simple distillation: minimize MSE between parent logits and child logits (teacher-student).
    Also add a small supervised cross-entropy if labels are provided in the dataloader batch.
    dataloader yields (images, labels) where labels may be None.
    """
    parent_model = parent_model.to(device).eval()
    child_model = child_model.to(device).train()

    opt = torch.optim.Adam(child_model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            imgs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 and batch[1] is not None else None

            with torch.no_grad():
                t_logits = parent_model(imgs)
            s_logits = child_model(imgs)

            # ensure shapes compatible: flatten if necessary
            if t_logits.shape != s_logits.shape:
                # try a linear head or global pooling — but for our tests we expect same output shapes
                logger.warning("Teacher and student logits shapes differ: %s vs %s", t_logits.shape, s_logits.shape)
                # fallback: try to match by slicing
                min_shape = tuple(min(a,b) for a,b in zip(t_logits.shape, s_logits.shape))
                t_logits = t_logits[..., :min_shape[-1]]
                s_logits = s_logits[..., :min_shape[-1]]

            loss_distill = mse(s_logits, t_logits)
            loss = (alpha * loss_distill)
            if labels is not None:
                try:
                    loss_sup = ce(s_logits, labels)
                    loss = loss + (1 - alpha) * loss_sup
                except Exception as e:
                    logger.debug("Supervised CE could not be computed due to shape mismatch")

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            count += 1
        logger.info("Distill epoch %d loss=%.6f", epoch, total_loss / max(1,count))
    return child_model
