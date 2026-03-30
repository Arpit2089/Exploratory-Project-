# train/distill.py
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
    alpha=0.5
):
    """
    Trains a student model using soft targets from a frozen teacher model.
    Args:
        student_model: The new child Architecture (to be trained).
        teacher_model: The parent Architecture (frozen).
        train_loader: DataLoader for the training set.
        device: "cpu" or "cuda".
        epochs: Number of epochs to train.
        temperature: Softens the teacher's probability distribution (higher = softer).
        alpha: Weighting between the hard loss (true labels) and soft loss (teacher labels).
               alpha=0.5 means a 50/50 split.
    """
    student_model.to(device)
    teacher_model.to(device)
    
    # The student learns; the teacher only predicts.
    student_model.train()
    teacher_model.eval()

    # Hard targets (Standard Cross Entropy)
    criterion_hard = nn.CrossEntropyLoss()
    # Soft targets (Kullback-Leibler Divergence)
    criterion_soft = nn.KLDivLoss(reduction='batchmean')
    
    # Setup matching the standard trainer.py to ensure fair comparisons
    optimizer = optim.SGD(
        student_model.parameters(), 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=1e-4 
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Starting Distillation: epochs=%d, T=%.1f, alpha=%.2f on %s", 
                epochs, temperature, alpha, device)

    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 1. Forward pass for the student
            student_logits = student_model(inputs)
            
            # 2. Forward pass for the frozen teacher
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            # 3. Calculate Hard Loss (Student vs. True Labels)
            loss_hard = criterion_hard(student_logits, targets)

            # 4. Calculate Soft Loss (Student vs. Teacher)
            # Log-Softmax for the student, standard Softmax for the teacher
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            
            # Multiply by T^2 to scale the gradients back up
            loss_soft = criterion_soft(student_log_probs, teacher_probs) * (temperature ** 2)

            # 5. Combine Losses
            loss = (alpha * loss_hard) + ((1.0 - alpha) * loss_soft)

            # 6. Backward pass and Optimization
            loss.backward()
            
            # Gradient clipping to match trainer.py and prevent loss spikes
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5.0)
            
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        logger.debug("Distillation Epoch [%d/%d] completed. Avg Loss: %.4f", epoch + 1, epochs, running_loss / len(train_loader))

    logger.info("Distillation training complete.")

# # train/distill.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from utils.logger import get_logger

# logger = get_logger("distill", logfile="logs/distill.log")

# def train_with_distillation(
#     student_model: nn.Module,
#     teacher_model: nn.Module,
#     train_loader,
#     device="cpu",
#     epochs=1,
#     temperature=3.0,
#     alpha=0.5
# ):
#     """
#     Trains a student model using soft targets from a frozen teacher model.
    
#     Args:
#         student_model: The new child Architecture (to be trained).
#         teacher_model: The parent Architecture (frozen).
#         train_loader: DataLoader for the training set.
#         device: "cpu" or "cuda".
#         epochs: Number of epochs to train.
#         temperature: Softens the teacher's probability distribution (higher = softer).
#         alpha: Weighting between the hard loss (true labels) and soft loss (teacher labels).
#                alpha=0.5 means a 50/50 split.
#     """
#     student_model.to(device)
#     teacher_model.to(device)
    
#     # The student learns; the teacher only predicts.
#     student_model.train()
#     teacher_model.eval()

#     # Hard targets (Standard Cross Entropy)
#     criterion_hard = nn.CrossEntropyLoss()
#     # Soft targets (Kullback-Leibler Divergence)
#     criterion_soft = nn.KLDivLoss(reduction='batchmean')
    
#     # Setup matching the standard trainer.py to ensure fair comparisons
#     optimizer = optim.SGD(
#         student_model.parameters(), 
#         lr=0.01, 
#         momentum=0.9, 
#         weight_decay=1e-4 
#     )
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

#     logger.info("Starting Distillation: epochs=%d, T=%.1f, alpha=%.2f on %s", 
#                 epochs, temperature, alpha, device)

#     for epoch in range(epochs):
#         running_loss = 0.0
        
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()

#             # 1. Forward pass for the student
#             student_logits = student_model(inputs)
            
#             # 2. Forward pass for the frozen teacher
#             with torch.no_grad():
#                 teacher_logits = teacher_model(inputs)

#             # 3. Calculate Hard Loss (Student vs. True Labels)
#             loss_hard = criterion_hard(student_logits, targets)

#             # 4. Calculate Soft Loss (Student vs. Teacher)
#             # Log-Softmax for the student, standard