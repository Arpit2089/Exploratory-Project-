# train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger("trainer", logfile="logs/trainer.log")

def train_model(model, train_loader, device="cpu", epochs=1, lr=0.01):
    model.to(device)
    model.train() 

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters found in model. Skipping training phase.")
        return

    criterion = nn.CrossEntropyLoss()
    
    # Use the dynamic Learning Rate
    optimizer = optim.SGD(
        trainable_params, 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4 
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Starting training for %d epochs on %s (LR: %f)", epochs, device, lr)

    for epoch in range(epochs):
        running_loss = 0.0
        batch_iter = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False, unit="batch")
        
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            running_loss += loss.item()
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        logger.debug("Epoch [%d/%d] completed. Avg Loss: %.4f", epoch + 1, epochs, running_loss / len(train_loader))

    logger.info("Training complete.")


# # train/trainer.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from utils.logger import get_logger

# logger = get_logger("trainer", logfile="logs/trainer.log")

# def train_model(model, train_loader, device="cpu", epochs=1):
#     """
#     Executes the training loop.
#     Modifies the model in-place.
#     """
#     model.to(device)
#     model.train() 

#     # Robustness Check: Prevent crash if evolutionary algorithm generated a parameter-less model
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
#     if not trainable_params:
#         logger.warning("No trainable parameters found in model. Skipping training phase.")
#         return

#     criterion = nn.CrossEntropyLoss()
    
#     optimizer = optim.SGD(
#         trainable_params, 
#         lr=0.01, 
#         momentum=0.9, 
#         weight_decay=1e-4 
#     )

#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

#     logger.info("Starting training for %d epochs on %s", epochs, device)

#     for epoch in range(epochs):
#         running_loss = 0.0
        
#         # Added tqdm progress bar for the batch loop
#         batch_iter = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False, unit="batch")
        
#         for batch_idx, (inputs, targets) in enumerate(batch_iter):
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
#             optimizer.step()
#             running_loss += loss.item()
            
#             # Update the progress bar with the current loss
#             batch_iter.set_postfix(loss=f"{loss.item():.4f}")

#         scheduler.step()
#         logger.debug("Epoch [%d/%d] completed. Avg Loss: %.4f", epoch + 1, epochs, running_loss / len(train_loader))

#     logger.info("Training complete.")

# # # train/trainer.py
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from utils.logger import get_logger

# # logger = get_logger("trainer", logfile="logs/trainer.log")

# # def train_model(model, train_loader, device="cpu", epochs=1):
# #     """
# #     Executes the training loop.
# #     Modifies the model in-place.
# #     """
# #     model.to(device)
# #     model.train() 

# #     # Robustness Check: Prevent crash if evolutionary algorithm generated a parameter-less model
# #     trainable_params = [p for p in model.parameters() if p.requires_grad]
# #     if not trainable_params:
# #         logger.warning("No trainable parameters found in model. Skipping training phase.")
# #         return

# #     criterion = nn.CrossEntropyLoss()
    
# #     optimizer = optim.SGD(
# #         trainable_params, 
# #         lr=0.01, 
# #         momentum=0.9, 
# #         weight_decay=1e-4 
# #     )

# #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# #     logger.info("Starting training for %d epochs on %s", epochs, device)

# #     for epoch in range(epochs):
# #         running_loss = 0.0
# #         for batch_idx, (inputs, targets) in enumerate(train_loader):
# #             inputs, targets = inputs.to(device), targets.to(device)

# #             optimizer.zero_grad()
# #             outputs = model(inputs)
# #             loss = criterion(outputs, targets)
# #             loss.backward()

# #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
# #             optimizer.step()
# #             running_loss += loss.item()

# #         scheduler.step()
# #         logger.debug("Epoch [%d/%d] completed. Avg Loss: %.4f", epoch + 1, epochs, running_loss / len(train_loader))

# #     logger.info("Training complete.")

# # # # train/trainer.py
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from utils.logger import get_logger

# # # logger = get_logger("trainer", logfile="logs/trainer.log")

# # # def train_model(model, train_loader, device="cpu", epochs=1):
# # #     """
# # #     Executes the training loop. Modifies the model in-place.
# # #     """
# # #     model.to(device)
# # #     model.train() # FIX: Explicitly set train mode for BatchNorm/Dropout

# # #     criterion = nn.CrossEntropyLoss()
    
# # #     # FIX: Added weight_decay to prevent larger architectures from 
# # #     # unfairly overfitting within the short 3-epoch window.
# # #     optimizer = optim.SGD(
# # #         model.parameters(), 
# # #         lr=0.01, 
# # #         momentum=0.9, 
# # #         weight_decay=1e-4 
# # #     )

# # #     # FIX: A flat LR overshoots in 3 epochs. CosineAnnealing gently 
# # #     # ramps down the LR so the model converges even on short runs.
# # #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# # #     logger.info("Starting training for %d epochs on %s", epochs, device)

# # #     for epoch in range(epochs):
# # #         running_loss = 0.0
# # #         for batch_idx, (inputs, targets) in enumerate(train_loader):
# # #             inputs, targets = inputs.to(device), targets.to(device)

# # #             optimizer.zero_grad()
# # #             outputs = model(inputs)
# # #             loss = criterion(outputs, targets)
# # #             loss.backward()

# # #             # FIX: Gradient Clipping. Newly initialized layers (from net2deeper or shape-fixes) 
# # #             # can produce massive gradients early on. This prevents loss spikes.
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
# # #             optimizer.step()
# # #             running_loss += loss.item()

# # #         scheduler.step()
# # #         logger.debug("Epoch [%d/%d] completed. Avg Loss: %.4f", epoch + 1, epochs, running_loss / len(train_loader))

# # #     logger.info("Training complete.")