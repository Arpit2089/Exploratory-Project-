
# evolution/individual.py
import uuid
import torch
from architectures.compiler import CompiledModel
from objectives.cheap import count_parameters, estimate_flops
from utils.logger import get_logger

logger = get_logger("individual", logfile="logs/individual.log")


class Individual:
    def __init__(self, graph):
        self.id = uuid.uuid4().hex[:8]
        self.graph = graph
        self.model = None
        self.f_cheap = None
        self.f_exp = None   

        # Dynamically assigned during generation to track morphism type
        self.op_name = None 

        logger.info("Created Individual %s", self.id)

    # ---------------- Build Model ----------------
    def build_model(self):
        if self.model is None:
            logger.info("Building model for Individual %s", self.id)
            self.model = CompiledModel(self.graph)
        return self.model

    # ---------------- Cheap Objectives ----------------
    def evaluate_cheap(self, input_size=(1, 3, 32, 32)):
        if self.f_cheap is not None:
            return self.f_cheap

        model = self.build_model()
        logger.info("Evaluating cheap objectives for Individual %s", self.id)

        model.eval()
        
        with torch.no_grad():
            params = count_parameters(model)
            flops = estimate_flops(model, input_size=input_size)

        self.f_cheap = {
            "params": params,
            "flops": flops
        }

        logger.info("Cheap objectives for Individual %s: %s", self.id, self.f_cheap)
        return self.f_cheap

    # ---------------- Expensive Objective ----------------
    def evaluate_expensive(
        self,
        train_loader,
        val_loader,
        device="cpu",
        epochs=1,
        lr=0.01 
    ):
        if self.f_exp is not None:
            return self.f_exp

        logger.info("Evaluating EXPENSIVE objective (training) for Individual %s", self.id)
        model = self.build_model()

        try:
            from objectives.expensive import evaluate_accuracy
            
            val_error = evaluate_accuracy(
                model,
                train_loader,
                val_loader,
                device=device,
                epochs=epochs,
                lr=lr  
            )
            
            self.f_exp = {
                "val_error": val_error
            }

            logger.info("Expensive objective for Individual %s: %s", self.id, self.f_exp)
            
        finally:
            if self.model is not None:
                self.model.cpu()
                
            if str(device) != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.f_exp

# # evolution/individual.py
# import uuid
# import torch
# from architectures.compiler import CompiledModel
# from objectives.cheap import count_parameters, estimate_flops
# from objectives.expensive import evaluate_accuracy   # NEW
# from utils.logger import get_logger

# logger = get_logger("individual", logfile="logs/individual.log")


# class Individual:
#     def __init__(self, graph):
#         # FIX: Worker-safe unique ID generation. No more class-level counters.
#         self.id = uuid.uuid4().hex[:8]

#         self.graph = graph
#         self.model = None

#         self.f_cheap = None
#         self.f_exp = None   # validation error / accuracy

#         logger.info("Created Individual %s", self.id)

#     # ---------------- Build Model ----------------
#     def build_model(self):
#         if self.model is None:
#             logger.info("Building model for Individual %s", self.id)
#             self.model = CompiledModel(self.graph)
#         return self.model

#     # ---------------- Cheap Objectives ----------------
#     def evaluate_cheap(self, input_size=(1, 3, 32, 32)):
#         if self.f_cheap is not None:
#             logger.debug(
#                 "Cheap objectives cached for Individual %s: %s",
#                 self.id, self.f_cheap
#             )
#             return self.f_cheap

#         model = self.build_model()
#         logger.info("Evaluating cheap objectives for Individual %s", self.id)

#         # FIX: Model MUST be in eval mode for FLOP counting, otherwise 
#         # BatchNorm layers will crash on batch_size=1 or corrupt running stats.
#         model.eval()
        
#         with torch.no_grad():
#             params = count_parameters(model)
#             flops = estimate_flops(model, input_size=input_size)

#         self.f_cheap = {
#             "params": params,
#             "flops": flops
#         }

#         logger.info(
#             "Cheap objectives for Individual %s: %s",
#             self.id, self.f_cheap
#         )

#         return self.f_cheap

#     # ---------------- Expensive Objective ----------------
#     def evaluate_expensive(
#         self,
#         train_loader,
#         val_loader,
#         device="cpu",
#         epochs=1,
#         teacher_model=None,
#         lr=0.01  # <--- ADDED: Default learning rate parameter
#     ):
#         if self.f_exp is not None:
#             logger.debug(
#                 "Expensive objective cached for Individual %s: %s",
#                 self.id, self.f_exp
#             )
#             return self.f_exp

#         logger.info(
#             "Evaluating EXPENSIVE objective (training) for Individual %s",
#             self.id
#         )

#         model = self.build_model()

#         try:
#             # Import here if not already imported at the top of individual.py
#             from objectives.expensive import evaluate_accuracy
            
#             val_error = evaluate_accuracy(
#                 model,
#                 train_loader,
#                 val_loader,
#                 device=device,
#                 epochs=epochs,
#                 teacher_model=teacher_model,
#                 lr=lr  # <--- ADDED: Passing the learning rate down to the trainer
#             )
            
#             self.f_exp = {
#                 "val_error": val_error
#             }

#             logger.info(
#                 "Expensive objective for Individual %s: %s",
#                 self.id, self.f_exp
#             )
            
#         finally:
#             # =========================================================
#             # MEMORY CLEANUP BLOCK: The new code goes exactly here
#             # =========================================================
            
#             # 1. Push the child model back to CPU
#             if self.model is not None:
#                 self.model.cpu()
                
#             # 2. Push the teacher model back to CPU (This is the new addition)
#             if teacher_model is not None:
#                 teacher_model.cpu() 
                
#             # 3. Force PyTorch to empty the VRAM cache
#             if str(device) != "cpu" and torch.cuda.is_available():
#                 torch.cuda.empty_cache()
                
#             # =========================================================

#         return self.f_exp

# # evolution/individual.py
# import uuid
# import torch
# from architectures.compiler import CompiledModel
# from objectives.cheap import count_parameters, estimate_flops
# from objectives.expensive import evaluate_accuracy   # NEW
# from utils.logger import get_logger

# logger = get_logger("individual", logfile="logs/individual.log")


# class Individual:
#     def __init__(self, graph):
#         # FIX: Worker-safe unique ID generation. No more class-level counters.
#         self.id = uuid.uuid4().hex[:8]

#         self.graph = graph
#         self.model = None

#         self.f_cheap = None
#         self.f_exp = None   # validation error / accuracy

#         logger.info("Created Individual %s", self.id)

#     # ---------------- Build Model ----------------
#     def build_model(self):
#         if self.model is None:
#             logger.info("Building model for Individual %s", self.id)
#             self.model = CompiledModel(self.graph)
#         return self.model

#     # ---------------- Cheap Objectives ----------------
#     def evaluate_cheap(self, input_size=(1, 3, 32, 32)):
#         if self.f_cheap is not None:
#             logger.debug(
#                 "Cheap objectives cached for Individual %s: %s",
#                 self.id, self.f_cheap
#             )
#             return self.f_cheap

#         model = self.build_model()
#         logger.info("Evaluating cheap objectives for Individual %s", self.id)

#         # FIX: Model MUST be in eval mode for FLOP counting, otherwise 
#         # BatchNorm layers will crash on batch_size=1 or corrupt running stats.
#         model.eval()
        
#         with torch.no_grad():
#             params = count_parameters(model)
#             flops = estimate_flops(model, input_size=input_size)

#         self.f_cheap = {
#             "params": params,
#             "flops": flops
#         }

#         logger.info(
#             "Cheap objectives for Individual %s: %s",
#             self.id, self.f_cheap
#         )

#         return self.f_cheap

#     # ---------------- Expensive Objective ----------------
#     def evaluate_expensive(
#         self,
#         train_loader,
#         val_loader,
#         device="cpu",
#         epochs=1,
#         teacher_model=None
#     ):
#         if self.f_exp is not None:
#             logger.debug(
#                 "Expensive objective cached for Individual %s: %s",
#                 self.id, self.f_exp
#             )
#             return self.f_exp

#         logger.info(
#             "Evaluating EXPENSIVE objective (training) for Individual %s",
#             self.id
#         )

#         model = self.build_model()

#         try:
#             # Import here if not already imported at the top of individual.py
#             from objectives.expensive import evaluate_accuracy
            
#             val_error = evaluate_accuracy(
#                 model,
#                 train_loader,
#                 val_loader,
#                 device=device,
#                 epochs=epochs,
#                 teacher_model=teacher_model
#             )
            
#             self.f_exp = {
#                 "val_error": val_error
#             }

#             logger.info(
#                 "Expensive objective for Individual %s: %s",
#                 self.id, self.f_exp
#             )
            
#         finally:
#             # =========================================================
#             # MEMORY CLEANUP BLOCK: The new code goes exactly here
#             # =========================================================
            
#             # 1. Push the child model back to CPU
#             if self.model is not None:
#                 self.model.cpu()
                
#             # 2. Push the teacher model back to CPU (This is the new addition)
#             if teacher_model is not None:
#                 teacher_model.cpu() 
                
#             # 3. Force PyTorch to empty the VRAM cache
#             if str(device) != "cpu" and torch.cuda.is_available():
#                 torch.cuda.empty_cache()
                
#             # =========================================================

#         return self.f_exp