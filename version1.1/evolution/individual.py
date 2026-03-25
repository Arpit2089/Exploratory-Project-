# evolution/individual.py

from architectures.compiler import CompiledModel
from objectives.cheap import count_parameters, estimate_flops
from objectives.expensive import evaluate_accuracy   # NEW
from utils.logger import get_logger

logger = get_logger("individual", logfile="logs/individual.log")


class Individual:
    _id_counter = 0

    def __init__(self, graph):
        self.id = Individual._id_counter
        Individual._id_counter += 1

        self.graph = graph
        self.model = None

        self.f_cheap = None
        self.f_exp = None   # validation error / accuracy

        logger.info("Created Individual %d", self.id)

    # ---------------- Build Model ----------------
    def build_model(self):
        if self.model is None:
            logger.info("Building model for Individual %d", self.id)
            self.model = CompiledModel(self.graph)
        return self.model

    # ---------------- Cheap Objectives ----------------
    def evaluate_cheap(self, input_size=(1, 3, 32, 32)):
        if self.f_cheap is not None:
            logger.debug(
                "Cheap objectives cached for Individual %d: %s",
                self.id, self.f_cheap
            )
            return self.f_cheap

        model = self.build_model()
        logger.info("Evaluating cheap objectives for Individual %d", self.id)

        params = count_parameters(model)
        flops = estimate_flops(model, input_size=input_size)

        self.f_cheap = {
            "params": params,
            "flops": flops
        }

        logger.info(
            "Cheap objectives for Individual %d: %s",
            self.id, self.f_cheap
        )

        return self.f_cheap

    # ---------------- Expensive Objective ----------------
    def evaluate_expensive(
        self,
        train_loader,
        val_loader,
        device="cpu",
        epochs=1
    ):
        if self.f_exp is not None:
            logger.debug(
                "Expensive objective cached for Individual %d: %s",
                self.id, self.f_exp
            )
            return self.f_exp

        logger.info(
            "Evaluating EXPENSIVE objective (training) for Individual %d",
            self.id
        )

        model = self.build_model()

        val_error = evaluate_accuracy(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs
        )

        self.f_exp = {
            "val_error": val_error
        }

        logger.info(
            "Expensive objective for Individual %d: %s",
            self.id, self.f_exp
        )

        return self.f_exp
