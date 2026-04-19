# evolution/individual.py
import uuid
import torch
from architectures.compiler import CompiledModel
from utils.logger import get_logger

logger = get_logger("individual", logfile="logs/individual.log")


class Individual:
    def __init__(self, graph):
        self.id       = uuid.uuid4().hex[:8]
        self.graph    = graph
        self.model    = None        # set by build_model(); cleared before pickling
        self.f_cheap  = None        # dict: e.g. {"params": ..., "flops": ...}
        self.f_exp    = None        # dict: {"val_error": float}
        self.op_name  = None        # set by lemonade loop ("exact" | "approx")
        logger.debug("Created Individual %s", self.id)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def build_model(self, input_shape=(1, 3, 32, 32)) -> CompiledModel:
        """
        Build (once) or return the cached CompiledModel.
        IMPORTANT: always use the returned reference — do not call
        build_model() expecting a different object each time.
        """
        if self.model is None:
            logger.debug("Building CompiledModel for Individual %s", self.id)
            self.model = CompiledModel(self.graph, input_shape=input_shape)
        return self.model

    # ------------------------------------------------------------------
    # Cheap objectives
    # ------------------------------------------------------------------

    def evaluate_cheap(self, objective_keys=("params", "flops"),
                       input_size=(1, 3, 32, 32)):
        """
        Compute cheap objectives.

        If objective_keys contains only "params" → uses fast graph-based
        counter, NO model build required.

        If objective_keys contains "flops" → builds model and runs thop.

        Results are cached in self.f_cheap after the first call.
        """
        if self.f_cheap is not None:
            return self.f_cheap

        result = {}

        if "params" in objective_keys:
            from objectives.cheap import count_params_from_graph
            result["params"] = count_params_from_graph(self.graph)

        if "flops" in objective_keys:
            from objectives.cheap import estimate_flops
            model = self.build_model(input_shape=input_size)
            model.eval()
            with torch.no_grad():
                result["flops"] = estimate_flops(model, input_size=input_size)

        self.f_cheap = result
        logger.debug("Individual %s cheap: %s", self.id, self.f_cheap)
        return self.f_cheap

    # ------------------------------------------------------------------
    # Expensive objective  (called inside worker processes)
    # ------------------------------------------------------------------

    def evaluate_expensive(self, train_loader, val_loader,
                           device="cpu", epochs=1, lr=0.01,
                           weight_decay=1e-4, optimizer_name="sgd",
                           show_progress=True):
        """
        NOTE: This method is provided for completeness.
        In practice, the lemonade worker calls train_model / evaluate_accuracy
        directly on child.model to keep the model reference consistent.
        """
        if self.f_exp is not None:
            return self.f_exp

        model = self.build_model()
        try:
            from objectives.expensive import evaluate_on_data
            val_error = evaluate_on_data(
                model, train_loader, val_loader,
                device=device, epochs=epochs, lr=lr,
                weight_decay=weight_decay, optimizer_name=optimizer_name,
                show_progress=show_progress,
            )
            self.f_exp = {"val_error": val_error}
        finally:
            if self.model is not None:
                self.model.cpu()
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.f_exp