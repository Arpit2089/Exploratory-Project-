# enas/controller.py
"""
ENAS-Style REINFORCE Controller  —  guided operator selection.

Motivation (from the integration doc):
  "Use ENAS to sample architecture candidates" — instead of selecting
  morphism operators uniformly at random we train a small MLP policy
  (via REINFORCE) to predict which operator is most likely to produce
  a child that improves the Pareto front for the current parent.

Architecture
------------
State (5-dim):
    [norm_params, norm_flops, val_error, gen/20, pop_size/20]

Output: softmax distribution over 6 operators
    [net2deeper, net2wider, skip, prune, sepconv, remove]

Training (REINFORCE):
    reward = +1.0  child is non-dominated (enters Pareto front)
    reward = +0.5  child dominates ≥1 existing member but doesn't dominate all
    reward =  0.0  neutral outcome
    reward = −0.5  child is dominated
    reward = −1.0  child exceeds MAX_PARAMS (hard constraint violation)

Baseline: exponential moving average of past rewards (variance reduction).
Entropy regularisation encourages exploration early in search.

The controller is only used in the GPU sequential path (main process).
CPU parallel workers use fixed OP_WEIGHTS as before — controller weights
cannot be safely shared across subprocesses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("enas_controller", logfile="logs/controller.log")

# Operator names — MUST match the keys in OP_WEIGHTS in operators.py
OPERATOR_NAMES = ["net2deeper", "net2wider", "skip", "prune", "sepconv", "remove"]
NUM_OPS        = len(OPERATOR_NAMES)
_OP_INDEX      = {name: i for i, name in enumerate(OPERATOR_NAMES)}


# =============================================================================
# Small MLP policy network
# =============================================================================

class _ControllerNet(nn.Module):
    """
    2-layer MLP: state → operator log-probabilities.
    Small by design — the NAS search budget is the bottleneck, not this.
    """
    def __init__(self, state_dim: int = 5, hidden: int = 64, num_ops: int = NUM_OPS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_ops),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-softmax probabilities over operators."""
        return torch.log_softmax(self.net(x), dim=-1)


# =============================================================================
# Public controller class
# =============================================================================

class ENASController:
    """
    REINFORCE-trained controller for morphism operator selection.

    Typical usage::

        ctrl = ENASController()

        # During candidate generation:
        op_name, log_prob = ctrl.select_operator(parent_ind, gen, pop_size)

        # After child is trained and Pareto reward is known:
        reward = ctrl.compute_reward(trained_child, population, max_params)
        ctrl.record_outcome(log_prob, reward)

        # Periodically (every update_every children):
        ctrl.update()   # runs REINFORCE gradient step
    """

    def __init__(self,
                 state_dim:      int   = 5,
                 hidden:         int   = 64,
                 lr:             float = 3e-4,
                 entropy_coef:   float = 0.05,
                 baseline_decay: float = 0.95,
                 update_every:   int   = 8):
        self._net          = _ControllerNet(state_dim, hidden, NUM_OPS)
        self._optim        = optim.Adam(self._net.parameters(), lr=lr)
        self._entropy_coef = entropy_coef
        self._update_every = update_every

        # Running baseline for REINFORCE variance reduction
        self._baseline       = 0.0
        self._baseline_decay = baseline_decay

        # Episode buffer: list of (log_prob_tensor, reward_float)
        self._buffer: List[Tuple[torch.Tensor, float]] = []

        # Normalisation constants (updated adaptively)
        self._max_params: float = 30_000_000.0
        self._max_flops:  float = 1_000_000_000.0

        logger.info("ENASController ready — ops: %s", OPERATOR_NAMES)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_operator(self,
                        individual,
                        generation: int,
                        pop_size:   int,
                        temperature: float = 1.0
                        ) -> Tuple[str, torch.Tensor]:
        """
        Sample an operator for *individual* using the current policy.

        Parameters
        ----------
        individual   : Individual — parent to mutate
        generation   : current generation index
        pop_size     : current population size
        temperature  : >1 for more exploration, <1 for exploitation

        Returns
        -------
        (op_name, log_prob)
            log_prob is a differentiable scalar tensor — keep it alive
            until you call record_outcome().
        """
        state = self._state_vector(individual, generation, pop_size)

        # Forward pass WITHOUT grad first — just to read probabilities
        with torch.no_grad():
            log_probs_nograd = self._net(state) / max(temperature, 1e-6)
            probs = torch.exp(log_probs_nograd).clamp(min=1e-8)
            probs = probs / probs.sum()

        # Sample operator index
        idx = int(torch.multinomial(probs, 1).item())

        # Forward pass WITH grad — needed for REINFORCE update
        log_probs_grad = self._net(state)
        log_prob       = log_probs_grad[idx]   # scalar, requires_grad=True

        op_name = OPERATOR_NAMES[idx]
        logger.debug("Controller: gen=%d op=%s p=%.3f ind=%s",
                     generation, op_name, probs[idx].item(), individual.id)
        return op_name, log_prob

    def record_outcome(self, log_prob: torch.Tensor, reward: float):
        """
        Store (log_prob, reward) pair for the next REINFORCE update.
        Call this after every child has been trained and evaluated.
        """
        self._buffer.append((log_prob, float(reward)))

    def update(self) -> Optional[float]:
        """
        Run a REINFORCE gradient step if enough outcomes have been buffered.

        Returns the scalar loss value, or None if the buffer is too small.
        """
        if len(self._buffer) < self._update_every:
            return None

        rewards      = [r for _, r in self._buffer]
        mean_r       = float(np.mean(rewards))
        self._baseline = (self._baseline_decay * self._baseline
                          + (1.0 - self._baseline_decay) * mean_r)

        # REINFORCE loss: -E[log_pi(a|s) * (R - baseline)]
        policy_loss = torch.zeros(1, requires_grad=False)
        for log_prob, reward in self._buffer:
            advantage   = reward - self._baseline
            # Detach advantage — only log_prob carries gradient
            policy_loss = policy_loss - log_prob * float(advantage)

        policy_loss = policy_loss / len(self._buffer)

        # Entropy bonus — encourages exploration
        dummy_state = torch.zeros(1, 5)
        log_probs   = self._net(dummy_state)
        probs       = torch.exp(log_probs)
        entropy     = -(probs * log_probs).sum()
        total_loss  = policy_loss - self._entropy_coef * entropy

        self._optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
        self._optim.step()

        loss_val = total_loss.item()
        self._buffer.clear()
        logger.info("Controller REINFORCE update: loss=%.4f baseline=%.3f "
                    "mean_reward=%.3f", loss_val, self._baseline, mean_r)
        return loss_val

    def force_update(self) -> Optional[float]:
        """Flush any remaining buffer — call at end of each generation."""
        if not self._buffer:
            return None
        # Temporarily lower the threshold
        original = self._update_every
        self._update_every = 1
        result = self.update()
        self._update_every = original
        return result

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(self,
                       child,
                       population: list,
                       max_params: int) -> float:
        """
        Compute REINFORCE reward for a trained child architecture.

        Reward signal:
            +1.0 : child is non-dominated (improves Pareto front)
            +0.5 : child is non-dominated within population but doesn't
                   dominate any existing member (pure diversity gain)
             0.0 : neutral
            -0.5 : child is dominated by ≥1 existing member
            -1.0 : child exceeds MAX_PARAMS (hard constraint)
        """
        from evolution.pareto import dominates, _get_all_objectives

        # Hard constraint: params budget
        if child.f_cheap:
            p = child.f_cheap.get("params", 0)
            if p > 0:
                self._max_params = max(self._max_params, p)
            if p > max_params:
                return -1.0

        child_objs = _get_all_objectives(child)
        if not child_objs:
            return 0.0

        dominated_by_count = 0
        dominates_count    = 0

        for ind in population:
            pop_objs = _get_all_objectives(ind)
            if dominates(pop_objs, child_objs):
                dominated_by_count += 1
            elif dominates(child_objs, pop_objs):
                dominates_count += 1

        if dominated_by_count == 0 and dominates_count > 0:
            return 1.0
        elif dominated_by_count == 0:
            return 0.5
        elif dominates_count > 0:
            return 0.0
        else:
            return -0.5

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def operator_probs(self, individual, generation: int, pop_size: int) -> dict:
        """Return current probability distribution as a dict (for logging)."""
        state = self._state_vector(individual, generation, pop_size)
        with torch.no_grad():
            log_probs = self._net(state)
            probs     = torch.exp(log_probs).tolist()
        return dict(zip(OPERATOR_NAMES, probs))

    def state_dict(self) -> dict:
        return {
            "net":      self._net.state_dict(),
            "optim":    self._optim.state_dict(),
            "baseline": self._baseline,
        }

    def load_state_dict(self, d: dict):
        self._net.load_state_dict(d["net"])
        self._optim.load_state_dict(d["optim"])
        self._baseline = d.get("baseline", 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_vector(self, individual, generation: int, pop_size: int
                      ) -> torch.Tensor:
        """Build a normalised 5-dimensional state tensor."""
        params  = 0.0
        flops   = 0.0
        val_err = 0.5   # default: unknown

        if individual.f_cheap:
            params = individual.f_cheap.get("params", 0) / max(self._max_params, 1.0)
            flops  = individual.f_cheap.get("flops",  0) / max(self._max_flops,  1.0)
        if individual.f_exp:
            val_err = float(individual.f_exp.get("val_error", 0.5))

        gen_norm  = min(float(generation) / 20.0, 1.0)
        pop_norm  = min(float(pop_size)   / 20.0, 1.0)

        return torch.tensor(
            [params, flops, val_err, gen_norm, pop_norm],
            dtype=torch.float32
        )