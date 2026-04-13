# evolution/sampling.py
import numpy as np
from utils.logger import get_logger

logger = get_logger("sampling", logfile="logs/sampling.log")


class KDESampler:
    """
    Inverse-density sampler for multi-objective LEMONADE.

    Samples individuals with probability proportional to 1 / KDE_density,
    so sparse regions of the objective space are explored preferentially.

    Bandwidth is adaptive: starts wider (exploration) and narrows over
    generations (exploitation).
    """

    def __init__(self, base_bandwidth: float = 0.3):
        self.base_bandwidth = base_bandwidth
        self._kde           = None
        self._obj_keys: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, individuals, objective_keys: list, generation: int = 0):
        """
        Fit KDE on the cheap objective values of *individuals*.

        Parameters
        ----------
        individuals    : list[Individual] with .f_cheap populated
        objective_keys : which keys from f_cheap to use (e.g. ["params","flops"])
        generation     : current generation number (used for bandwidth annealing)
        """
        self._obj_keys = list(objective_keys)

        if not self._obj_keys:
            # No cheap objectives → uniform sampling; no KDE needed
            self._kde = None
            return

        X = self._build_feature_matrix(individuals)
        if X is None or len(X) == 0:
            self._kde = None
            return

        # Adaptive bandwidth: shrinks from base → base/3 over 50 generations
        bandwidth = max(self.base_bandwidth / (1.0 + generation * 0.02),
                        self.base_bandwidth / 3.0)

        try:
            from sklearn.neighbors import KernelDensity
            self._kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
            self._kde.fit(X)
            logger.debug("KDE fitted on %d individuals, bw=%.3f, gen=%d",
                         len(X), bandwidth, generation)
        except Exception as e:
            logger.warning("KDE fit failed: %s — falling back to uniform", e)
            self._kde = None

    def sample(self, individuals, k: int, allow_repeats: bool = False) -> list:
        """
        Sample k individuals with probability ∝ 1/density.

        allow_repeats=False (default) means each individual is selected at
        most once per call; set True only if len(individuals) < k.
        """
        n = len(individuals)
        if n == 0:
            return []

        k = min(k, n)
        probs = self._compute_probs(individuals)

        replace = allow_repeats or (k > n)
        indices = np.random.choice(n, size=k, replace=replace, p=probs)
        return [individuals[i] for i in indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, individuals):
        rows = []
        for ind in individuals:
            if ind.f_cheap is None:
                continue
            row = [float(ind.f_cheap.get(k, 0.0)) for k in self._obj_keys]
            if any(v != v for v in row):   # NaN check
                continue
            rows.append(row)
        if not rows:
            return None
        X = np.array(rows, dtype=float)
        X = np.log1p(X)                    # stabilise dynamic range
        return X

    def _raw_score(self, ind) -> float:
        """Returns −log_density (higher = sparser region)."""
        if self._kde is None or not self._obj_keys:
            return 1.0
        if ind.f_cheap is None:
            return 1.0
        row = np.array([[float(ind.f_cheap.get(k, 0.0)) for k in self._obj_keys]],
                       dtype=float)
        row = np.log1p(row)
        try:
            log_d = float(self._kde.score_samples(row)[0])
        except Exception:
            return 1e6
        if np.isnan(log_d) or np.isinf(log_d):
            return 1e6
        return float(-log_d)

    def _compute_probs(self, individuals) -> np.ndarray:
        scores = np.array([self._raw_score(ind) for ind in individuals],
                          dtype=float)

        # Sanitise
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.clip(scores - scores.min(), 0.0, None) + 1e-8

        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(len(individuals)) / len(individuals)

        probs = scores / total
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()
        return probs