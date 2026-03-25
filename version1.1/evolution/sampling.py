# evolution/sampling.py
import numpy as np
from sklearn.neighbors import KernelDensity
from utils.logger import get_logger

logger = get_logger("sampling", logfile="logs/sampling.log")


class KDESampler:
    def __init__(self, bandwidth=0.2):
        self.bandwidth = bandwidth
        self.kde = None

    def fit(self, individuals):
        """
        Fit KDE on cheap objectives.
        individuals: list of Individual (with .f_cheap dict containing numeric keys)
        """
        X = []
        for ind in individuals:
            f = ind.f_cheap
            # adapt keys used in your project; here we assume 'params' and 'flops'
            X.append([float(f.get('params', 0.0)), float(f.get('flops', 0.0))])

        X = np.array(X, dtype=float)
        if X.size == 0:
            logger.warning("KDESampler.fit called with empty individuals list")
            self.kde = None
            return

        # stabilize dynamic range
        X = np.log1p(X)

        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
        self.kde.fit(X)
        logger.info("Fitted KDE on %d individuals", len(X))

    def _raw_score(self, individual):
        """
        Returns raw score = -log_density (higher -> sparser).
        This function never returns NaN/inf; it returns finite float.
        """
        if self.kde is None:
            # if kde not fitted, fallback to uniform-ish constant
            return 1.0

        x = np.array([[float(individual.f_cheap.get('params', 0.0)),
                       float(individual.f_cheap.get('flops', 0.0))]], dtype=float)
        x = np.log1p(x)
        try:
            log_density = float(self.kde.score_samples(x)[0])
        except Exception as e:
            logger.warning("KDE score_samples failed for individual: %s — %s", str(individual), str(e))
            # fallback: very small density (i.e., large score)
            return 1e6

        # convert to sparse-preference; handle -inf/nan
        if np.isnan(log_density) or np.isinf(log_density):
            logger.warning("KDE returned invalid log_density=%s for ind=%s", log_density, str(individual))
            return 1e6

        raw = -log_density
        if np.isnan(raw) or np.isinf(raw):
            logger.warning("Computed raw score invalid (%s) for ind=%s", raw, str(individual))
            return 1e6

        return float(raw)

    def sample(self, individuals, k):
        """
        Sample k individuals with probability proportional to inverse KDE density.
        Robust to NaN/inf, and falls back to uniform sampling if needed.
        """
        if len(individuals) == 0:
            return []

        # compute raw scores
        scores = np.array([self._raw_score(ind) for ind in individuals], dtype=float)

        # detect invalid entries
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            logger.warning("KDESampler: raw scores contained NaN/inf; sanitizing. scores=%s", scores)

        # make all non-negative by shifting min -> 0
        min_s = np.nanmin(scores)
        if np.isfinite(min_s) and min_s < 0:
            scores = scores - min_s

        # clip negatives (safety), add small epsilon
        eps = 1e-8
        scores = np.where(np.isfinite(scores), scores, 0.0)  # replace inf/nan with 0
        scores = np.clip(scores, 0.0, None) + eps

        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            # fallback to uniform sampling if KDE degenerated
            logger.warning("KDESampler: score sum not finite/positive (sum=%s). Falling back to uniform sampling.", total)
            probs = np.ones(len(individuals), dtype=float) / float(len(individuals))
        else:
            probs = scores / total

        # final sanity: ensure probs are non-negative and sum ~= 1
        probs = np.clip(probs, 0.0, 1.0)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones(len(individuals), dtype=float) / float(len(individuals))
        else:
            probs = probs / probs_sum

        chosen_idx = np.random.choice(len(individuals), size=k, replace=True, p=probs)
        logger.info("Sampled %d individuals via KDE (k=%d). probs_min=%.3e probs_max=%.3e", k, k, float(probs.min()), float(probs.max()))
        return [individuals[i] for i in chosen_idx]
