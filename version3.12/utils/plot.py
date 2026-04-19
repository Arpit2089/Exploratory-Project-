# utils/plot.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.logger import get_logger

logger = get_logger("plotter", logfile="logs/plotter.log")


# =============================================================================
# Internal helpers
# =============================================================================

def _extract_series(history):
    """
    Parse the history dict into a structured form.
    history[gen] = list of record dicts with keys:
        id, params, flops, val_error, model_path, graph_path
    """
    generations = sorted(history.keys())
    return generations


def _safe_val(record, key):
    v = record.get(key)
    return float(v) if v is not None else None


# =============================================================================
# 2-D Pareto evolution plots
# =============================================================================

def plot_pareto_2d(history, obj_x, obj_y, save_dir="logs"):
    """
    Plot Pareto front evolution for a pair of objectives.
    One coloured curve per generation, fading light → dark over time.
    """
    os.makedirs(save_dir, exist_ok=True)
    generations = _extract_series(history)
    if not generations:
        logger.warning("Empty history — skipping plot %s vs %s", obj_x, obj_y)
        return

    colors = cm.viridis(np.linspace(0.2, 1.0, len(generations)))
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, gen in enumerate(generations):
        records = history[gen]
        points = []
        for r in records:
            xv = _safe_val(r, obj_x)
            yv = _safe_val(r, obj_y)
            if xv is not None and yv is not None:
                points.append((xv, yv))

        if not points:
            continue

        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        label = f"Gen {gen}" if (idx == 0 or idx == len(generations) - 1
                                  or gen % max(1, len(generations) // 5) == 0) \
                else None

        ax.scatter(xs, ys, color=colors[idx], alpha=0.8, s=55, label=label)
        ax.plot(xs, ys, color=colors[idx], alpha=0.35, linestyle="--")

    ax.set_title(f"Pareto Front: {obj_x} vs {obj_y}", fontsize=14)
    ax.set_xlabel(obj_x, fontsize=12)
    ax.set_ylabel(obj_y, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(generations) > 1:
        ax.legend(title="Generations", bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=9)

    path = os.path.join(save_dir, f"pareto_{obj_x}_vs_{obj_y}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved 2D Pareto plot → %s", path)


def plot_all_pairs(history, cheap_objectives, save_dir="logs"):
    """
    Generate all meaningful 2-D Pareto plots based on which cheap objectives
    were actually used in this run.

    Always plots each cheap objective vs val_error.
    If both params and flops are used, also plots params vs flops.
    """
    obj_x_y_pairs = [(obj, "val_error") for obj in cheap_objectives]
    if "params" in cheap_objectives and "flops" in cheap_objectives:
        obj_x_y_pairs.append(("params", "flops"))

    if not obj_x_y_pairs:
        # Single-objective run — only val_error available
        logger.info("No cheap objectives configured — skipping 2D pair plots")
        return

    for ox, oy in obj_x_y_pairs:
        try:
            plot_pareto_2d(history, ox, oy, save_dir=save_dir)
        except Exception as e:
            logger.error("plot_pareto_2d(%s, %s) failed: %s", ox, oy, e)


# =============================================================================
# 3-D Pareto evolution plot  (requires both params and flops)
# =============================================================================

def plot_3d_pareto(history, save_dir="logs"):
    """
    3-D scatter (params, flops, val_error) coloured by generation.
    Silently skips if any axis data is unavailable.
    """
    generations = _extract_series(history)
    if not generations:
        return

    # Check if we have 3D data
    sample_gen = history[generations[0]]
    has_flops  = any(_safe_val(r, "flops")     is not None for r in sample_gen)
    has_params = any(_safe_val(r, "params")    is not None for r in sample_gen)
    has_err    = any(_safe_val(r, "val_error") is not None for r in sample_gen)
    if not (has_flops and has_params and has_err):
        logger.info("3D plot skipped — missing params/flops/val_error in history")
        return

    os.makedirs(save_dir, exist_ok=True)
    fig  = plt.figure(figsize=(12, 9))
    ax   = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))

    has_data = False
    for idx, gen in enumerate(generations):
        records = history[gen]
        pts = [
            (r["params"], r["flops"], r["val_error"])
            for r in records
            if r.get("params") is not None
            and r.get("flops")  is not None
            and r.get("val_error") is not None
        ]
        if not pts:
            continue
        has_data = True
        ps, fs, vs = zip(*pts)
        ax.scatter(ps, fs, vs,
                   color=colors[idx], label=f"Gen {gen}",
                   alpha=0.9, s=50, edgecolor="w", linewidth=0.4)

        if len(pts) >= 3:
            try:
                ax.plot_trisurf(list(ps), list(fs), list(vs),
                                color=colors[idx], alpha=0.18, edgecolor="none")
            except Exception:
                pass

    if not has_data:
        plt.close()
        return

    ax.set_title("3D Pareto Front Evolution", fontsize=14, pad=15)
    ax.set_xlabel("Parameters", fontsize=10, labelpad=8)
    ax.set_ylabel("FLOPs",      fontsize=10, labelpad=8)
    ax.set_zlabel("Val Error",  fontsize=10, labelpad=8)
    ax.view_init(elev=20, azim=-45)
    if len(generations) > 1:
        ax.legend(title="Generations",
                  bbox_to_anchor=(1.12, 0.9), loc="upper left", fontsize=8)

    path = os.path.join(save_dir, "pareto_3d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved 3D Pareto plot → %s", path)


# =============================================================================
# Val-error convergence curve
# =============================================================================

def plot_convergence(history, save_dir="logs"):
    """
    Plot best / median val_error per generation to visualise convergence.
    """
    generations = _extract_series(history)
    if not generations:
        return

    os.makedirs(save_dir, exist_ok=True)
    best_errs   = []
    median_errs = []

    for gen in generations:
        errs = [
            r["val_error"] for r in history[gen]
            if r.get("val_error") is not None
        ]
        if errs:
            best_errs.append(min(errs))
            median_errs.append(float(np.median(errs)))
        else:
            best_errs.append(None)
            median_errs.append(None)

    fig, ax = plt.subplots(figsize=(9, 5))
    valid_best   = [(g, v) for g, v in zip(generations, best_errs)   if v is not None]
    valid_median = [(g, v) for g, v in zip(generations, median_errs) if v is not None]

    if valid_best:
        ax.plot(*zip(*valid_best),   "b-o", label="Best val_error",   linewidth=2)
    if valid_median:
        ax.plot(*zip(*valid_median), "r--s", label="Median val_error", linewidth=1.5)

    ax.set_title("Val-Error Convergence per Generation", fontsize=13)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Validation Error", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    path = os.path.join(save_dir, "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved convergence plot → %s", path)