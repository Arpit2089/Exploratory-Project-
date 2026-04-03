# utils/plot.py
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from utils.logger import get_logger

logger = get_logger("plotter", logfile="logs/plotter.log")

def plot_pareto_evolution(history, obj_x, obj_y, save_dir="logs"):
    """
    Plots the convergence of the Pareto front over generations for a pair of objectives.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    generations = sorted(list(history.keys()))
    
    # Create a color gradient from light to dark to represent time (generations)
    colors = cm.viridis(np.linspace(0.3, 1, len(generations)))
    
    for idx, gen in enumerate(generations):
        points = []
        for ind_stats in history[gen]:
            x_val = ind_stats.get(obj_x)
            y_val = ind_stats.get(obj_y)
            
            if x_val is not None and y_val is not None:
                points.append((x_val, y_val))
        
        if not points:
            continue
            
        # Sort points by X-axis to draw a continuous Pareto front line
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # Only label the first, last, and even-numbered generations to prevent legend clutter
        label = f"Gen {gen}" if gen in [generations[0], generations[-1]] or gen % 2 == 0 else None
        
        # Scatter the individual models
        plt.scatter(xs, ys, color=colors[idx], label=label, alpha=0.7, s=50)
        # Draw a dashed line connecting the Pareto front for this generation
        plt.plot(xs, ys, color=colors[idx], alpha=0.4, linestyle='--')

    plt.title(f"Pareto Front Convergence: {obj_x.upper()} vs {obj_y.upper()}", fontsize=14, pad=15)
    plt.xlabel(obj_x.upper(), fontsize=12)
    plt.ylabel(obj_y.upper(), fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if len(generations) > 1:
        plt.legend(title="Generations")
        
    save_path = os.path.join(save_dir, f"pareto_{obj_x}_vs_{obj_y}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved Pareto convergence plot to %s", save_path)

def plot_all_pairs(history, save_dir="logs"):
    """
    Automatically generates 2D Pareto plots for all objective pairs.
    """
    pairs = [
        ('flops', 'val_error'),
        ('params', 'val_error'),
        ('params', 'flops')
    ]
    logger.info("Generating Pareto front visualization plots...")
    for obj_x, obj_y in pairs:
        plot_pareto_evolution(history, obj_x, obj_y, save_dir)