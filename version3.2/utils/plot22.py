################################################################################
# FILE:   plot_interactive_3d.py
# DESCRIPTION: Standalone script to load history.pkl and render a rotatable 
#              3D surface plot of the Pareto front with interactive checkboxes
#              and hover tooltips.
################################################################################

import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np

# Try importing mplcursors for the hover tooltips
try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False
    print("\n" + "="*65)
    print(" NOTICE: To see hover tooltips, you need to install 'mplcursors'.")
    print(" Please run this command in your terminal:")
    print(" pip install mplcursors")
    print("="*65 + "\n")

def plot_interactive_3d(history):
    """
    Renders an interactive 3D scatter and surface plot from the given history
    with checkboxes to toggle visibility of generations and hover tooltips.
    """
    fig = plt.figure(figsize=(13, 9))
    
    # Adjust layout to make room for checkboxes on the left
    plt.subplots_adjust(left=0.25)
    
    ax = fig.add_subplot(111, projection='3d')
    
    generations = sorted(list(history.keys()))
    
    # Use a colormap to differentiate generations visually
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
    has_data = False
    
    # Dictionaries to store plotted elements for toggling
    plot_elements = {}
    labels = []
    visibility = []
    
    for idx, gen in enumerate(generations):
        pop = history[gen]
        
        # Safely extract objectives
        params = [ind.get('params') for ind in pop if ind.get('params') is not None]
        flops = [ind.get('flops') for ind in pop if ind.get('flops') is not None]
        val_error = [ind.get('val_error') for ind in pop if ind.get('val_error') is not None] 
        
        if len(params) == len(flops) == len(val_error) and len(params) > 0:
            has_data = True
            
            label_name = f"Gen {gen}"
            labels.append(label_name)
            visibility.append(True)
            plot_elements[label_name] = []
            
            # 1. Scatter the exact models as solid dots (Made dots smaller: s=25)
            scatter = ax.scatter(
                params, flops, val_error, 
                color=colors[idx], label=label_name, 
                alpha=1.0, s=25, edgecolor='w', linewidth=0.5
            )
            plot_elements[label_name].append(scatter)
            
            # 2. Draw the 3D Surface Mesh (Curvature) connecting the models
            if len(params) >= 3:
                try:
                    surf = ax.plot_trisurf(
                        params, flops, val_error, 
                        color=colors[idx], alpha=0.3, edgecolor='none'
                    )
                    plot_elements[label_name].append(surf)
                except Exception as e:
                    print(f"Skipped surface mesh for Gen {gen} due to collinear points: {e}")
    
    if not has_data:
        print("ERROR: Cannot plot 3D surface. Missing one or more required objectives (params, flops, val_error).")
        plt.close()
        return

    # Formatting and labels
    ax.set_title("Interactive 3D Pareto Front Evolution", fontsize=16, pad=20)
    ax.set_xlabel("Parameters", fontsize=12, labelpad=10)
    ax.set_ylabel("FLOPs", fontsize=12, labelpad=10)
    ax.set_zlabel("Validation Error", fontsize=12, labelpad=10)
    
    # Initial camera angle
    ax.view_init(elev=20, azim=-45)
    
    # --- Checkbox Implementation ---
    # Determine axes size for checkboxes based on number of generations
    height = min(0.8, 0.05 * len(labels))
    ax_check = plt.axes([0.02, 0.5 - height/2, 0.15, height])
    
    # Create the CheckButtons widget
    check = CheckButtons(ax_check, labels, visibility)
    
    # Define the toggle logic
    def toggle_visibility(label):
        # Flip the visibility of both scatter and surface components for the clicked generation
        for element in plot_elements[label]:
            element.set_visible(not element.get_visible())
        fig.canvas.draw_idle()
        
    check.on_clicked(toggle_visibility)
    
    # Keep a reference to the widget so it doesn't get garbage collected
    fig.check = check 

    # --- Hover Tooltips Implementation ---
    if HAS_MPLCURSORS:
        # Collect only the scatter objects (the 0th index in plot_elements lists) to attach tooltips
        scatter_plots = [elements[0] for elements in plot_elements.values()]
        
        # Attach the hover cursor
        cursor = mplcursors.cursor(scatter_plots, hover=True)
        
        @cursor.connect("add")
        def on_add(sel):
            x, y, z = sel.target
            # Format the text to display the specific values smoothly
            sel.annotation.set_text(f"Params: {x:g}\nFLOPs: {y:g}\nError: {z:.4g}")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.85)

    print("Opening interactive 3D plot window...")
    print("-> USE CHECKBOXES on the left to toggle visibility.")
    if HAS_MPLCURSORS:
        print("-> HOVER over the dots to see exact coordinates.")
    print("-> CLICK AND DRAG the plot to rotate.")
    print("-> SCROLL to zoom.")
    print("-> Close the window to terminate the script.")
    
    # This triggers the interactive window
    plt.show()

def main():
    # =========================================================================
    # CONFIGURATION: Set your specific timestamp / experiment folder here
    # =========================================================================
    RUN_DIR = r"results_CIFAR-100_20260412_174406" 
    # 100_20260412_174406
    # RUN_DIR = r"results_CIFAR-10_20260412_111857" 
    # results_CIFAR-100_20260412_191158
    
    history_path = os.path.join(RUN_DIR, "history", "history.pkl")
    
    if not os.path.exists(history_path):
        print(f"ERROR: Cannot find history file at {history_path}")
        print("Please check the RUN_DIR path and ensure the experiment saved a history.pkl file.")
        return
        
    print(f"Loading history from {history_path}...")
    with open(history_path, "rb") as f:
        history = pickle.load(f)
        
    print(f"Successfully loaded {len(history)} generations.")
    plot_interactive_3d(history)

if __name__ == "__main__":
    main()


# ################################################################################
# # FILE:   plot_interactive_3d.py
# # DESCRIPTION: Standalone script to load history.pkl and render a rotatable 
# #              3D surface plot of the Pareto front with interactive checkboxes.
# ################################################################################

# import os
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib.widgets import CheckButtons
# import numpy as np

# def plot_interactive_3d(history):
#     """
#     Renders an interactive 3D scatter and surface plot from the given history
#     with checkboxes to toggle visibility of generations.
#     """
#     fig = plt.figure(figsize=(13, 9))
    
#     # Adjust layout to make room for checkboxes on the left
#     plt.subplots_adjust(left=0.25)
    
#     ax = fig.add_subplot(111, projection='3d')
    
#     generations = sorted(list(history.keys()))
    
#     # Use a colormap to differentiate generations visually
#     colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
#     has_data = False
    
#     # Dictionaries to store plotted elements for toggling
#     plot_elements = {}
#     labels = []
#     visibility = []
    
#     for idx, gen in enumerate(generations):
#         pop = history[gen]
        
#         # Safely extract objectives
#         params = [ind.get('params') for ind in pop if ind.get('params') is not None]
#         flops = [ind.get('flops') for ind in pop if ind.get('flops') is not None]
#         val_error = [ind.get('val_error') for ind in pop if ind.get('val_error') is not None] 
        
#         if len(params) == len(flops) == len(val_error) and len(params) > 0:
#             has_data = True
            
#             label_name = f"Gen {gen}"
#             labels.append(label_name)
#             visibility.append(True)
#             plot_elements[label_name] = []
            
#             # 1. Scatter the exact models as solid dots
#             scatter = ax.scatter(
#                 params, flops, val_error, 
#                 color=colors[idx], label=label_name, 
#                 alpha=1.0, s=60, edgecolor='w', linewidth=0.5
#             )
#             plot_elements[label_name].append(scatter)
            
#             # 2. Draw the 3D Surface Mesh (Curvature) connecting the models
#             if len(params) >= 3:
#                 try:
#                     surf = ax.plot_trisurf(
#                         params, flops, val_error, 
#                         color=colors[idx], alpha=0.3, edgecolor='none'
#                     )
#                     plot_elements[label_name].append(surf)
#                 except Exception as e:
#                     print(f"Skipped surface mesh for Gen {gen} due to collinear points: {e}")
    
#     if not has_data:
#         print("ERROR: Cannot plot 3D surface. Missing one or more required objectives (params, flops, val_error).")
#         plt.close()
#         return

#     # Formatting and labels
#     ax.set_title("Interactive 3D Pareto Front Evolution", fontsize=16, pad=20)
#     ax.set_xlabel("Parameters", fontsize=12, labelpad=10)
#     ax.set_ylabel("FLOPs", fontsize=12, labelpad=10)
#     ax.set_zlabel("Validation Error", fontsize=12, labelpad=10)
    
#     # Initial camera angle
#     ax.view_init(elev=20, azim=-45)
    
#     # --- Checkbox Implementation ---
#     # Determine axes size for checkboxes based on number of generations
#     height = min(0.8, 0.05 * len(labels))
#     ax_check = plt.axes([0.02, 0.5 - height/2, 0.15, height])
    
#     # Create the CheckButtons widget
#     check = CheckButtons(ax_check, labels, visibility)
    
#     # Define the toggle logic
#     def toggle_visibility(label):
#         # Flip the visibility of both scatter and surface components for the clicked generation
#         for element in plot_elements[label]:
#             element.set_visible(not element.get_visible())
#         fig.canvas.draw_idle()
        
#     check.on_clicked(toggle_visibility)
    
#     # Keep a reference to the widget so it doesn't get garbage collected
#     fig.check = check 
    
#     print("Opening interactive 3D plot window...")
#     print("-> USE CHECKBOXES on the left to toggle visibility.")
#     print("-> CLICK AND DRAG the plot to rotate.")
#     print("-> SCROLL to zoom.")
#     print("-> Close the window to terminate the script.")
    
#     # This triggers the interactive window
#     plt.show()

# def main():
#     # =========================================================================
#     # CONFIGURATION: Set your specific timestamp / experiment folder here
#     # =========================================================================
#     RUN_DIR = r"results_CIFAR-10_20260412_111857" 
    
#     history_path = os.path.join(RUN_DIR, "history", "history.pkl")
    
#     if not os.path.exists(history_path):
#         print(f"ERROR: Cannot find history file at {history_path}")
#         print("Please check the RUN_DIR path and ensure the experiment saved a history.pkl file.")
#         return
        
#     print(f"Loading history from {history_path}...")
#     with open(history_path, "rb") as f:
#         history = pickle.load(f)
        
#     print(f"Successfully loaded {len(history)} generations.")
#     plot_interactive_3d(history)

# if __name__ == "__main__":
#     main()

# # ################################################################################
# # # FILE:   plot_interactive_3d.py
# # # DESCRIPTION: Standalone script to load history.pkl and render a rotatable 
# # #              3D surface plot of the Pareto front.
# # ################################################################################

# # import os
# # import pickle
# # import matplotlib.pyplot as plt
# # import numpy as np

# # def plot_interactive_3d(history):
# #     """
# #     Renders an interactive 3D scatter and surface plot from the given history.
# #     """
# #     fig = plt.figure(figsize=(12, 9))
# #     ax = fig.add_subplot(111, projection='3d')
    
# #     generations = sorted(list(history.keys()))
    
# #     # Use a colormap to differentiate generations visually
# #     colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
# #     has_data = False
    
# #     for idx, gen in enumerate(generations):
# #         pop = history[gen]
        
# #         # Safely extract objectives
# #         params = [ind.get('params') for ind in pop if ind.get('params') is not None]
# #         flops = [ind.get('flops') for ind in pop if ind.get('flops') is not None]
# #         val_error = [ind.get('val_error') for ind in pop if ind.get('val_error') is not None] 
        
# #         if len(params) == len(flops) == len(val_error) and len(params) > 0:
# #             has_data = True
            
# #             # 1. Scatter the exact models as solid dots
# #             ax.scatter(
# #                 params, flops, val_error, 
# #                 color=colors[idx], label=f"Gen {gen}", 
# #                 alpha=1.0, s=60, edgecolor='w', linewidth=0.5
# #             )
            
# #             # 2. Draw the 3D Surface Mesh (Curvature) connecting the models
# #             if len(params) >= 3:
# #                 try:
# #                     ax.plot_trisurf(
# #                         params, flops, val_error, 
# #                         color=colors[idx], alpha=0.3, edgecolor='none'
# #                     )
# #                 except Exception as e:
# #                     print(f"Skipped surface mesh for Gen {gen} due to collinear points: {e}")
    
# #     if not has_data:
# #         print("ERROR: Cannot plot 3D surface. Missing one or more required objectives (params, flops, val_error).")
# #         plt.close()
# #         return

# #     # Formatting and labels
# #     ax.set_title("Interactive 3D Pareto Front Evolution", fontsize=16, pad=20)
# #     ax.set_xlabel("Parameters", fontsize=12, labelpad=10)
# #     ax.set_ylabel("FLOPs", fontsize=12, labelpad=10)
# #     ax.set_zlabel("Validation Error", fontsize=12, labelpad=10)
    
# #     # Initial camera angle (can be rotated interactively once opened)
# #     ax.view_init(elev=20, azim=-45)
    
# #     if len(generations) > 1:
# #         # Move legend outside the main graph area
# #         ax.legend(title="Generations", bbox_to_anchor=(1.15, 0.9), loc='upper left')
        
# #     print("Opening interactive 3D plot window...")
# #     print("-> CLICK AND DRAG to rotate.")
# #     print("-> SCROLL to zoom.")
# #     print("-> Close the window to terminate the script.")
    
# #     # This triggers the interactive window
# #     plt.show()

# # def main():
# #     # =========================================================================
# #     # CONFIGURATION: Set your specific timestamp / experiment folder here
# #     # =========================================================================
# #     RUN_DIR = r"results_CIFAR-10_20260412_111857" 
# #     # results_CIFAR-10_20260412_111857
    
# #     history_path = os.path.join(RUN_DIR, "history", "history.pkl")
    
# #     if not os.path.exists(history_path):
# #         print(f"ERROR: Cannot find history file at {history_path}")
# #         print("Please check the RUN_DIR path and ensure the experiment saved a history.pkl file.")
# #         return
        
# #     print(f"Loading history from {history_path}...")
# #     with open(history_path, "rb") as f:
# #         history = pickle.load(f)
        
# #     print(f"Successfully loaded {len(history)} generations.")
# #     plot_interactive_3d(history)

# # if __name__ == "__main__":
# #     main()