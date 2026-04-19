import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_re_plots(csv_path, dataset_name):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    
    # --- PLOT 1: Aging (Time vs Accuracy) ---
    plt.figure(figsize=(10, 6))
    
    # Red Dotted line for Current Pop
    plt.plot(df['Elapsed_Time_min'], df['Current_Pop_Best'], 
             label='Current Population Best', color='red', linestyle=':', linewidth=3, marker='o', markersize=4)
    
    # Solid Blue line for All-Time Best
    plt.plot(df['Elapsed_Time_min'], df['Overall_Best_So_Far'], 
             label='Best Model Found So Far', color='blue', linewidth=2.5)
    
    plt.title(f'RE Performance over Time ({dataset_name.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Elapsed Time (Minutes)', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f're_aging_{dataset_name}.png', dpi=300)
    print(f"Saved -> re_aging_{dataset_name}.png")
    
    # --- PLOT 2: Trajectory (Time vs Parameters) ---
    # Since X must be time, we plot the parameters of every evaluated model over time, 
    # colored by its accuracy to show the "migration" toward complexity.
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(df['Elapsed_Time_min'], df['Params_M'], 
                          c=df['Accuracy'], cmap='viridis', alpha=0.8, edgecolor='k')
    
    plt.colorbar(scatter, label='Validation Accuracy (%)')
    plt.title(f'Search Space Trajectory over Time ({dataset_name.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Elapsed Time (Minutes)', fontsize=12, fontweight='bold')
    plt.ylabel('Model Complexity (Millions of Parameters)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f're_trajectory_{dataset_name}.png', dpi=300)
    print(f"Saved -> re_trajectory_{dataset_name}.png")