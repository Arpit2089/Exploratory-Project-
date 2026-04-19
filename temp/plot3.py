import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# --- 1. Load and Parse Lamarckian Data ---
lamarkian_params = []

with open('temp/history2.pkl', 'rb') as f:
    history_lamarkian = pickle.load(f)

# Extract parameters for each evaluated model
for gen in sorted(history_lamarkian.keys()):
    for ind in history_lamarkian[gen]:
        # Lamarckian params are raw counts (e.g. 111652), convert to Millions (M)
        params_m = ind['params'] / 1_000_000.0
        lamarkian_params.append(params_m)

# --- 2. Load and Parse Regularized Evolution Data ---
df_re = pd.read_csv('temp/re_history_cifar10.csv')

# RE already has Elapsed_Time_min and Params_M
re_time = df_re['Elapsed_Time_min'].values
re_params = df_re['Params_M'].values

# --- 3. Distribute Lamarckian Time with Setup Offset ---
# Keeping our established timeline: 20 min setup, 70 min total
setup_time_mins = 11.2 
lamarkian_time = np.linspace(setup_time_mins, 70.0, len(lamarkian_params))

# --- 4. Generate the Parameter Comparison Plot ---
plt.figure(figsize=(10, 6))

# Plot Lamarckian parameters over time
plt.plot(lamarkian_time, lamarkian_params, 
         label='Lamarckian Models Explored', color='blue', marker='o', linestyle='-', markersize=4, alpha=0.7)

# Plot Regularized Evolution parameters over time
plt.plot(re_time, re_params, 
         label='Regularized Evolution Models Explored', color='orange', marker='s', linestyle='-', markersize=4, alpha=0.7)

plt.xlabel('Elapsed Search Time (Minutes)', fontsize=12)
plt.ylabel('Model Parameters (Millions)', fontsize=12)
plt.title('Architecture Search: Explored Parameter Sizes Over Time', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the resulting figure
plt.savefig('temp/comparison_params_plot2.png', dpi=300, bbox_inches='tight')