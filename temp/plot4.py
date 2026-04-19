import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --- 1. Load and Parse Lamarckian Data (Keeping it Real) ---
lamarkian_params = []
lamarkian_val_errors = []

with open('temp/history2.pkl', 'rb') as f:
    history_lamarkian = pickle.load(f)

for gen in sorted(history_lamarkian.keys()):
    for ind in history_lamarkian[gen]:
        # Convert raw parameter counts to Millions
        params_m = ind['params'] / 1_000_000.0
        val_error = ind['val_error']
        
        lamarkian_params.append(params_m)
        lamarkian_val_errors.append(val_error)

# --- 2. Load and Parse Regularized Evolution Data ---
df_re = pd.read_csv('temp/re_history_cifar10.csv')

re_params = df_re['Params_M'].values
# Convert Accuracy (%) to Validation Error (decimal) to match Lamarckian
re_val_errors = 1.0 - (df_re['Accuracy'] / 100.0)

# --- 3. Generate Scatter Plot (Swapped Axes) ---
plt.figure(figsize=(10, 6))

# X-axis: Validation Error, Y-axis: Parameters
plt.scatter(lamarkian_val_errors, lamarkian_params, 
            label='Lamarckian Models', color='blue', alpha=0.7, marker='o', edgecolors='w', s=60)

plt.scatter(re_val_errors, re_params, 
            label='Regularized Evolution Models', color='orange', alpha=0.7, marker='s', edgecolors='w', s=60)

plt.xlabel('Validation Error (Lower is Better)', fontsize=12)
plt.ylabel('Model Parameters (Millions)', fontsize=12)
plt.title('Parameter Efficiency: Model Size vs Validation Error', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the resulting figure
plt.savefig('temp/error_vs_param_scatter2.png', dpi=300, bbox_inches='tight')