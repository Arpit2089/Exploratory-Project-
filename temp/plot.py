import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --- 1. Load and Parse Lamarckian Data ---
lamarkian_evals = []
lamarkian_best_acc = []
best_acc_lam = 0.0
eval_count_lam = 0


with open('temp/history2.pkl', 'rb') as f:
    history_lamarkian = pickle.load(f)

# Loop through sorted generations and extract validation error
for gen in sorted(history_lamarkian.keys()):
    for ind in history_lamarkian[gen]:
        # Convert validation error decimal to accuracy percentage
        acc = (1.0 - ind['val_error']) * 100.0
        
        # Track the best accuracy so far
        if acc > best_acc_lam:
            best_acc_lam = acc
            
        eval_count_lam += 1
        lamarkian_evals.append(eval_count_lam)
        lamarkian_best_acc.append(best_acc_lam)

# --- 2. Load and Parse Regularized Evolution Data ---
df_re = pd.read_csv('temp/re_history_cifar10.csv')
re_evals = []
re_best_acc = []
best_acc_re = 0.0
eval_count_re = 0

# Loop through sequential evaluations and track accuracy
for acc in df_re['Accuracy']:
    # Track the best accuracy so far
    if acc > best_acc_re:
        best_acc_re = acc
        
    eval_count_re += 1
    re_evals.append(eval_count_re)
    re_best_acc.append(best_acc_re)

# --- 3. Generate the Comparison Plot ---
plt.figure(figsize=(10, 6))

plt.plot(lamarkian_evals, lamarkian_best_acc, 
         label='Lamarckian (.pkl)', color='blue', marker='o', markersize=4)

plt.plot(re_evals, re_best_acc, 
         label='Regularized Evolution (.csv)', color='orange', marker='s', markersize=4)

plt.xlabel('Number of Models Evaluated', fontsize=12)
plt.ylabel('Best Accuracy So Far (%)', fontsize=12)
plt.title('Architecture Search Performance: Lamarckian vs Regularized Evolution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the resulting figure
plt.savefig('temp/comparison_plot2.png', dpi=300, bbox_inches='tight')