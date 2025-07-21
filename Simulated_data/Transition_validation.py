import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# Load your DataFrame (replace with your actual file path)
df = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/HMM_output_adjusted_notransitions.csv')  # Update this path as needed

# Increment 'Cluster' by 1, leave 'transition_label' as is
df['Cluster_plus1'] = df['transition_label']

# Custom mapping for 'Cluster' column
cluster_map = {1: 2, 2: 3, 0: 1}
df['Cluster_mapped'] = df['Cluster'].map(cluster_map)

print(f"Available columns in DataFrame: {df.columns.tolist()}")
if 'labels' not in df.columns:
    raise KeyError("Column 'labels' not found in the DataFrame. Available columns: " + str(df.columns.tolist()))

# Align predicted and simulated clusters using Hungarian algorithm
# Convert string-based simulated labels ('A', 'B', 'C') to integer labels (0, 1, 2)
sim_labels, _ = pd.factorize(df['Cluster'])
pred_labels = df['labels'].astype(int).values

# Build confusion matrix for alignment
D = max(sim_labels.max(), pred_labels.max()) + 1
cost_matrix = np.zeros((D, D))
for i in range(D):
    for j in range(D):
        cost_matrix[i, j] = -np.sum((sim_labels == i) & (pred_labels == j))
row_ind, col_ind = linear_sum_assignment(cost_matrix)
# Build mapping dict
mapping = {j: i for i, j in zip(row_ind, col_ind)}
aligned_pred = np.array([mapping.get(lbl, lbl) for lbl in pred_labels])

# Compute metrics
acc = accuracy_score(sim_labels, aligned_pred)
nmi = normalized_mutual_info_score(sim_labels, aligned_pred)

# Confusion matrices
conf_matrix_raw = pd.crosstab(sim_labels, aligned_pred)
conf_matrix_norm = conf_matrix_raw.div(conf_matrix_raw.sum(axis=1), axis=0)

# Plot raw confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_raw, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted (aligned)')
plt.ylabel('Simulated')
plt.title('Raw Confusion Matrix')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/Cluster_vs_Transition_Heatmap_raw.png')
plt.close()

# Plot normalized confusion matrix with metrics
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted (aligned)')
plt.ylabel('Simulated')
plt.title(f'Normalized Confusion Matrix\nAccuracy={acc:.3f}, NMI={nmi:.3f}')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/Cluster_vs_Transition_Heatmap_normalized.png')
plt.close()

print(f'Optimal state mapping (predicted → simulated): {mapping}')
print(f'Accuracy: {acc:.4f}')
print(f'NMI: {nmi:.4f}')
