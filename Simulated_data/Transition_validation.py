import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# Load your DataFrame (replace with your actual file path)
# This should be the output file from your HMM analysis of the 4-state simulated data
df = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/HMM_output_adjusted_notransitions.csv')  # Update this path as needed

print(f"Available columns in DataFrame: {df.columns.tolist()}")
if 'labels' not in df.columns:
    raise KeyError("Column 'labels' not found in the DataFrame. Available columns: " + str(df.columns.tolist()))
if 'Cluster' not in df.columns:
    raise KeyError("Column 'Cluster' (ground truth) not found in the DataFrame. Available columns: " + str(df.columns.tolist()))

# Align predicted and simulated clusters using Hungarian algorithm
# Convert string-based simulated labels ('A', 'B', 'C', etc.) to integer labels
sim_labels, sim_categories = pd.factorize(df['Cluster'])
pred_labels = df['labels'].astype(int).values

# Build cost matrix for alignment (negative of confusion matrix)
D = max(sim_labels.max(), pred_labels.max()) + 1
cost_matrix = np.zeros((D, D))
for i in range(len(sim_labels)):
    cost_matrix[sim_labels[i], pred_labels[i]] -= 1

# Find optimal assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Build mapping dict from predicted to simulated
# The mapping ensures that the predicted label 'j' is mapped to the true label 'i'
# that maximizes overlap.
mapping = {col: row for row, col in zip(row_ind, col_ind)}
aligned_pred = np.array([mapping.get(lbl, -1) for lbl in pred_labels]) # Use -1 for unmapped

# Create aligned labels for the confusion matrix, preserving original category names
aligned_pred_named = [sim_categories[i] if i != -1 else "Unmapped" for i in aligned_pred]
sim_labels_named = df['Cluster']

# Compute metrics
acc = accuracy_score(sim_labels, aligned_pred)
nmi = normalized_mutual_info_score(sim_labels, aligned_pred)

# Confusion matrices
# Use the named labels for crosstab to get a readable matrix
conf_matrix_raw = pd.crosstab(sim_labels_named, aligned_pred_named, rownames=['Simulated'], colnames=['Predicted (aligned)'])
# Reorder to match simulation state order if needed
state_order = list(sim_categories)
if "Unmapped" in conf_matrix_raw.columns:
    state_order.append("Unmapped")
conf_matrix_raw = conf_matrix_raw.reindex(index=sim_categories, columns=state_order, fill_value=0)

conf_matrix_norm = conf_matrix_raw.div(conf_matrix_raw.sum(axis=1), axis=0)

# Plot raw confusion matrix
plt.figure(figsize=(8, 7))
sns.heatmap(conf_matrix_raw, annot=True, fmt='d', cmap='Blues')
plt.title('Raw Confusion Matrix')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/Cluster_vs_Transition_Heatmap_4states_raw.png')
plt.close()

# Plot normalized confusion matrix with metrics
plt.figure(figsize=(8, 7))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues')
plt.title(f'Normalized Confusion Matrix (4-State)\nAccuracy={acc:.3f}, NMI={nmi:.3f}')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/Cluster_vs_Transition_Heatmap_4states_normalized.png')
plt.close()

# Create a readable mapping from predicted state index to simulated state name
readable_mapping = {k: sim_categories[v] for k, v in mapping.items()}
print(f'Optimal state mapping (predicted index → simulated name): {readable_mapping}')
print(f'Accuracy: {acc:.4f}')
print(f'NMI: {nmi:.4f}')
