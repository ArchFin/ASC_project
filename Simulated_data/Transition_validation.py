import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your DataFrame (replace with your actual file path)
df = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/HMM_output_adjusted_notransitions.csv')  # Update this path as needed

# Increment 'Cluster' by 1, leave 'transition_label' as is
df['Cluster_plus1'] = df['Cluster'] + 1

# Custom mapping for 'Cluster' column
cluster_map = {1: 1, 2: 3, 0: 2}
df['Cluster_mapped'] = df['Cluster'].map(cluster_map)

# Create confusion matrix (crosstab)
conf_matrix = pd.crosstab(df['Cluster_mapped'], df['transition_label'])

# Normalize each row to sum to 1
conf_matrix_normalized = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Simulated')
plt.title('Mapped Cluster vs. Transition Label Heatmap')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/Cluster_vs_Transition_Heatmap_normalized.png')  # Save the figure
plt.show()
