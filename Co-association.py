import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(hmm_path, vkm_path):
    """Load HMM and VKM data from CSV files."""
    try:
        hmm_df = pd.read_csv(hmm_path)
        vkm_df = pd.read_csv(vkm_path)
    except Exception as e:
        raise IOError(f"Error loading files: {e}")
    return hmm_df, vkm_df

def validate_columns(df, required_columns):
    """Ensure DataFrame contains required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def merge_data(hmm_df, vkm_df, merge_keys):
    """Merge HMM and VKM DataFrames."""
    for df, name in zip([hmm_df, vkm_df], ['HMM', 'VKM']):
        validate_columns(df, merge_keys)
    return pd.merge(hmm_df, vkm_df, on=merge_keys)

def compute_optimal_mapping(merged_df, source_col, target_col):
    """Compute optimal mapping using Hungarian algorithm."""
    cross_tab = pd.crosstab(merged_df[source_col], merged_df[target_col])
    cost_matrix = -cross_tab.values  # Convert to minimization problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {cross_tab.index[r]: cross_tab.columns[c] for r, c in zip(row_ind, col_ind)}, cross_tab

def force_binary_mapping(transitional_df, transition_col, cluster_col):
    """Force transition labels to map exclusively to clusters 1 and 2."""
    # Create filtered cross-tab for clusters 1 and 2
    ct = pd.crosstab(transitional_df[transition_col], transitional_df[cluster_col])
    ct = ct[[c for c in ct.columns if str(c) in {'1', '2'}]]
    
    # Handle empty case
    if ct.empty:
        return {}

    # Hungarian algorithm on filtered clusters
    cost_matrix = -ct.values
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    # Create base mapping
    mapping = {ct.index[row]: ct.columns[col] for row, col in zip(row_idx, col_idx)}
    
    # Ensure all transitions get mapped (handle zero-count cases)
    all_transitions = transitional_df[transition_col].unique()
    for t in all_transitions:
        if t not in mapping:
            # Get counts for clusters 1 and 2
            counts = ct.loc[t] if t in ct.index else pd.Series({'1':0, '2':0})
            mapping[t] = '1' if counts.get('1', 0) >= counts.get('2', 0) else '2'
    
    return mapping

# Configuration
hmm_path = "/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv"
vkm_path = "/Users/a_fin/Desktop/Year 4/Project/Data/VKM_output.csv"
merge_keys = ['Subject', 'Week', 'Session', 'Condition']
cluster_col = 'cluster_label'
transition_col = 'transition_label'

# Load and merge data
hmm_df, vkm_df = load_data(hmm_path, vkm_path)
merged_df = merge_data(hmm_df, vkm_df, merge_keys)

# Phase 1: Process transitional labels
transition_mask = merged_df[transition_col].astype(str).str.contains('to')
transitional_subset = merged_df[transition_mask].copy()

# Force mapping to VKM clusters 1 and 2
transition_mapping = force_binary_mapping(transitional_subset, transition_col, cluster_col)
merged_df.loc[transition_mask, transition_col] = merged_df.loc[transition_mask, transition_col].map(transition_mapping)

# Phase 2: Global alignment
final_mapping, cross_tab = compute_optimal_mapping(merged_df, transition_col, cluster_col)
merged_df['aligned_cluster'] = merged_df[transition_col].map(final_mapping)

# Verification and visualization
verification_ct = pd.crosstab(merged_df['aligned_cluster'], merged_df[cluster_col])
plt.figure(figsize=(10, 6))
sns.heatmap(verification_ct, annot=True, fmt='d', cmap='Blues')
plt.title('Final Cluster Alignment')
plt.xlabel('VKM Clusters')
plt.ylabel('Aligned Clusters')
plt.show()