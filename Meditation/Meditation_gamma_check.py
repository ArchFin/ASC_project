import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HMM.HMM_methods import CustomHMMClustering, principal_component_finder, csv_splitter
import yaml
from joblib import Parallel, delayed

# --- Config & Paths ---
meditation_csv = '/Users/a_fin/Desktop/Year 4/Project/Data/Meditation_TET_data_labelled_noThought.csv'
config_path     = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/Meditation.yaml'
saveloc         = '/Users/a_fin/Desktop/Year 4/Project/Data/'
os.makedirs(saveloc, exist_ok=True)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
feelings       = config['feelings']
n_pca_dims     = config['no_dimensions_PCA']
n_jumps        = config.get('no_of_jumps', 1)

# --- Load Data using csv_splitter (as in HMM.py) ---
csv_splitter_instance = csv_splitter(meditation_csv)
df_csv_file_original = csv_splitter_instance.read_CSV()
if df_csv_file_original is None:
    raise ValueError("CSV file could not be read. Check the file path and try again.")

# --- PCA on ALL Data (as in HMM.py) ---
pc_finder = principal_component_finder(df_csv_file_original, feelings, n_pca_dims, saveloc)
principal_components, _, _ = pc_finder.PCA_TOT()

# --- True-Label Encoding (only OM & LK get labels; others NaN) ---
true_map = {'Open Monitoring': 3, 'Loving Kindness': 4}
df_csv_file_original['true_label'] = df_csv_file_original['Med_type'].map(true_map)

transition_contributions = 5.0

# --- Parameter grids ---
gamma_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
nu_values = np.arange(2, 31)

def hmm_grid_search(gamma, nu):
    print(f"Testing gamma={gamma}, nu={nu}")
    hmm = CustomHMMClustering(
        meditation_csv, saveloc, df_csv_file_original, feelings,
        principal_components, n_jumps, transition_contributions
    )
    _, _, _, labels_array = hmm.run(
        num_base_states=4,
        num_iterations=30,
        num_repetitions=1,
        gamma_threshold=gamma,
        min_nu=nu
    )
    pred = labels_array['transition_label']
    # Calculate accuracy for LK (label=4) and OM (label=3)
    lk_mask = df_csv_file_original['true_label'] == 4
    om_mask = df_csv_file_original['true_label'] == 3

    # Avoid division by zero
    n_lk = lk_mask.sum()
    n_om = om_mask.sum()

    acc_lk = np.nan
    acc_om = np.nan

    if n_lk > 0:
        acc_lk = (pred[lk_mask] == '4').sum() / n_lk
    if n_om > 0:
        acc_om = (pred[om_mask] == '3').sum() / n_om

    # Combined accuracy (average of both, ignoring NaN)
    accs = [a for a in [acc_lk, acc_om] if not np.isnan(a)]
    acc_combined = np.mean(accs) if accs else np.nan

    return acc_lk, acc_om, acc_combined

param_grid = [(gamma, nu) for gamma in gamma_thresholds for nu in nu_values]
results = Parallel(n_jobs=-1, verbose=10)(delayed(hmm_grid_search)(gamma, nu) for gamma, nu in param_grid)

acc_matrix_lk = np.zeros((len(gamma_thresholds), len(nu_values)))
acc_matrix_om = np.zeros((len(gamma_thresholds), len(nu_values)))
acc_matrix_combined = np.zeros((len(gamma_thresholds), len(nu_values)))

for idx, (acc_lk, acc_om, acc_combined) in enumerate(results):
    i = idx // len(nu_values)
    j = idx % len(nu_values)
    gamma = gamma_thresholds[i]
    nu = nu_values[j]
    acc_matrix_lk[i, j] = acc_lk
    acc_matrix_om[i, j] = acc_om
    acc_matrix_combined[i, j] = acc_combined
    print(f"gamma={gamma}, nu={nu} | LK acc={acc_lk:.3f}, OM acc={acc_om:.3f}, Combined acc={acc_combined:.3f}")

# --- Plotting ---
import seaborn as sns
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(acc_matrix_lk, xticklabels=np.round(nu_values,2), yticklabels=np.round(gamma_thresholds,2), cmap='Blues', annot=False)
plt.xlabel('Nu Value')
plt.ylabel('Gamma Threshold')
plt.title('LK Accuracy (label=4)')

plt.subplot(1, 3, 2)
sns.heatmap(acc_matrix_om, xticklabels=np.round(nu_values,2), yticklabels=np.round(gamma_thresholds,2), cmap='Greens', annot=False)
plt.xlabel('Nu Value')
plt.ylabel('Gamma Threshold')
plt.title('OM Accuracy (label=3)')

plt.subplot(1, 3, 3)
sns.heatmap(acc_matrix_combined, xticklabels=np.round(nu_values,2), yticklabels=np.round(gamma_thresholds,2), cmap='Oranges', annot=False)
plt.xlabel('Nu Value')
plt.ylabel('Gamma Threshold')
plt.title('Combined Accuracy (OM & LK)')
# Pinpoint max
max_idx = np.unravel_index(np.nanargmax(acc_matrix_combined), acc_matrix_combined.shape)
plt.scatter(max_idx[1]+0.5, max_idx[0]+0.5, s=120, c='red', marker='*', label='Max')
plt.legend(loc='upper right')

plt.tight_layout()
outpath = os.path.join(saveloc, 'meditation_hmm_gridsearch_accuracy_heatmaps.png')
plt.savefig(outpath, dpi=200)
plt.show()

print(f"Saved grid search heatmaps to: {outpath}")
