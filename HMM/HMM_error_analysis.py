import yaml
import pickle
import os
import pandas as pd
import numpy as np
from HMM_methods import csv_splitter, principal_component_finder, CustomHMMClustering

# Load configuration and data
with open("Breathwork.yaml") as f:
    config = yaml.safe_load(f)

csv_splitter_instance = csv_splitter(config['filelocation_TET'])
df_original = csv_splitter_instance.read_CSV()
_, split_array = csv_splitter_instance.split_by_header(df_original, config['header'])

# Compute PCA once
pc_finder = principal_component_finder(
    df_original,
    config['feelings'],
    config['no_dimensions_PCA'],
    config['savelocation_TET']
)
principal_components, _, _ = pc_finder.PCA_TOT()

# Parameter grids for sensitivity analysis
seeds = [0, 1, 2, 3, 4]
curr_weights = [0.0, 0.1, 0.2, 0.5]
gamma_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]

results = []

for seed in seeds:
    for cw in curr_weights:
        # Set up clustering with this curriculum weight
        clustering = CustomHMMClustering(
            filelocation_TET=config['filelocation_TET'],
            savelocation_TET=config['savelocation_TET'],
            df_csv_file_original=df_original,
            feelings=config['feelings'],
            principal_components=principal_components,
            no_of_jumps=config['no_of_jumps'],
            transition_contributions=cw
        )

        # Preprocess and cluster with seed
        np.random.seed(seed)
        clustering.preprocess_data()

        # Patch: pass base_seed into perform_clustering if supported
        if 'base_seed' in clustering.perform_clustering.__code__.co_varnames:
            clustering.perform_clustering(
                num_base_states=2,
                num_iterations=30,
                num_repetitions=30,
                base_seed=seed
            )
        else:
            clustering.perform_clustering(
                num_base_states=2,
                num_iterations=30,
                num_repetitions=30
            )

        # Copy pre-threshold labels
        labels = clustering.array['labels'].copy()

        # Try to grab average log-likelihood (if stored)
        if hasattr(clustering, 'avg_log_lik'):
            avg_loglik = clustering.avg_log_lik
        else:
            avg_loglik = np.nan  # fallback if not available

        for gt in gamma_thresholds:
            # Reset labels
            clustering.array['labels'] = labels.copy()
            # Apply new gamma threshold
            clustering.post_process_cluster_three(cluster_three_label=2, gamma_threshold=gt)
            # Calculate bridge rate
            bridge_rate = np.mean(clustering.array['labels'] == 2)
            results.append({
                'seed': seed,
                'curr_weight': cw,
                'gamma_thr': gt,
                'bridge_rate': bridge_rate,
                'avg_loglik': avg_loglik
            })
        print(f"Completed seed {seed}, curr_weight {cw}")

# Save results
df_res = pd.DataFrame(results)
output_path = "/Users/a_fin/Desktop/Year 4/Project/Data/TSHMM_sensitivity_analysis.csv"
df_res.to_csv(output_path, index=False)
print(f"Saved sensitivity results to {output_path}")
