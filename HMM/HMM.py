from sklearn.cluster import KMeans
from itertools import combinations
import pandas as pd
from scipy.spatial import distance
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.patches as mpatches
#from statsmodels.tsa.stattools import acf
import yaml
import csv
import json
import pickle


# Load YAML file
print("[MAIN] ========== LOADING CONFIGURATION ==========")
with open("Simulation.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary
print(f"[MAIN] Configuration loaded successfully from Simulation.yaml")
print(f"[MAIN] Input file: {config['filelocation_TET']}")
print(f"[MAIN] Output location: {config['savelocation_TET']}")
print(f"[MAIN] Number of PCA dimensions: {config['no_dimensions_PCA']}")

from HMM_methods import csv_splitter, principal_component_finder, PipelineManager, Visualiser, HMMModel

print("[MAIN] ========== READING AND PREPROCESSING DATA ==========")
# Read the CSV file using the configuration from YAML.
csv_splitter_instance = csv_splitter(config['filelocation_TET'])
df_csv_file_original = csv_splitter_instance.read_CSV()
if df_csv_file_original is None:
    raise ValueError("CSV file could not be read. Check the file path and try again.")
print(f"[MAIN] Data loaded successfully. Shape: {df_csv_file_original.shape}")

# Split by a header specified in the configuration.
split_df, split_csv_array = csv_splitter_instance.split_by_header(df_csv_file_original, config['header'])
print(f"[MAIN] Data split by '{config['header']}' column")
print(f"[MAIN] Number of unique groups: {len(split_df) if split_df else 0}")

# Extract principal components using the specified feelings and number of dimensions.
print("[MAIN] ========== COMPUTING PRINCIPAL COMPONENTS ==========")
feelings = config['feelings']
no_of_jumps = config.get('no_of_jumps', 1)  # Get from config or default to 1
print(f"[MAIN] Extracted feelings: {feelings}")
print(f"[MAIN] Number of jumps: {no_of_jumps}")

principal_component_finder_instance = principal_component_finder(df_csv_file_original, feelings,
                                                                    config['no_dimensions_PCA'],
                                                                    config['savelocation_TET'])

principal_components, explained_variance_ratio, df_TET_feelings_prin = principal_component_finder_instance.PCA_TOT()
df_TET_feelings_prin_dict = principal_component_finder_instance.PCA_split(split_csv_array)
print(f"[MAIN] PCA complete. Explained variance ratios: {explained_variance_ratio.round(3)}")

# Instantiate and run the pipeline manager for clustering
print("[MAIN] ========== INITIALIZING PIPELINE MANAGER ==========")
pm = PipelineManager(
    config_path="Simulation.yaml",  # Use same config as loaded above
    feelings=feelings,
    principal_components=principal_components,
    no_of_jumps=no_of_jumps
)
print(f"[MAIN] PipelineManager initialized with smoothness threshold: {pm.SMOOTHNESS_THRESHOLD}")

print("[MAIN] ========== STARTING MODEL SELECTION AND CLUSTERING ==========")
results_array, dictionary_clust_labels, transitions, notransitions = pm.run_pipeline(
    df_csv_file_original,
    num_base_states=2,
    num_iterations=15,
    num_repetitions=2
)
print(f"[MAIN] Pipeline execution completed. Results shape: {results_array.shape}")
print(f"[MAIN] Dictionary cluster labels: {dictionary_clust_labels}")

print("[MAIN] ========== SAVING RESULTS ==========")
results_array.to_csv("/Users/a_fin/Desktop/Year 4/Project/Summer_Data/HMM_output_adjusted.csv", index=False)
notransitions.to_csv("/Users/a_fin/Desktop/Year 4/Project/Summer_Data/HMM_output_adjusted_notransitions.csv", index=False)
print("[MAIN] Results saved to CSV files")

with open("/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_transitions.pkl", "wb") as f:
    pickle.dump(transitions, f)
print("[MAIN] Transitions saved to pickle file")


# Instantiate and run the visualiser.
print("[MAIN] ========== STARTING VISUALIZATION ==========")
visualiser_instance = Visualiser(
    filelocation_TET=config['filelocation_TET'],
    savelocation_TET=config['savelocation_TET'],
    array=results_array,
    df_csv_file_original=df_csv_file_original,
    dictionary_clust_labels=dictionary_clust_labels,
    principal_components=principal_components,
    feelings=feelings,
    no_of_jumps=no_of_jumps,
    colours = config['colours'],
    transitions = transitions,
)
print("[MAIN] Running visualizer...")
visualiser_instance.run()
print("[MAIN] ✓ Visualization completed")

# === VALIDATION: Compare learned vs true transition matrix ===
print("[MAIN] ========== STARTING VALIDATION PHASE ==========")
# Load the true transition matrix from your simulation
true_transition_matrix = [
    [0.90, 0.00, 0.10],  # state 0: mostly stays, can go to metastable
    [0.00, 0.90, 0.10],  # state 1: mostly stays, can go to metastable
    [0.15, 0.15, 0.70]   # metastable: likely to go to 0 or 1, rarely stays
]

# Load the true state sequence from the simulated data
true_states = df_csv_file_original['Cluster'].values
print(f"[MAIN] True states loaded. Length: {len(true_states)}")

print("\n" + "="*60)
print("VALIDATION: HMM Transition Matrix Recovery")
print("="*60)

# Print learned transition matrix before validation - use the model from PipelineManager
clustering = pm.model  # Get the actual model instance from PipelineManager
print(f"[MAIN] Selected model type: {type(clustering).__name__}")
if hasattr(clustering, 'avg_trans_prob'):
    print("Raw learned transition matrix:")
    print(np.round(clustering.avg_trans_prob, 3))
    print(f"Shape: {clustering.avg_trans_prob.shape}")
    
    # Check if we need to validate only the base states (first 2x2 submatrix)
    if clustering.avg_trans_prob.shape[0] == 3:
        print("\nUsing only base states (2x2 submatrix) for validation:")
        learned_base = clustering.avg_trans_prob[:2, :2]
        # Renormalize the 2x2 submatrix
        learned_base = learned_base / learned_base.sum(axis=1, keepdims=True)
        print("Learned base transition matrix (renormalized):")
        print(np.round(learned_base, 3))
        
        true_base = np.array(true_transition_matrix)[:2, :2]
        true_base = true_base / true_base.sum(axis=1, keepdims=True)
        print("True base transition matrix (renormalized):")
        print(np.round(true_base, 3))
        
        validation = clustering.validate_transition_matrix(
            learned_base, 
            true_base, 
            align_states=True
        )
    else:
        validation = clustering.validate_transition_matrix(
            clustering.avg_trans_prob, 
            np.array(true_transition_matrix), 
            align_states=True
        )
    
    print(f"Frobenius norm error: {validation['frob_norm']:.4f}")
    print(f"Maximum element-wise error: {validation['max_err']:.4f}")
    print(f"Mean element-wise error: {validation['mean_err']:.4f}")
    print(f"Mean KL divergence: {validation['mean_kl_divergence']:.4f}")
    
    print("\nLearned transition matrix (aligned):")
    print(np.round(validation['learned_aligned'], 3))
    print("\nTrue transition matrix (aligned):")
    print(np.round(validation['true_aligned'], 3))
    print("\nAbsolute error matrix:")
    print(np.round(validation['abs_err'], 3))
    
    # Save validation plots
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot learned matrix
    sns.heatmap(validation['learned_aligned'], annot=True, fmt='.3f', 
                cmap='Blues', ax=axes[0], cbar=True)
    axes[0].set_title('Learned Transition Matrix')
    axes[0].set_xlabel('To State')
    axes[0].set_ylabel('From State')
    
    # Plot true matrix
    sns.heatmap(validation['true_aligned'], annot=True, fmt='.3f', 
                cmap='Blues', ax=axes[1], cbar=True)
    axes[1].set_title('True Transition Matrix')
    axes[1].set_xlabel('To State')
    axes[1].set_ylabel('From State')
    
    # Plot error matrix
    sns.heatmap(validation['abs_err'], annot=True, fmt='.3f', 
                cmap='Reds', ax=axes[2], cbar=True)
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('To State')
    axes[2].set_ylabel('From State')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['savelocation_TET'], 'transition_matrix_validation.png'))
    plt.close()
    
    print(f"\nValidation plot saved to: {config['savelocation_TET']}transition_matrix_validation.png")

else:
    print("WARNING: avg_trans_prob not found. Check perform_clustering method.")

# Validate state sequence
if hasattr(clustering, 'avg_state_seq'):
    seq_validation = clustering.validate_state_sequence(clustering.avg_state_seq, true_states)
    print(f"\nState sequence accuracy: {seq_validation['accuracy']:.4f}")
    print(f"Normalized mutual information: {seq_validation['nmi']:.4f}")
    print(f"State mapping used: {seq_validation['state_mapping']}")
else:
    print("WARNING: avg_state_seq not found. Check perform_clustering method.")

print("="*60)
print("[MAIN] ========== PIPELINE EXECUTION COMPLETED ==========")
print(f"[MAIN] ✓ All operations completed successfully")
print(f"[MAIN] ✓ Results saved to: {config['savelocation_TET']}")
print(f"[MAIN] ✓ Model used: {type(clustering).__name__}")
print("[MAIN] ==========================================================")