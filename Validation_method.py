import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import yaml
from HMM.HMM_methods import csv_splitter, principal_component_finder, CustomHMMClustering

from Vectorised_Kmeans.Max_kmeans_methods import csv_splitter, principal_component_finder, KMeansVectorClustering, KMeansVectorVisualizer, JumpAnalysis

# Load configuration
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load and preprocess data
csv_splitter_instance = csv_splitter(config['filelocation_TET'])
df_csv_file_original = csv_splitter_instance.read_CSV()

# Run K-means once with no_of_jumps=7
kmeans_clustering = KMeansVectorClustering(
    filelocation_TET=config['filelocation_TET'],
    savelocation_TET=config['savelocation_TET'],
    df_csv_file_original=df_csv_file_original,
    feelings=config['feelings'],
    feelings_diffs=config['feelings_diffs'],
    principal_components=principal_component_finder(
        df_csv_file_original, config['feelings'], config['no_dimensions_PCA'], config['savelocation_TET']
    ).PCA_TOT()[0],
    no_of_jumps=7,  # K-means uses jump=7
    colours=config['colours'],
    colours_list=config['colours_list']
)
kmeans_clustering.run()
df_kmeans_expanded = kmeans_clustering.expand_to_original_shape()
kmeans_labels = df_kmeans_expanded['cluster_label'].dropna().astype(int)

# Define HMM repetitions to test
hmm_repetitions = np.arange(0.1, 1, 0.1)
mi_scores = []
abrupt_counts = []

for rep in hmm_repetitions:
    print(f"Running HMM with {rep} repetitions...")
    # Run HMM with current repetitions
    hmm_clustering = CustomHMMClustering(
        filelocation_TET=config['filelocation_TET'],
        savelocation_TET=config['savelocation_TET'],
        df_csv_file_original=df_csv_file_original,
        feelings=config['feelings'],
        principal_components=principal_component_finder(
            df_csv_file_original, config['feelings'], config['no_dimensions_PCA'], config['savelocation_TET']
        ).PCA_TOT()[0],
        no_of_jumps=1,  # HMM uses jump=1
        transition_contributions = rep
    )
    results_array, _, group_transitions, __ = hmm_clustering.run(
        num_base_states=2,  # Number of states for HMM
        num_iterations=30,
        num_repetitions=25
    )
    # Extract HMM labels
    hmm_labels = results_array['labels'].astype(int)
    
    # Compute Mutual Information
    mi = mutual_info_score(kmeans_labels, hmm_labels)
    mi_scores.append(mi)
    
    # Count abrupt transitions
    abrupt_count = 0
    for transitions in group_transitions.values():
        for trans in transitions:
            if trans[-1] == 'abrupt':  # transition_type is the last element
                abrupt_count += 1
    abrupt_counts.append(abrupt_count)

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(hmm_repetitions, mi_scores, 'bo-')
plt.xlabel('Number of HMM Transition scaling')
plt.ylabel('Mutual Information')
plt.title('Mutual Information vs HMM Transition scaling')

plt.subplot(1, 2, 2)
plt.plot(hmm_repetitions, abrupt_counts, 'ro-')
plt.xlabel('Number of HMM Transition scaling')
plt.ylabel('Number of Abrupt Transitions')
plt.title('Abrupt Transitions vs HMM Transition scaling')

plt.tight_layout()
plt.savefig(config['savelocation_TET'] + 'mutual_info_abrupt_transitions_transition_scaling.png')
plt.close()