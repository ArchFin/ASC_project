import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import csv
import json
from HMM.HMM_methods_copy import csv_splitter, principal_component_finder, CustomHMMClustering, HMMModel, Visualiser
from Vectorised_Kmeans.Max_kmeans_methods import KMeansVectorClustering, KMeansVectorVisualizer
import os  # Added for path handling
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.patches as mpatches
from statsmodels.tsa.stattools import acf
import yaml
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Load YAML file
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

csv_splitter_instance = csv_splitter(config['filelocation_TET'])
df_csv_file_original = csv_splitter_instance.read_CSV()

# -----------------------------------------------------------------
# Run KMeans Pipeline
# -----------------------------------------------------------------
principal_component_finder_instance = principal_component_finder(
    df_csv_file_original, config['feelings'], 
    config['no_dimensions_PCA'], config['savelocation_TET']
)
principal_components, _, _ = principal_component_finder_instance.PCA_TOT()

def plot_stacked_trajectories(kmeans_obj, HMM_obj):
    time_jump = 28  # Default time interval between samples

    common_headings = (
        set(kmeans_obj.traj_transitions_dict_original.keys())
        .intersection(HMM_obj.traj_transitions_dict_original.keys())
    )

    for heading in common_headings:
        original_data_kmeans = kmeans_obj.traj_transitions_dict_original[heading]
        original_data_visualiser = HMM_obj.traj_transitions_dict_original[heading]
        
        n_points = min(len(original_data_kmeans), len(original_data_visualiser))
        time_array = np.arange(0, time_jump * n_points, time_jump)
        
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # KMeans Top Subplot
        # Plot feeling trajectories
        for feeling in kmeans_obj.feelings:
            ax_top.plot(time_array, original_data_kmeans[feeling][:n_points] * 10, label=feeling, color= kmeans_obj.feeling_colors[feeling])

        # Add cluster shading if available
        if heading in kmeans_obj.traj_transitions_dict:
            traj_group = kmeans_obj.traj_transitions_dict[heading]
            # Use the original scaling for cluster boundaries:
            # Each index in traj_group corresponds to time = index * (time_jump * self.no_of_jumps)
            prev_color_val = traj_group['clust'].iloc[0]
            start_index = 0
            for index, color_val in enumerate(traj_group['clust']):
                # Check for a change in cluster or if we are at the last point
                if color_val != prev_color_val or index == traj_group.shape[0] - 1:
                    # Compute the end time for the shaded region.
                    if index != traj_group.shape[0] - 1:
                        end_time = index * (time_jump * kmeans_obj.no_of_jumps)
                    else:
                        end_time = time_array[-1]
                    start_time = start_index * (time_jump * kmeans_obj.no_of_jumps)
                    ax_top.axvspan(start_time, end_time, 
                            facecolor=kmeans_obj.color_map.get(prev_color_val, 'grey'), alpha=0.3)
                    start_index = index
                    prev_color_val = color_val

        # Finalize plot appearance
        combined = ''.join(map(str, heading)).translate({ord(c): None for c in "\\'() "})
        ax_top.set_title(combined)
        ax_top.set_xlabel('Time (s)')
        ax_top.set_ylabel('Rating')

        # HMM Bottom Subplot
        # Plot feeling trajectories
        for feeling in HMM_obj.feelings:
            ax_bottom.plot(time_array, original_data_visualiser[feeling]*10, 
                    label=feeling, color=HMM_obj.feeling_colors[feeling])

        # Add cluster shading
        if heading in HMM_obj.traj_transitions_dict:
            traj_group = HMM_obj.traj_transitions_dict[heading]
            prev_color_val = traj_group['labels'].iloc[0]
            start_index = 0
            for index, color_val in enumerate(traj_group['labels']):
                if color_val != prev_color_val or index == traj_group.shape[0]-1:
                    end_idx = index if index != traj_group.shape[0]-1 else len(time_array)-1
                    ax_bottom.axvspan(time_array[start_index], time_array[end_idx],
                                facecolor=HMM_obj.color_map.get(prev_color_val, 'grey'), alpha=0.3)
                    start_index = index
                    prev_color_val = color_val

        # Add annotations
        if heading in HMM_obj.group_transitions:
            HMM_obj.annotate_state_durations(ax_bottom, time_array, 
                                            HMM_obj.group_transitions[heading])
        if 'Condition' in original_data_visualiser.columns:
            HMM_obj.annotate_conditions(ax_bottom, time_array, original_data_visualiser)

        # Finalize plot
        combined = ''.join(map(str, heading)).translate(
            {ord(c): None for c in "\\'() "})
        ax_bottom.set_title(combined)
        ax_bottom.set_xlabel('Time (s)')
        ax_bottom.set_ylabel('Rating')

        # Legend handling
        handles_top, labels_top = ax_top.get_legend_handles_labels()
        handles_bot, labels_bot = ax_bottom.get_legend_handles_labels()
        unique_labels = dict(zip(labels_top + labels_bot, handles_top + handles_bot))
        ax_top.legend(unique_labels.values(), unique_labels.keys(),
                         bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Legend')

        # Save figure
        combined_title = ''.join(map(str, heading)).translate({ord(c): None for c in "\\'() "})
        fig.tight_layout()
        save_path = os.path.join(kmeans_obj.savelocation_TET, f"Stacked_{combined_title}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

differences_path = "/Users/a_fin/Desktop/Year 4/Project/Data/VKM_output_differences_array.csv"
differences_array = pd.read_csv(differences_path)
dictionary_clust_labels = {np.int64(1): '1a', np.int64(3): 3, np.int64(2): 2, np.int64(4): '1b'}

# KMeans Visualizer
kmeans_visualizer = KMeansVectorVisualizer(
    config['filelocation_TET'], config['savelocation_TET'], 
    differences_array, df_csv_file_original, 
    dictionary_clust_labels, principal_components, 
    config['feelings'], 7, 
    config['colours']
)
kmeans_visualizer.run()  # Generates KMeans trajectory plots


results_path = "/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted.csv"
results_array = pd.read_csv(results_path)
hmm_clust_labels = {np.int64(0): 'Cluster 1', np.int64(1): 'Cluster 2', np.int64(2): 'Cluster 3'}

# Load from a file
with open("/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_transitions.pkl", "rb") as f:
    transitions = pickle.load(f)

print("Loaded Dictionary:", transitions)

# HMM Visualizer
hmm_visualizer = Visualiser(
    config['filelocation_TET'], config['savelocation_TET'], 
    results_array, df_csv_file_original, 
    hmm_clust_labels, principal_components, 
    config['feelings'], 1, 
    config['colours'], transitions
)
hmm_visualizer.run()


plot_stacked_trajectories(kmeans_visualizer, hmm_visualizer)