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
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

from HMM_methods import csv_splitter, principal_component_finder, CustomHMMClustering, Visualiser, HMMJumpAnalysis

# Read the CSV file using the configuration from YAML.
csv_splitter_instance = csv_splitter(config['filelocation_TET'])
df_csv_file_original = csv_splitter_instance.read_CSV()
if df_csv_file_original is None:
    raise ValueError("CSV file could not be read. Check the file path and try again.")

# Split by a header specified in the configuration.
split_df, split_csv_array = csv_splitter_instance.split_by_header(df_csv_file_original, config['header'])

# Extract principal components using the specified feelings and number of dimensions.
feelings = config['feelings']
principal_component_finder_instance = principal_component_finder(df_csv_file_original, feelings,
                                                                    config['no_dimensions_PCA'],
                                                                    config['savelocation_TET'])

principal_components, explained_variance_ratio, df_TET_feelings_prin = principal_component_finder_instance.PCA_TOT()
df_TET_feelings_prin_dict = principal_component_finder_instance.PCA_split(split_csv_array)

# Instantiate and run the custom clustering.
# Note: Here we use the original CSV file (df_csv_file_original) for clustering.
# You might wish to change this if you have a different dataset.
no_of_jumps = config['no_of_jumps']
smoothness = CustomHMMClustering.calculate_smoothness(df_csv_file_original, feelings)
print(f"Data smoothness: {smoothness}")

# Try to load optimal parameters from realistic validation results
realistic_params_path = config.get('hyperparameters', {}).get('filelocation_smoothness', None)
gamma_col = config.get('hyperparameters', {}).get('gamma_col', 'gamma_threshold')
nu_col = config.get('hyperparameters', {}).get('nu_col', 'min_nu')
tc_col = config.get('hyperparameters', {}).get('tc_col', 'transition_contribution')
smoothness_col = config.get('hyperparameters', {}).get('smoothness_col', 'measured_smoothness')

if realistic_params_path and os.path.exists(realistic_params_path):
    print("Using realistic smoothness validation results...")
    optimal_params = CustomHMMClustering.get_optimal_params_for_smoothness(
        smoothness, realistic_params_path,
    )
else:
    print("Realistic validation results not found. Using default parameters...")
    print("Consider running realistic_smoothness_validation.py first for optimal results.")
    optimal_params = {
        'gamma_threshold': 0.8,
        'min_nu': 29,
        'transition_contribution': 5,
        'selection_method': 'default'
    }

gamma_threshold = optimal_params['gamma_threshold']
min_nu = optimal_params['min_nu'] 
transition_contributions = optimal_params['transition_contribution']

print(f"Using parameters: gamma={gamma_threshold}, min_nu={min_nu}, tc={transition_contributions}")
print(f"Selection method: {optimal_params.get('selection_method', 'unknown')}")



clustering = CustomHMMClustering(config['filelocation_TET'], config['savelocation_TET'],
                                    df_csv_file_original, feelings, principal_components, no_of_jumps, transition_contributions)

results_array, dictionary_clust_labels, transitions, notransitions = clustering.run(num_base_states=config['no_clust'], num_iterations=30, num_repetitions=1, gamma_threshold = gamma_threshold, min_nu = min_nu)
results_array.to_csv('/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted.csv', index=False)
notransitions.to_csv('/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv', index=False)

with open('/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_transitions.pkl', 'wb') as f:
    pickle.dump(transitions, f)


# Instantiate and run the visualiser.
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
    time_jump=config['time_jump'],  
)
visualiser_instance.run()
