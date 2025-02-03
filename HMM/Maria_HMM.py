import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
from Maria_HMM_methods import DataLoader, HMMTrainer, Visualizer
import yaml
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary
# =============================================
# Main Workflow
# =============================================
if __name__ == "__main__":
    # Load data
    main_folder = 'path/to/main/folder'
    participant_folders = ['participant1', 'participant2', 'participant3']
    group_label = 'GroupA'
    data_loader = DataLoader(main_folder, participant_folders, group_label)
    df = data_loader.load_tet_data()

    # Extract feelings (same as PCA/K-Means)
    feelings = ['feeling1', 'feeling2', 'feeling3', 'feeling4', 'feeling5', 'feeling6',
                'feeling7', 'feeling8', 'feeling9', 'feeling10', 'feeling11', 'feeling12']
    data = df[feelings].values

    # Handle missing values (if any)
    data = np.nan_to_num(data)  # Replace NaNs with zeros

    # Train HMM
    hmm_trainer = HMMTrainer(num_states=3)
    hmm_model = hmm_trainer.train_hmm(data)
    state_seq, log_prob = hmm_trainer.decode_hmm(hmm_model, data)

    # Visualize HMM states and transitions
    visualizer = Visualizer()
    visualizer.visualize_clusters_and_transitions(data, state_seq)