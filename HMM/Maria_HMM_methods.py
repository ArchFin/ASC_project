import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from hmmlearn import hmm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =============================================
# DataLoader Class
# =============================================
class DataLoader:
    """
    A class to load TET data from participant folders.
    """

    def __init__(self, main_folder, participant_folders, group_label):
        self.main_folder = main_folder
        self.participant_folders = participant_folders
        self.group_label = group_label

    def load_tet_data(self):
        """
        Load TET data from participant folders.
        """
        all_tet_data = []
        participant_ids = []
        groups = []
        meditation_styles = []
        session_ids = []

        for participant_folder in self.participant_folders:
            participant_path = os.path.join(self.main_folder, participant_folder, '20-SubjExp')
            for file_name in os.listdir(participant_path):
                if file_name.endswith('_TET.mat'):
                    file_path = os.path.join(participant_path, file_name)
                    data = loadmat(file_path)
                    if 'Subjective' in data:
                        tet_data = data['Subjective'][:, :12]  # Assuming 12 feelings
                        if len(all_tet_data) == 0:
                            all_tet_data = tet_data
                        else:
                            all_tet_data = np.vstack((all_tet_data, tet_data))
                        participant_ids.extend([f"{participant_folder}_{self.group_label}"] * tet_data.shape[0])
                        groups.extend([self.group_label] * tet_data.shape[0])
                        meditation_styles.extend(data['Subjective'][:, 12])
                        session_ids.extend([file_name] * tet_data.shape[0])

        # Convert to DataFrame for consistency with PCA/K-Means
        feelings = ['feeling1', 'feeling2', 'feeling3', 'feeling4', 'feeling5', 'feeling6',
                    'feeling7', 'feeling8', 'feeling9', 'feeling10', 'feeling11', 'feeling12']
        df = pd.DataFrame(all_tet_data, columns=feelings)
        df['Participant'] = participant_ids
        df['Group'] = groups
        df['MeditationStyle'] = meditation_styles
        df['Session'] = session_ids

        return df


# =============================================
# HMMTrainer Class
# =============================================
class HMMTrainer:
    """
    A class to train and decode Hidden Markov Models (HMMs).
    """

    def __init__(self, num_states, max_iter=100, tol=1e-4):
        self.num_states = num_states
        self.max_iter = max_iter
        self.tol = tol

    def train_hmm(self, data):
        """
        Train a Hidden Markov Model (HMM) using the Baum-Welch algorithm.
        """
        model = hmm.GaussianHMM(n_components=self.num_states, covariance_type="full", n_iter=self.max_iter, tol=self.tol)
        model.fit(data)
        return model

    def decode_hmm(self, model, data):
        """
        Decode the most likely state sequence using the Viterbi algorithm.
        """
        state_seq = model.predict(data)
        log_prob = model.score(data)
        return state_seq, log_prob


# =============================================
# Visualizer Class
# =============================================
class Visualizer:
    """
    A class to visualize clusters, transitions, and sessions.
    """

    @staticmethod
    def visualize_clusters_and_transitions(data, state_seq, labels=None):
        """
        Visualize clusters and transitions using PCA.
        """
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(state_seq))))
        for i, state in enumerate(np.unique(state_seq)):
            plt.scatter(pca_data[state_seq == state, 0], pca_data[state_seq == state, 1], color=colors[i], label=f'State {state}')

        if labels:
            for i in range(1, len(labels)):
                if labels[i] and 'Transition' in labels[i]:
                    plt.plot(pca_data[i-1:i+1, 0], pca_data[i-1:i+1, 1], 'k--', linewidth=1.5)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('HMM States and Transitions (PCA)')
        plt.legend()
        plt.show()


