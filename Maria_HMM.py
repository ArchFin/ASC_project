import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t

# Load data functions
def load_TET_data(main_folder, participant_folders, group_label):
    all_TET_data = []
    participant_ids, groups, meditation_styles, session_ids = [], [], [], []
    
    for participant in participant_folders:
        participant_path = Path(main_folder) / participant / '20-SubjExp'
        if not participant_path.exists():
            continue
            
        for file in participant_path.iterdir():
            if file.suffix == '.mat' and '_TET' in file.stem:
                mat = loadmat(file)
                if 'Subjective' not in mat or mat['Subjective'].shape[1] < 13:
                    continue
                
                data = mat['Subjective'][:, :12]  # First 12 columns
                meditation_style = mat['Subjective'][:, 12].flatten()

                n_rows = data.shape[0]
                all_TET_data.append(data)
                participant_ids.extend([f"{participant}_{group_label}"] * n_rows)
                groups.extend([group_label] * n_rows)
                meditation_styles.extend(meditation_style)
                session_ids.extend([file.name] * n_rows)
    
    return (np.vstack(all_TET_data), 
            np.array(participant_ids), 
            np.array(groups), 
            np.array(meditation_styles), 
            np.array(session_ids))

# Main processing
main_folder1 = '/Users/mariakarampela/Downloads/DreemEEG'
main_folder2 = '/Users/mariakarampela/Downloads/EEG-Group1'

participant_folders1 = ['1425', '1733_BandjarmasinKomodoDragon', '1871', 
                        '1991_MendozaCow', '2222_JiutaiChicken', '2743_HuaianKoi']

participant_folders2 = ['184_WestYorkshireWalrus', '1465_WashingtonQuelea', '1867_BucharestTrout',
                        '1867_GoianiaCrane', '3604_LichuanHookworm', '3604_ShangquiHare',
                        '3614_BrisbaneHornet', '3614_VientianeWhippet', '3938_YingchengSeaLion',
                        '4765_NouakchottMoose', '5644_AkesuCoral', '5892_LvivRooster',
                        '5892_NonthaburiHalibut', '7135_TampicoWallaby', '8681_NanchangAlbatross',
                        '8725_SishouMosquito', '8725_YangchunCobra']

# Load and process data
data1, ids1, grp1, med1, sess1 = load_TET_data(main_folder1, participant_folders1, 'Group1')
data2, ids2, grp2, med2, sess2 = load_TET_data(main_folder2, participant_folders2, 'Group2')

if data1.shape[1] != data2.shape[1]:
    raise ValueError("Column mismatch between groups")

all_data = np.vstack((data1, data2))
participant_ids = np.concatenate((ids1, ids2))
groups = np.concatenate((grp1, grp2))
meditation = np.concatenate((med1, med2))
sessions = np.concatenate((sess1, sess2))

# Normalisation
scaler = StandardScaler()
data_normalized = scaler.fit_transform(all_data)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=12345, n_init='auto')
clusters = kmeans.fit_predict(data_normalized)

# HMM implementation
class HMM:
    def __init__(self, n_states, max_iter=100):
        self.n_states = n_states
        self.max_iter = max_iter
    
    def forward_backward(self, data):
        n_samples = data.shape[0]
        alpha = np.zeros((n_samples, self.n_states))
        alpha[0] = 1.0 / self.n_states
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                likelihood = multivariate_t.pdf(data[t], self.means_[j], self.covs_[j], self.df_[j])
                alpha[t, j] = likelihood * np.sum(alpha[t - 1] * self.trans_mat_[:, j])
            alpha[t] /= np.sum(alpha[t])
        
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1
        
        for t in range(n_samples - 2, -1, -1):
            for j in range(self.n_states):
                beta[t, j] = np.sum(beta[t + 1] * self.trans_mat_[j] * 
                                    [multivariate_t.pdf(data[t + 1], self.means_[k], self.covs_[k], self.df_[k])
                                     for k in range(self.n_states)])
            beta[t] /= np.sum(beta[t])
        
        return alpha, beta
    
    def update_parameters(self, data, alpha, beta):
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        for j in range(self.n_states):
            weighted_sum = np.sum(gamma[:, j][:, np.newaxis] * data, axis=0)
            self.means_[j] = weighted_sum / gamma[:, j].sum()
            self.covs_[j] = np.cov(data.T, aweights=gamma[:, j]) + np.eye(data.shape[1]) * 1e-6
    
    def fit(self, data):
        for _ in range(self.max_iter):
            alpha, beta = self.forward_backward(data)
            self.update_parameters(data, alpha, beta)
        return self
    
    def predict(self, data):
        return np.argmax(self.forward_backward(data)[0], axis=1)

# Train and predict with HMM
hmm = HMM(n_states=4, max_iter=50)
hmm.fit(data_normalized)
hmm_states = hmm.predict(data_normalized)

# PCA Visualisation
def visualize_clusters_pca(data, labels, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.show()

visualize_clusters_pca(data_normalized, clusters, 'K-Means Clusters')
visualize_clusters_pca(data_normalized, hmm_states, 'HMM States')
