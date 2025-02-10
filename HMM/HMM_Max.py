import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.linalg import inv, det
from scipy.special import gamma
from matplotlib.lines import Line2D
import random
from scipy.io import savemat
import os

# Load YAML file
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

from Max_HMM_methods import Visualiser, DataLoader, HMMModel

all_tet_data, session_ids, weeks, subjects, unique_session_ids = DataLoader.load_tet_data(config['filelocation_TET'], config['feelings'])

all_tet_data = zscore(all_tet_data).values

# Clustering
optimal_k = 4
np.random.seed(12345)
random.seed(12345)
kmeans = KMeans(n_clusters=optimal_k, n_init=1000, random_state=12345)
idx = kmeans.fit_predict(all_tet_data)
C = kmeans.cluster_centers_

# Prepare storage for HMM results across repetitions
num_repetitions = 2
all_trans_probs = np.zeros((optimal_k, optimal_k, num_repetitions))
all_emission_means = np.zeros((optimal_k, all_tet_data.shape[1], num_repetitions))
all_emission_covs = np.zeros((optimal_k, all_tet_data.shape[1], all_tet_data.shape[1], num_repetitions))
all_fs = np.zeros((all_tet_data.shape[0], optimal_k, num_repetitions))
all_state_seqs = np.zeros((all_tet_data.shape[0], num_repetitions))

save_folder = config['savelocation_TET']
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for rep in range(num_repetitions):
    print(f'Repetition {rep + 1} of {num_repetitions}')
    np.random.seed(12345 + rep)
    random.seed(12345 + rep)
    num_states = optimal_k
    num_emissions = all_tet_data.shape[1]
    hmm = HMMModel(num_states, num_emissions, random_seed=12345 + rep)
    print(f"Transition probabilities shape: {hmm.trans_prob.shape}")
    print(f"Emission means shape: {hmm.emission_means.shape}")
    print(f"Emission covariances shape: {hmm.emission_covs.shape}")
    print(f"Degrees of freedom shape: {hmm.nu.shape}")
    
    # Train HMM (using the Baum–Welch training loop)
    trans_prob, emission_means, emission_covs, nu = hmm.train(all_tet_data, num_iterations=10)
    
    # Decode using Viterbi (note: using a subset of features as in the original code)
    state_seq, log_prob = hmm.decode(all_tet_data[:, :len(config['feelings'])])
    alpha, beta, fs, log_lik = hmm.forward_backward(all_tet_data[:, :len(config['feelings'])])
    
    all_trans_probs[:, :, rep] = trans_prob
    all_emission_means[:, :, rep] = emission_means
    all_emission_covs[:, :, :, rep] = emission_covs
    all_fs[:, :, rep] = fs
    all_state_seqs[:, rep] = state_seq
    
    save_file = os.path.join(save_folder, f'repetition_{rep + 1}.mat')
    savemat(save_file, {
        'transProb': trans_prob,
        'emissionMean': emission_means,
        'emissionCov': emission_covs,
        'nu': nu,
        'stateSeq': state_seq,
        'fs': fs,
        'logProb': log_prob,
        'logLik': log_lik
    })

# Average results across repetitions
avg_trans_prob = np.mean(all_trans_probs, axis=2)
avg_emission_means = np.mean(all_emission_means, axis=2)
avg_emission_covs = np.mean(all_emission_covs, axis=3)
avg_fs = np.mean(all_fs, axis=2)
avg_state_seq = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_state_seqs)

min_state_length = 10
changes = np.abs(np.diff(avg_fs, axis=0))
mean_change = np.mean(changes)
std_change = np.std(changes)
print(f'Mean of changes: {mean_change}')
print(f'Standard deviation of changes: {std_change}')
threshold = mean_change + 0.05 * std_change
print(f'Chosen threshold based on data distribution: {threshold}')

labels = [''] * all_tet_data.shape[0]
transitional_windows = [''] * all_tet_data.shape[0]
initial_buffer_size = 5
session_length = 101

for i in range(1, len(avg_state_seq)):
    if avg_state_seq[i] != avg_state_seq[i-1] and (i % session_length) != 1:
        prev_state = avg_state_seq[i-1]
        prev_state_length = 1
        for j in range(i-2, -1, -1):
            if avg_state_seq[j] == prev_state:
                prev_state_length += 1
            else:
                break
        new_state = avg_state_seq[i]
        new_state_length = 1
        for j in range(i+1, len(avg_state_seq)):
            if avg_state_seq[j] == new_state:
                new_state_length += 1
            else:
                break
        if prev_state_length < min_state_length or new_state_length < min_state_length:
            continue
        if new_state_length <= 10:
            continue
        if new_state_length < min_state_length and prev_state_length >= min_state_length and new_state_length >= min_state_length:
            continue
        start_idx = i
        while start_idx > 0 and np.abs(avg_fs[start_idx, new_state] - avg_fs[start_idx-1, new_state]) > threshold:
            start_idx -= 1
        end_idx = i
        while end_idx < len(avg_state_seq) - 1 and np.abs(avg_fs[end_idx, new_state] - avg_fs[end_idx+1, new_state]) > threshold:
            end_idx += 1
        dynamic_buffer_size_start = initial_buffer_size
        while start_idx - dynamic_buffer_size_start > 0 and np.abs(avg_fs[max(0, start_idx - dynamic_buffer_size_start), new_state] - avg_fs[max(0, start_idx - dynamic_buffer_size_start + 1), new_state]) > threshold / 2:
            dynamic_buffer_size_start += 1
        dynamic_buffer_size_end = initial_buffer_size
        while end_idx + dynamic_buffer_size_end < len(avg_state_seq) and np.abs(avg_fs[min(len(avg_state_seq)-1, end_idx + dynamic_buffer_size_end), new_state] - avg_fs[min(len(avg_state_seq)-1, end_idx + dynamic_buffer_size_end - 1), new_state]) > threshold / 2:
            dynamic_buffer_size_end += 1
        start_idx = max(0, start_idx - dynamic_buffer_size_start)
        end_idx = min(len(avg_state_seq)-1, end_idx + dynamic_buffer_size_end)
        if avg_state_seq[start_idx] == avg_state_seq[end_idx]:
            continue
        for j in range(start_idx, end_idx + 1):
            labels[j] = f'Transition from S{avg_state_seq[start_idx]} to S{avg_state_seq[end_idx]}'
            transitional_windows[j] = f'Start: {start_idx}, End: {end_idx}'

num_transitions = sum(1 for label in labels if label)
print(f'Number of transitions labeled: {num_transitions}')

results_table = pd.DataFrame({
    'TETData': list(all_tet_data),
    'State': avg_state_seq,
    'SessionID': session_ids,
    'Label': labels,
    'TransitionalWindow': transitional_windows,
    'Week': weeks,
    'SessionID': unique_session_ids,
    'Subject': subjects
})

print(results_table)
print('Averaged Transition Matrix:')
print(avg_trans_prob)
Visualiser.visualise_clusters_and_transitions(all_tet_data, avg_state_seq, labels, config['savelocation_TET'])
Visualiser.visualise_session(results_table, 1, 101, config['savelocation_TET'])