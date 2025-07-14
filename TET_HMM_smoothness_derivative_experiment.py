"""
TET-HMM Smoothness Derivative Experiment
---------------------------------------
This script augments TET features with first and second temporal derivatives,
then runs the smoothness/accuracy analysis as in your main experiment.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulated_data.TET_simulation import TETSimulator
from HMM.HMM_methods import HMMModel
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# --- 1. Simulation and HMM parameters ---
params = {
    0: {'mean': [0]*14, 'cov': np.eye(14), 'df': 5},
    1: {'mean': [2]*14, 'cov': np.eye(14)*1.5, 'df': 5},
    2: {'mean': [1]*14, 'cov': np.eye(14)*2.0, 'df': 5}
}
feature_names = [
    'MetaAwareness', 'Presence', 'PhysicalEffort','MentalEffort','Boredom','Receptivity',
    'EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience'
]
transition_matrix = [
    [0.90, 0.00, 0.10],
    [0.00, 0.90, 0.10],
    [0.15, 0.15, 0.70]
]
initial_probs = [1.0, 0.0, 0.0]

smoothness_values = np.unique(np.concatenate([
    np.arange(0, 5, 1),
    np.arange(5, 15, 2),
    np.arange(15, 30, 5),
    np.arange(30, 55, 10)
]))

accuracies = []
nmis = []
per_state_accuracies = {i: [] for i in range(3)}

# --- Derivative feature augmentation ---
def augment_with_derivatives(df):
    X = df.values
    dX = np.diff(X, axis=0, prepend=X[:1])
    d2X = np.diff(dX, axis=0, prepend=dX[:1])
    X_aug = np.hstack([X, dX, d2X])
    return X_aug

def simulate_full_tet_dataset(smoothness):
    n_subjects = 14
    n_weeks = 4
    n_sessions = 7
    timepoints_per_session = 180
    subjects = [f"sim{str(i+1).zfill(2)}" for i in range(n_subjects)]
    weeks = [f"week_{i+1}" for i in range(n_weeks)]
    sessions = [f"run_{str(i+1).zfill(2)}" for i in range(n_sessions)]
    all_data = []
    all_states = []
    sim = TETSimulator(
        params=params,
        feature_names=feature_names,
        transition_matrix=np.array(transition_matrix),
        initial_probs=np.array(initial_probs),
        smoothness=smoothness
    )
    for subj in subjects:
        for week in weeks:
            for sess in sessions:
                sim_data_block, sim_states_block = sim.simulate_tet(
                    timepoints_per_session
                )
                all_data.append(sim_data_block)
                all_states.append(sim_states_block)
    sim_data_full = pd.concat(all_data, ignore_index=True)
    sim_states_full = np.concatenate(all_states)
    return sim_data_full, sim_states_full

def align_states(true, pred):
    D = max(true.max(), pred.max()) + 1
    cost = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            cost[i, j] = -np.sum((true == i) & (pred == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {j: i for i, j in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(lbl, lbl) for lbl in pred])
    return aligned, mapping

# --- Main experiment loop ---
for smooth in smoothness_values:
    print(f"\n=== Smoothness: {smooth} (with derivatives) ===")
    data_df, true_states = simulate_full_tet_dataset(smooth)
    X_aug = augment_with_derivatives(data_df)
    true_states = np.array(true_states, dtype=int)
    # Fit HMM
    hmm = HMMModel(
        num_base_states=2,
        num_emissions=X_aug.shape[1],
        data=X_aug,
        random_seed=42,
        base_prior=0.1,
        extra_prior=0.1,
        extra_self_prior=0.05,
        transition_temp=1.0
    )
    hmm.train(X_aug, num_iterations=15, transition_contributions=1.0, transition_constraint_lim=0.5)
    pred_states, _ = hmm.decode(X_aug)
    pred_aligned, mapping = align_states(true_states, pred_states)
    acc = accuracy_score(true_states, pred_aligned)
    nmi = normalized_mutual_info_score(true_states, pred_aligned)
    accuracies.append(acc)
    nmis.append(nmi)
    # Per-state accuracy
    conf = confusion_matrix(true_states, pred_aligned, labels=[0,1,2])
    row_sums = conf.sum(axis=1)
    for i in range(3):
        per_state_accuracies[i].append(conf[i, i] / row_sums[i] if row_sums[i] > 0 else np.nan)
    if smooth in [0, 5, 15, 30, 50]:
        print(f"Confusion matrix (smoothness={smooth}):\n{conf}")
        print(f"Accuracy: {acc:.3f}, NMI: {nmi:.3f}")

# --- Plot accuracy vs. smoothness ---
plt.figure(figsize=(10, 6))
plt.plot(smoothness_values, accuracies, marker='o', label='Overall Accuracy')
plt.plot(smoothness_values, nmis, marker='s', label='NMI')
for i in range(3):
    plt.plot(smoothness_values, per_state_accuracies[i], marker='^', label=f'State {i} Accuracy')
plt.xlabel('Smoothness (Gaussian sigma)')
plt.ylabel('Score')
plt.title('HMM (with Derivatives) State Labelling Accuracy vs. Data Smoothness')
plt.legend()
plt.grid(True)
plt.tight_layout()
script_dir = '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/'
plot_path = os.path.join(script_dir, 'TET_HMM_smoothness_derivative_accuracy.png')
plt.savefig(plot_path)
plt.show()
