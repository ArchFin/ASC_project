"""
TET-HMM Smoothness Experiment
----------------------------
This script systematically evaluates the effect of data smoothness on HMM state recovery accuracy using simulated TET data.

Simulated Data:
- Three archetypal states: two base states and one transition/metastable state
- 14 features per timepoint, normalized to [0, 1]
- Markov chain transitions (with a transition matrix matching the real-data-like simulation)
- Each dataset simulates multiple subjects, weeks, and sessions, as in the real TET data structure

Workflow:
- For a range of smoothness values (Gaussian sigma):
    - Simulate a full dataset with known ground-truth states
    - Fit a custom HMM (with a transition state) to the data
    - Decode the most likely state sequence
    - Align predicted states to true states using the Hungarian algorithm
    - Compute overall accuracy, normalized mutual information (NMI), and per-state accuracy
    - Store confusion matrices for selected smoothness values
- Plot accuracy, NMI, and per-state accuracy as a function of smoothness
- Print confusion matrices for selected smoothness values

Outputs:
- PNG plot of accuracy/NMI vs. smoothness
- Printed confusion matrices for selected smoothness values
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Simulated_data.TET_simulation import TETSimulator
from HMM.HMM_methods import HMMModel, CustomHMMClustering
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# Ensure submodules can be imported regardless of working directory
script_dir = '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/'
sim_data_dir = os.path.join(script_dir, 'Simulated_data')
hmm_dir = os.path.join(script_dir, 'HMM')
if sim_data_dir not in sys.path:
    sys.path.insert(0, sim_data_dir)
if hmm_dir not in sys.path:
    sys.path.insert(0, hmm_dir)



# --- 1. Simulation and HMM parameters ---
# Using structured parameters from TET_simulation.py for realistic data
feature_names = [
    'MetaAwareness', 'Presence', 'PhysicalEffort','MentalEffort','Boredom','Receptivity',
    'EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience'
]
n_features = len(feature_names)
get_indices = lambda features: [feature_names.index(f) for f in features]

# Define feature groups for structured, anti-correlated experiences
group1_features = ['MetaAwareness', 'Clarity', 'Insightfulness', 'Receptivity', 'Presence', 'Release', 'SpiritualExperience']
group2_features = ['PhysicalEffort', 'EmotionalIntensity', 'Bliss', 'Boredom', 'Anxiety', 'MentalEffort']
group3_features = ['Embodiment']
g1_idx, g2_idx, g3_idx = get_indices(group1_features), get_indices(group2_features), get_indices(group3_features)

# State 0: "Focused Internal" - High on cognitive, low on somatic
mean_A = np.full(n_features, 0.5)
mean_A[g1_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8, 0.85]
mean_A[g2_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2]
mean_A[g3_idx] = 0.4

# State 1: "Somatic Release" - Low on cognitive, high on somatic
mean_B = np.full(n_features, 0.5)
mean_B[g1_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2, 0.25]
mean_B[g2_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8]
mean_B[g3_idx] = 0.6

# State 2: "Metastable/Transition" - In-between state, higher variance
mean_C = (mean_A + mean_B) / 2
mean_C[g3_idx] = 0.5

params = {
    0: { 'mean': mean_A, 'cov': np.eye(n_features)*0.05, 'df': 5 },
    1: { 'mean': mean_B, 'cov': np.eye(n_features)*0.05, 'df': 5 },
    2: { 'mean': mean_C, 'cov': np.eye(n_features)*0.1,  'df': 5 }
}

transition_matrix = [
    [0.90, 0.00, 0.10],
    [0.00, 0.90, 0.10],
    [0.15, 0.15, 0.70]
]
initial_probs = [1.0, 0.0, 0.0]

# --- 2. Smoothness schedule (small steps at low, larger at high) ---
smoothness_values = np.unique(np.concatenate([
    np.arange(0, 5, 1),
    np.arange(5, 15, 2),
    np.arange(15, 30, 5),
    np.arange(30, 55, 10)
]))

# --- 3. Results storage ---
accuracies = []
nmis = []
conf_matrices = {}
per_state_accuracies = {i: [] for i in range(3)}  # 3 states

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
    subject_col = []
    week_col = []
    session_col = []
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
                subject_col.extend([subj]*timepoints_per_session)
                week_col.extend([week]*timepoints_per_session)
                session_col.extend([sess]*timepoints_per_session)
    sim_data_full = pd.concat(all_data, ignore_index=True)
    sim_states_full = np.concatenate(all_states)
    sim_data_full['Subject'] = subject_col
    sim_data_full['Week'] = week_col
    sim_data_full['Session'] = session_col
    return sim_data_full, sim_states_full

# --- 4. Main experiment loop ---
for smooth in smoothness_values:
    print(f"\n=== Smoothness: {smooth} ===")
    # Simulate realistic data structure for this smoothness
    data_df, true_states = simulate_full_tet_dataset(smooth)
    true_states = np.array(true_states, dtype=int)

    # Fit HMM using the CustomHMMClustering pipeline
    clustering = CustomHMMClustering(
        filelocation_TET=None,
        savelocation_TET='./temp_results/', # Dummy path
        df_csv_file_original=data_df,
        feelings=feature_names,
        principal_components=None, # Not needed for clustering itself
        no_of_jumps=1,
        transition_contributions=1.0
    )
    # Run clustering but skip plotting and other non-essential steps
    clustering.preprocess_data()
    clustering.perform_clustering(
        num_base_states=2,
        num_iterations=15,
        num_repetitions=1,
        base_seed=42
    )
    pred_states = clustering.array['labels'].values


    # Align predicted states to true using Hungarian algorithm
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

    pred_aligned, mapping = align_states(true_states, pred_states)
    acc = accuracy_score(true_states, pred_aligned)
    nmi = normalized_mutual_info_score(true_states, pred_aligned)
    accuracies.append(acc)
    nmis.append(nmi)
    print(f"Smoothness={smooth}: Accuracy={acc:.3f}, NMI={nmi:.3f}")

    # Store confusion matrix for selected smoothness
    conf = confusion_matrix(true_states, pred_aligned, labels=[0,1,2])
    # Per-state accuracy: diagonal / row sum
    row_sums = conf.sum(axis=1)
    for i in range(3):
        per_state_accuracies[i].append(conf[i, i] / row_sums[i] if row_sums[i] > 0 else np.nan)
    if smooth in [0, 5, 15, 30, 50]:
        conf_matrices[smooth] = conf
        print(f"Confusion matrix (smoothness={smooth}):\n{conf}")
        print(f"Accuracy: {acc:.3f}, NMI: {nmi:.3f}")

# --- 5. Plot accuracy vs. smoothness ---
plt.figure(figsize=(10, 6))
plt.plot(smoothness_values, accuracies, marker='o', label='Overall Accuracy')
plt.plot(smoothness_values, nmis, marker='s', label='NMI')
for i in range(3):
    plt.plot(smoothness_values, per_state_accuracies[i], marker='^', label=f'State {i} Accuracy')
plt.xlabel('Smoothness (Gaussian sigma)')
plt.ylabel('Score')
plt.title('HMM State Labelling Accuracy vs. Data Smoothness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(script_dir, 'TET_HMM_smoothness_accuracy_smooth_varied_states.png')
plt.savefig(plot_path)
plt.close()

# --- 6. Print confusion matrices for selected smoothness values ---
for smooth in conf_matrices:
    print(f"\n=== Confusion Matrix for Smoothness {smooth} ===")
    print(conf_matrices[smooth])
