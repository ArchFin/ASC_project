import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import sys
import os
import yaml
from collections import defaultdict

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from HMM.HMM_methods import CustomHMMClustering, principal_component_finder, csv_splitter

def run_validation_for_gamma(gamma_threshold, config):
    """
    Runs the HMM clustering and validation for a single gamma threshold.
    """
    # --- Data Loading and Preparation ---
    print("--- Running validation for gamma = {:.3f} ---".format(gamma_threshold))
    csv_splitter_instance = csv_splitter(config['filelocation_TET'])
    df_csv_file_original = csv_splitter_instance.read_CSV()
    if df_csv_file_original is None:
        raise ValueError("CSV file could not be read.")
    print(f"Loaded data with {len(df_csv_file_original)} rows.")
    print("Data head:\n", df_csv_file_original.head())

    feelings = config['feelings']
    print(f"Feelings being used: {feelings}")
    pc_finder = principal_component_finder(df_csv_file_original, feelings,
                                           config['no_dimensions_PCA'],
                                           config['savelocation_TET'])
    principal_components, _, _ = pc_finder.PCA_TOT()
    print(f"PCA complete. Principal components shape: {principal_components.shape}")
    print("Sample of principal components:\n", principal_components[:5])

    # --- HMM Clustering ---
    clustering = CustomHMMClustering(
        config['filelocation_TET'], config['savelocation_TET'],
        df_csv_file_original, feelings, principal_components,
        config['no_of_jumps'], 0.1
    )
    
    _, _, _, notransitions_df = clustering.run(
        num_base_states=2, num_iterations=30, num_repetitions=1,
        gamma_threshold=0.05, min_nu=gamma_threshold
    )
    print("HMM clustering complete.")
    print("Clustering results head:\n", notransitions_df.head())

    # --- Validation ---
    sim_labels, _ = pd.factorize(notransitions_df['Cluster'])
    pred_labels = notransitions_df['labels'].astype(int).values
    print(f"Simulated labels (true): {np.unique(sim_labels)}")
    print(f"Predicted labels (raw): {np.unique(pred_labels)}")
    print(f"Number of simulated labels: {len(sim_labels)}")
    print(f"Number of predicted labels: {len(pred_labels)}")

    D = max(sim_labels.max(), pred_labels.max()) + 1
    cost_matrix = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = -np.sum((sim_labels == i) & (pred_labels == j))
    
    print("Cost matrix for label alignment:\n", cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {j: i for i, j in zip(row_ind, col_ind)}
    aligned_pred = np.array([mapping.get(lbl, lbl) for lbl in pred_labels])
    print(f"Label mapping for alignment: {mapping}")
    print("Aligned predicted labels (sample):", aligned_pred[:20])

    acc = accuracy_score(sim_labels, aligned_pred)
    nmi = normalized_mutual_info_score(sim_labels, aligned_pred)
    print(f"  Accuracy: {acc:.4f}, NMI: {nmi:.4f}")
    
    # --- Per-state accuracy calculation ---
    unique_labels = np.unique(sim_labels)
    per_state_acc = {}
    for label in unique_labels:
        true_mask = (sim_labels == label)
        if np.sum(true_mask) > 0:
            state_accuracy = np.mean(aligned_pred[true_mask] == label)
            per_state_acc[label] = state_accuracy
        else:
            per_state_acc[label] = np.nan # No instances of this true label
    
    print(f"  Per-state accuracies: {per_state_acc}")
    print("-" * 20)

    return acc, nmi, per_state_acc

def main():
    """
    Main function to run the gamma validation sweep.
    """
    # Load YAML config
    config_path = os.path.join(project_root, 'Simulation.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Define the range of gamma thresholds to test
    gamma_thresholds = np.linspace(2, 20, 20)
    accuracies = []
    nmis = []
    state_accuracies = defaultdict(list)

    print("Starting gamma threshold validation sweep...")
    print(f"Testing {len(gamma_thresholds)} gamma values from {gamma_thresholds[0]:.2f} to {gamma_thresholds[-1]:.2f}")
    for gamma in gamma_thresholds:
        print(f"  Testing gamma = {gamma:.3f}")
        try:
            acc, nmi, per_state_acc = run_validation_for_gamma(gamma, config)
            accuracies.append(acc)
            nmis.append(nmi)
            for state, state_acc in per_state_acc.items():
                state_accuracies[state].append(state_acc)
        except Exception as e:
            print(f"    Error processing gamma={gamma:.3f}: {e}")
            accuracies.append(np.nan)
            nmis.append(np.nan)
            # Ensure state_accuracies lists stay aligned
            known_states = state_accuracies.keys()
            for state in known_states:
                 state_accuracies[state].append(np.nan)

    print("Validation sweep complete.")
    print("\n--- Final Results ---")
    print(f"Gamma Thresholds: {np.round(gamma_thresholds, 3).tolist()}")
    print(f"Overall Accuracies: {[round(a, 4) if not np.isnan(a) else 'NaN' for a in accuracies]}")
    print(f"NMIs: {[round(n, 4) if not np.isnan(n) else 'NaN' for n in nmis]}")
    for state, accs in state_accuracies.items():
        print(f"State {state} Accuracies: {[round(a, 4) if not np.isnan(a) else 'NaN' for a in accs]}")
    print("---------------------\n")


    # --- Plotting Results ---
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'P', 'D', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, len(state_accuracies) + 2))

    ax.plot(gamma_thresholds, accuracies, marker=markers[0], color=colors[0], label='Overall Accuracy')
    ax.plot(gamma_thresholds, nmis, marker=markers[1], color=colors[1], label='NMI')

    # Plot per-state accuracies
    sorted_states = sorted(state_accuracies.keys())
    for i, state in enumerate(sorted_states):
        ax.plot(gamma_thresholds, state_accuracies[state], marker=markers[i+2], color=colors[i+2], linestyle='--', label=f'State {state} Accuracy')

    ax.set_xlabel('Minimum Degrees of Freedom')
    ax.set_ylabel('Score')
    ax.set_title('HMM Clustering Performance vs. Degrees of Freedom')
    ax.legend()
    ax.grid(True)
    
    fig.tight_layout()
    
    # Save the plot
    save_path = os.path.join(config['savelocation_TET'], 'nu_validation_performance_10.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Results plot saved to: {save_path}")

if __name__ == '__main__':
    main()