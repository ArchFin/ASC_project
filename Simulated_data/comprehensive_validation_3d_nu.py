"""
Comprehensive 3D Validation Experiment
======================================
This script combines min_nu (degrees of freedom) validation and smoothness validation 
into a single comprehensive experiment that produces a 3D plot showing:
- X-axis: Minimum Nu values
- Y-axis: Data smoothness values  
- Z-axis: Clustering accuracy

The script generates simulated data with varying smoothness, then tests HMM clustering
performance across different min_nu values for each smoothness level.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import sys
import os
import yaml
from collections import defaultdict
import seaborn as sns

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from HMM.HMM_methods import CustomHMMClustering, principal_component_finder, csv_splitter
from Simulated_data.TET_simulation import TETSimulator

def generate_simulated_data(smoothness_value, save_path):
    """
    Generate simulated TET data with specified smoothness and save to CSV.
    """
    print(f"Generating simulated data with smoothness = {smoothness_value}")
    
    # TET simulation parameters
    feature_names = ['MetaAwareness', 'Presence', 'PhysicalEffort','MentalEffort','Boredom','Receptivity',
                     'EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience']
    n_features = len(feature_names)

    # Define feature groups for structured experiences
    group1_features = ['MetaAwareness', 'Clarity', 'Insightfulness', 'Receptivity', 'Presence', 'Release', 'SpiritualExperience']
    group2_features = ['PhysicalEffort', 'EmotionalIntensity', 'Bliss', 'Boredom', 'Anxiety', 'MentalEffort']
    group3_features = ['Embodiment']

    get_indices = lambda features: [feature_names.index(f) for f in features]
    g1_idx, g2_idx, g3_idx = get_indices(group1_features), get_indices(group2_features), get_indices(group3_features)

    # State definitions
    mean_A = np.full(n_features, 0.5)
    mean_A[g1_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8, 0.85]
    mean_A[g2_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2]
    mean_A[g3_idx] = 0.4

    mean_B = np.full(n_features, 0.5)
    mean_B[g1_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2, 0.25]
    mean_B[g2_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8]
    mean_B[g3_idx] = 0.6

    mean_C = (mean_A + mean_B) / 2
    mean_C[g3_idx] = 0.5

    params = {
        'A': {'mean': mean_A, 'cov': np.eye(n_features)*0.05, 'df': 5},
        'B': {'mean': mean_B, 'cov': np.eye(n_features)*0.05, 'df': 5},
        'C': {'mean': mean_C, 'cov': np.eye(n_features)*0.1, 'df': 5}
    }
    
    # Transition matrix with longer-lasting C state
    transition_matrix = [
        [0.90, 0.00, 0.10],  # state A: mostly stays, can go to C
        [0.00, 0.90, 0.10],  # state B: mostly stays, can go to C  
        [0.05, 0.05, 0.90]   # state C: mostly stays (longer duration)
    ]
    initial_probs = [1.0, 0.0, 0.0]

    # Generate data structure
    n_subjects = 10  # Reduced for faster computation
    n_weeks = 2
    n_sessions = 5
    timepoints_per_session = 120  # Reduced for faster computation
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
        smoothness=smoothness_value
    )
    
    print(f"Simulating {n_subjects} subjects × {n_weeks} weeks × {n_sessions} sessions × {timepoints_per_session} timepoints")
    
    for subj in subjects:
        for week in weeks:
            for sess in sessions:
                sim_data_block, sim_states_block = sim.simulate_tet(timepoints_per_session)
                all_data.append(sim_data_block)
                all_states.append(sim_states_block)
                subject_col.extend([subj]*timepoints_per_session)
                week_col.extend([week]*timepoints_per_session)
                session_col.extend([sess]*timepoints_per_session)

    sim_data_full = pd.concat(all_data, ignore_index=True)
    sim_states_full = np.concatenate(all_states)
    
    print(f"Generated {len(sim_data_full)} total timepoints")
    print(f"State distribution: {np.unique(sim_states_full, return_counts=True)}")
    
    extra_cols = {
        'Subject': subject_col,
        'Week': week_col,
        'Session': session_col,
        'Condition': 'Simulated'
    }
    
    template_columns = [
        'Subject','Week','Session','Condition','MetaAwareness','Presence','PhysicalEffort','MentalEffort',
        'Boredom','Receptivity','EmotionalIntensity','Clarity','Release','Bliss','Embodiment',
        'Insightfulness','Anxiety','SpiritualExperience','Cluster'
    ]
    
    TETSimulator.save_simulated_tet_data(
        sim_data_full, sim_states_full,
        save_path,
        extra_columns=extra_cols,
        template_columns=template_columns
    )
    
    print(f"Saved simulated data to: {save_path}")
    return save_path

def run_validation_for_nu_and_smoothness(min_nu_value, data_file_path, config):
    """
    Run HMM clustering and validation for a specific min_nu value on pre-generated data.
    """
    print(f"    Running HMM validation for min_nu = {min_nu_value:.2f}")
    
    # Load the pre-generated data
    csv_splitter_instance = csv_splitter(data_file_path)
    df_csv_file_original = csv_splitter_instance.read_CSV()
    if df_csv_file_original is None:
        raise ValueError("CSV file could not be read.")
    
    print(f"      Loaded {len(df_csv_file_original)} rows of data")
    
    feelings = config['feelings']
    pc_finder = principal_component_finder(df_csv_file_original, feelings,
                                           config['no_dimensions_PCA'],
                                           config['savelocation_TET'])
    principal_components, _, _ = pc_finder.PCA_TOT()
    print(f"      PCA complete. Shape: {principal_components.shape}")

    # HMM Clustering
    clustering = CustomHMMClustering(
        data_file_path, config['savelocation_TET'],
        df_csv_file_original, feelings, principal_components,
        config['no_of_jumps'], 0.1
    )
    
    _, _, _, notransitions_df = clustering.run(
        num_base_states=2, num_iterations=30, num_repetitions=1,
        gamma_threshold=0.03, min_nu=min_nu_value  # Using fixed gamma, variable nu
    )
    print(f"      HMM clustering complete. Output shape: {notransitions_df.shape}")

    # Validation
    sim_labels, _ = pd.factorize(notransitions_df['Cluster'])
    pred_labels = notransitions_df['labels'].astype(int).values
    
    print(f"      True labels: {np.unique(sim_labels)} (counts: {np.bincount(sim_labels)})")
    print(f"      Pred labels: {np.unique(pred_labels)} (counts: {np.bincount(pred_labels)})")

    # Hungarian alignment
    D = max(sim_labels.max(), pred_labels.max()) + 1
    cost_matrix = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = -np.sum((sim_labels == i) & (pred_labels == j))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {j: i for i, j in zip(row_ind, col_ind)}
    aligned_pred = np.array([mapping.get(lbl, lbl) for lbl in pred_labels])
    
    print(f"      Label mapping: {mapping}")

    acc = accuracy_score(sim_labels, aligned_pred)
    nmi = normalized_mutual_info_score(sim_labels, aligned_pred)
    
    # Per-state accuracy
    unique_labels = np.unique(sim_labels)
    per_state_acc = {}
    for label in unique_labels:
        true_mask = (sim_labels == label)
        if np.sum(true_mask) > 0:
            state_accuracy = np.mean(aligned_pred[true_mask] == label)
            per_state_acc[label] = state_accuracy
        else:
            per_state_acc[label] = np.nan

    print(f"      Results: Acc={acc:.4f}, NMI={nmi:.4f}, Per-state={per_state_acc}")
    
    return acc, nmi, per_state_acc

def main():
    """
    Main function to run comprehensive 3D validation.
    """
    print("=" * 60)
    print("COMPREHENSIVE 3D VALIDATION EXPERIMENT")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(project_root, 'Simulation.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Define parameter ranges
    smoothness_values = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    min_nu_values = np.arange(2, 41) # Use integers from 2 to 40

    print(f"Testing {len(smoothness_values)} smoothness values: {smoothness_values}")
    print(f"Testing {len(min_nu_values)} min_nu values: {np.round(min_nu_values, 2).tolist()}")
    print(f"Total combinations: {len(smoothness_values) * len(min_nu_values)}")
    
    # Results storage
    results = []
    smoothness_grid = []
    min_nu_grid = []
    accuracy_grid = []
    nmi_grid = []
    
    temp_data_dir = '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/temp_validation_data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Main experiment loop
    for i, smoothness in enumerate(smoothness_values):
        print(f"\n{'='*50}")
        print(f"SMOOTHNESS LEVEL {i+1}/{len(smoothness_values)}: {smoothness}")
        print(f"{'='*50}")
        
        # Generate data for this smoothness level
        data_file_path = os.path.join(temp_data_dir, f'sim_data_smooth_{smoothness}.csv')
        generate_simulated_data(smoothness, data_file_path)
        
        # Test all min_nu values for this smoothness
        for j, min_nu in enumerate(min_nu_values):
            print(f"\n  Min Nu {j+1}/{len(min_nu_values)}: {min_nu:.2f}")
            
            try:
                acc, nmi, per_state_acc = run_validation_for_nu_and_smoothness(
                    min_nu, data_file_path, config
                )
                
                # Store results
                results.append({
                    'smoothness': smoothness,
                    'min_nu': min_nu,
                    'accuracy': acc,
                    'nmi': nmi,
                    'per_state_acc': per_state_acc
                })
                
                smoothness_grid.append(smoothness)
                min_nu_grid.append(min_nu)
                accuracy_grid.append(acc)
                nmi_grid.append(nmi)
                
                print(f"    SUCCESS: Acc={acc:.4f}, NMI={nmi:.4f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                # Store NaN results to maintain grid structure
                results.append({
                    'smoothness': smoothness,
                    'min_nu': min_nu,
                    'accuracy': np.nan,
                    'nmi': np.nan,
                    'per_state_acc': {}
                })
                
                smoothness_grid.append(smoothness)
                min_nu_grid.append(min_nu)
                accuracy_grid.append(np.nan)
                nmi_grid.append(np.nan)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE - GENERATING RESULTS")
    print(f"{'='*60}")
    
    # Convert to numpy arrays
    smoothness_grid = np.array(smoothness_grid)
    min_nu_grid = np.array(min_nu_grid)
    accuracy_grid = np.array(accuracy_grid)
    nmi_grid = np.array(nmi_grid)
    
    # Print summary statistics
    valid_acc = accuracy_grid[~np.isnan(accuracy_grid)]
    valid_nmi = nmi_grid[~np.isnan(nmi_grid)]
    
    print(f"Valid results: {len(valid_acc)}/{len(accuracy_grid)}")
    print(f"Accuracy range: {valid_acc.min():.4f} - {valid_acc.max():.4f} (mean: {valid_acc.mean():.4f})")
    print(f"NMI range: {valid_nmi.min():.4f} - {valid_nmi.max():.4f} (mean: {valid_nmi.mean():.4f})")
    
    # --- Find and print optimal nu for each smoothness ---
    results_df = pd.DataFrame(results)
    results_df.dropna(subset=['accuracy'], inplace=True)
    
    if not results_df.empty:
        optimal_indices = results_df.loc[results_df.groupby('smoothness')['accuracy'].idxmax()]
        optimal_params = optimal_indices[['smoothness', 'min_nu', 'accuracy']].rename(
            columns={'min_nu': 'Optimal Min Nu', 'accuracy': 'Max Accuracy'}
        ).sort_values('smoothness').reset_index(drop=True)
        
        print("\n" + "="*60)
        print("Optimal Min Nu for each Smoothness Level (based on max accuracy)")
        print("="*60)
        print(optimal_params.to_string())
        print("="*60 + "\n")
    else:
        optimal_params = pd.DataFrame()
        print("\nNo valid results to determine optimal parameters.\n")
    # ----------------------------------------------------

    # Create 3D plots
    fig = plt.figure(figsize=(16, 14))
    
    # 3D Accuracy plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter1 = ax1.scatter(min_nu_grid, smoothness_grid, accuracy_grid, 
                          c=accuracy_grid, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Minimum Nu')
    ax1.set_ylabel('Data Smoothness')
    ax1.set_zlabel('Accuracy')
    ax1.set_title('3D: Accuracy vs Min Nu vs Smoothness')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # 3D NMI plot
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(min_nu_grid, smoothness_grid, nmi_grid, 
                          c=nmi_grid, cmap='plasma', s=50, alpha=0.7)
    ax2.set_xlabel('Minimum Nu')
    ax2.set_ylabel('Data Smoothness')
    ax2.set_zlabel('NMI')
    ax2.set_title('3D: NMI vs Min Nu vs Smoothness')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # 2D Heatmap - Accuracy
    ax3 = fig.add_subplot(223)
    acc_matrix = np.full((len(smoothness_values), len(min_nu_values)), np.nan)
    min_nu_to_idx = {val: i for i, val in enumerate(min_nu_values)}
    for result in results:
        if not np.isnan(result['accuracy']):
            smooth_idx = smoothness_values.index(result['smoothness'])
            nu_idx = min_nu_to_idx.get(result['min_nu'])
            if nu_idx is not None:
                acc_matrix[smooth_idx, nu_idx] = result['accuracy']
    
    im1 = ax3.imshow(acc_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax3.set_xticks(np.arange(len(min_nu_values)))
    ax3.set_xticklabels([f'{g}' for g in min_nu_values], rotation=45, ha="right")
    ax3.set_yticks(np.arange(len(smoothness_values)))
    ax3.set_yticklabels(smoothness_values)
    ax3.set_xlabel('Minimum Nu')
    ax3.set_ylabel('Data Smoothness')
    ax3.set_title('2D Heatmap: Accuracy with Optimal Nu')
    plt.colorbar(im1, ax=ax3, label='Accuracy')
    
    # Overlay optimal points on heatmap
    if not optimal_params.empty:
        optimal_smoothness_idx = [smoothness_values.index(s) for s in optimal_params['smoothness']]
        optimal_nu_idx = [min_nu_to_idx[n] for n in optimal_params['Optimal Min Nu']]
        ax3.scatter(optimal_nu_idx, optimal_smoothness_idx, marker='o', s=80, facecolors='none', edgecolors='r', linewidth=1.5, label='Optimal Nu')
        ax3.legend()

    # 2D Heatmap - NMI
    ax4 = fig.add_subplot(224)
    nmi_matrix = np.full((len(smoothness_values), len(min_nu_values)), np.nan)
    for result in results:
        if not np.isnan(result['nmi']):
            smooth_idx = smoothness_values.index(result['smoothness'])
            nu_idx = min_nu_to_idx.get(result['min_nu'])
            if nu_idx is not None:
                nmi_matrix[smooth_idx, nu_idx] = result['nmi']
    
    im2 = ax4.imshow(nmi_matrix, cmap='plasma', aspect='auto', origin='lower')
    ax4.set_xticks(np.arange(len(min_nu_values)))
    ax4.set_xticklabels([f'{g}' for g in min_nu_values], rotation=45, ha="right")
    ax4.set_yticks(np.arange(len(smoothness_values)))
    ax4.set_yticklabels(smoothness_values)
    ax4.set_xlabel('Minimum Nu')
    ax4.set_ylabel('Data Smoothness')
    ax4.set_title('2D Heatmap: NMI with Optimal Nu')
    plt.colorbar(im2, ax=ax4, label='NMI')

    # Overlay optimal points on NMI heatmap as well
    if not optimal_params.empty:
        ax4.scatter(optimal_nu_idx, optimal_smoothness_idx, marker='o', s=80, facecolors='none', edgecolors='cyan', linewidth=1.5, label='Optimal Nu (from Acc)')
        ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle('Comprehensive Validation: Min Nu vs. Data Smoothness', fontsize=18, weight='bold')
    
    # Save results
    save_path = os.path.join(config['savelocation_TET'], 'comprehensive_3d_validation_nu_smoothness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"3D validation plot saved to: {save_path}")
    
    # Save results data
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(config['savelocation_TET'], 'comprehensive_3d_validation_nu_smoothness_data.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results data saved to: {results_csv_path}")
    
    # Clean up temporary files
    import shutil
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)
        print("Cleaned up temporary data files")
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE 3D VALIDATION COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()