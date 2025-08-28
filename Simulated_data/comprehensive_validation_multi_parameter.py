#!/usr/bin/env python3
"""
Comprehensive Multi-Parameter Validation Experiment
===================================================
This script combines all three parameter validation experiments into a single 
comprehensive analysis that produces optimal hyperparameter combinations for 
different data smoothness levels.

The script tests:
- Gamma threshold values (transition state detection sensitivity)
- Min Nu values (degrees of freedom for t-distribution)
- Transition contribution values (transition state weight in training)

For each smoothness level, it finds the optimal combination of all three parameters
that maximizes clustering accuracy.
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
from joblib import Parallel, delayed
import itertools
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from HMM.HMM_methods import CustomHMMClustering, principal_component_finder, csv_splitter
from Simulated_data.TET_simulation import TETSimulator

def generate_simulated_data(smoothness_value, save_path, n_primary_states=2):
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

    # --- State Definitions ---
    all_primary_states = {}
    
    # State A: "Focused Internal"
    mean_A = np.full(n_features, 0.5)
    mean_A[g1_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8, 0.85]
    mean_A[g2_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2]
    mean_A[g3_idx] = 0.4
    all_primary_states['A'] = {'mean': mean_A, 'cov': np.eye(n_features)*0.05, 'df': 5}
    
    # State B: "Somatic Release"
    mean_B = np.full(n_features, 0.5)
    mean_B[g1_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2, 0.25]
    mean_B[g2_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8]
    mean_B[g3_idx] = 0.6
    all_primary_states['B'] = {'mean': mean_B, 'cov': np.eye(n_features)*0.05, 'df': 5}
    
    if n_primary_states >= 3:
        # State D: "Neutral/Mindful"
        mean_D = np.full(n_features, 0.5)
        mean_D[g1_idx] = 0.5
        mean_D[g2_idx] = 0.3
        mean_D[g3_idx] = 0.5
        all_primary_states['D'] = {'mean': mean_D, 'cov': np.eye(n_features)*0.03, 'df': 5}

    if n_primary_states >= 4:
        # State E: "High Arousal/Overwhelmed" - High on both, high variance
        mean_E = np.full(n_features, 0.5)
        mean_E[g1_idx] = 0.8
        mean_E[g2_idx] = 0.8
        mean_E[g3_idx] = 0.7
        all_primary_states['E'] = {'mean': mean_E, 'cov': np.eye(n_features)*0.12, 'df': 5}

    primary_state_labels = list(all_primary_states.keys())[:n_primary_states]
    primary_states = {label: all_primary_states[label] for label in primary_state_labels}

    # State C: "Metastable/Transition" - Average of all primary states
    mean_C = np.mean([p['mean'] for p in primary_states.values()], axis=0)
    
    params = primary_states.copy()
    params['C'] = { 'mean': mean_C, 'cov': np.eye(n_features)*0.1,  'df': 5 }
    
    final_state_labels = primary_state_labels + ['C']
    n_states = len(final_state_labels)
    transition_matrix = np.zeros((n_states, n_states))
    
    p_stay = 0.90
    p_to_C = 1 - p_stay
    c_idx = final_state_labels.index('C')
    for i in range(len(primary_states)):
        transition_matrix[i, i] = p_stay
        transition_matrix[i, c_idx] = p_to_C

    p_C_stay = 0.90
    p_C_to_primary = (1 - p_C_stay) / len(primary_states)
    transition_matrix[c_idx, c_idx] = p_C_stay
    for i in range(len(primary_states)):
        transition_matrix[c_idx, i] = p_C_to_primary
    
    initial_probs = np.zeros(n_states)
    initial_probs[0] = 1.0

    # Generate data structure
    n_subjects = 8  # Reduced for faster computation with more parameter combinations
    n_weeks = 2
    n_sessions = 4
    timepoints_per_session = 100  # Reduced for faster computation
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
    
    # Calculate the measured smoothness of the generated data
    measured_smoothness = CustomHMMClustering.calculate_smoothness(sim_data_full, feature_names)
    print(f"Generated data measured smoothness: {measured_smoothness:.6f}")
    
    return save_path, measured_smoothness

def run_validation_for_all_params(gamma_threshold, min_nu_value, tc_value, data_file_path, config, n_primary_states=2):
    """
    Run HMM clustering and validation for a specific combination of all three parameters.
    """
    print(f"    Running HMM validation for gamma={gamma_threshold:.3f}, min_nu={min_nu_value}, tc={tc_value}")
    
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

    # HMM Clustering with all three parameters
    clustering = CustomHMMClustering(
        data_file_path, config['savelocation_TET'],
        df_csv_file_original, feelings, principal_components,
        config['no_of_jumps'], tc_value
    )
    
    _, _, _, notransitions_df = clustering.run(
        num_base_states=n_primary_states, num_iterations=25, num_repetitions=1,
        gamma_threshold=gamma_threshold, min_nu=min_nu_value
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

def run_single_validation(smoothness, gamma, min_nu, tc, config, temp_data_dir, n_primary_states_to_test, smoothness_mapping):
    """
    A wrapper function to run a single validation instance for a given parameter combination.
    This function can be called in parallel. It assumes data has been pre-generated.
    """
    print(f"Starting validation for smoothness={smoothness}, gamma={gamma:.3f}, min_nu={min_nu}, tc={tc}")
    data_file_path = os.path.join(temp_data_dir, f'sim_data_smooth_{smoothness}.csv')

    try:
        acc, nmi, per_state_acc = run_validation_for_all_params(
            gamma, min_nu, tc, data_file_path, config, n_primary_states=n_primary_states_to_test
        )
        
        # Get the measured smoothness for this simulation parameter
        measured_smoothness = smoothness_mapping.get(smoothness, smoothness)
        
        result = {
            'sim_smoothness_param': smoothness,
            'measured_smoothness': measured_smoothness,
            'gamma_threshold': gamma,
            'min_nu': min_nu,
            'transition_contributions': tc,
            'accuracy': acc,
            'nmi': nmi,
            'per_state_acc': per_state_acc
        }
        print(f"  SUCCESS for smoothness={smoothness} (measured={measured_smoothness:.6f}), gamma={gamma:.3f}, min_nu={min_nu}, tc={tc}: Acc={acc:.4f}, NMI={nmi:.4f}")
        return result
        
    except Exception as e:
        print(f"  ERROR for smoothness={smoothness}, gamma={gamma:.3f}, min_nu={min_nu}, tc={tc}: {e}")
        return {
            'sim_smoothness_param': smoothness,
            'measured_smoothness': smoothness_mapping.get(smoothness, smoothness),
            'gamma_threshold': gamma,
            'min_nu': min_nu,
            'transition_contributions': tc,
            'accuracy': np.nan,
            'nmi': np.nan,
            'per_state_acc': {}
        }

def create_comprehensive_plots(results_df, config, n_primary_states_to_test):
    """
    Create comprehensive visualization plots for all parameter combinations.
    """
    print("Creating comprehensive visualization plots...")
    
    # Filter valid results
    valid_results = results_df.dropna(subset=['accuracy']).copy()
    
    if valid_results.empty:
        print("No valid results to plot!")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 3D scatter plot: Smoothness vs Accuracy vs Gamma (colored by min_nu)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(valid_results['measured_smoothness'], valid_results['accuracy'], valid_results['gamma_threshold'],
                          c=valid_results['min_nu'], cmap='viridis', s=30, alpha=0.7)
    ax1.set_xlabel('Data Smoothness')
    ax1.set_ylabel('Accuracy')
    ax1.set_zlabel('Gamma Threshold')
    ax1.set_title('3D: Smoothness vs Accuracy vs Gamma\n(colored by min_nu)')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Min Nu')
    
    # 2. 3D scatter plot: Smoothness vs Accuracy vs TC (colored by gamma)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(valid_results['measured_smoothness'], valid_results['accuracy'], valid_results['transition_contributions'],
                          c=valid_results['gamma_threshold'], cmap='plasma', s=30, alpha=0.7)
    ax2.set_xlabel('Data Smoothness')
    ax2.set_ylabel('Accuracy')
    ax2.set_zlabel('Transition Contributions')
    ax2.set_title('3D: Smoothness vs Accuracy vs TC\n(colored by gamma)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label='Gamma Threshold')
    
    # 3. 2D heatmap: Smoothness vs Gamma (average accuracy across other parameters)
    ax3 = fig.add_subplot(2, 3, 3)
    gamma_smooth_acc = valid_results.groupby(['measured_smoothness', 'gamma_threshold'])['accuracy'].mean().reset_index()
    gamma_smooth_matrix = gamma_smooth_acc.pivot(index='measured_smoothness', columns='gamma_threshold', values='accuracy')
    sns.heatmap(gamma_smooth_matrix, ax=ax3, cmap='viridis', cbar_kws={'label': 'Average Accuracy'})
    ax3.set_title('Smoothness vs Gamma\n(avg accuracy across min_nu & TC)')
    ax3.set_xlabel('Gamma Threshold')
    ax3.set_ylabel('Data Smoothness')
    
    # 4. 2D heatmap: Smoothness vs Min Nu (average accuracy across other parameters)
    ax4 = fig.add_subplot(2, 3, 4)
    nu_smooth_acc = valid_results.groupby(['measured_smoothness', 'min_nu'])['accuracy'].mean().reset_index()
    nu_smooth_matrix = nu_smooth_acc.pivot(index='measured_smoothness', columns='min_nu', values='accuracy')
    sns.heatmap(nu_smooth_matrix, ax=ax4, cmap='plasma', cbar_kws={'label': 'Average Accuracy'})
    ax4.set_title('Smoothness vs Min Nu\n(avg accuracy across gamma & TC)')
    ax4.set_xlabel('Min Nu')
    ax4.set_ylabel('Data Smoothness')
    
    # 5. 2D heatmap: Smoothness vs TC (average accuracy across other parameters)
    ax5 = fig.add_subplot(2, 3, 5)
    tc_smooth_acc = valid_results.groupby(['measured_smoothness', 'transition_contributions'])['accuracy'].mean().reset_index()
    tc_smooth_matrix = tc_smooth_acc.pivot(index='measured_smoothness', columns='transition_contributions', values='accuracy')
    sns.heatmap(tc_smooth_matrix, ax=ax5, cmap='coolwarm', cbar_kws={'label': 'Average Accuracy'})
    ax5.set_title('Smoothness vs Transition Contributions\n(avg accuracy across gamma & min_nu)')
    ax5.set_xlabel('Transition Contributions')
    ax5.set_ylabel('Data Smoothness')
    
    # 6. Parameter interaction plot: Show how optimal parameters change with smoothness
    ax6 = fig.add_subplot(2, 3, 6)
    optimal_by_smoothness = valid_results.loc[valid_results.groupby('measured_smoothness')['accuracy'].idxmax()]
    
    ax6_twin1 = ax6.twinx()
    ax6_twin2 = ax6.twinx()
    ax6_twin2.spines['right'].set_position(('outward', 60))
    
    line1 = ax6.plot(optimal_by_smoothness['measured_smoothness'], optimal_by_smoothness['gamma_threshold'], 
                     'o-', color='red', label='Optimal Gamma', linewidth=2)
    line2 = ax6_twin1.plot(optimal_by_smoothness['measured_smoothness'], optimal_by_smoothness['min_nu'], 
                          's-', color='blue', label='Optimal Min Nu', linewidth=2)
    line3 = ax6_twin2.plot(optimal_by_smoothness['measured_smoothness'], optimal_by_smoothness['transition_contributions'], 
                          '^-', color='green', label='Optimal TC', linewidth=2)
    
    ax6.set_xlabel('Data Smoothness')
    ax6.set_ylabel('Optimal Gamma Threshold', color='red')
    ax6_twin1.set_ylabel('Optimal Min Nu', color='blue')
    ax6_twin2.set_ylabel('Optimal Transition Contributions', color='green')
    ax6.set_title('Optimal Parameters vs Data Smoothness')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    save_path = os.path.join(config['savelocation_TET'], 
                            f'comprehensive_multi_parameter_validation_{n_primary_states_to_test}states.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive plots saved to: {save_path}")

def main():
    """
    Main function to run comprehensive multi-parameter validation.
    """
    print("=" * 70)
    print("COMPREHENSIVE MULTI-PARAMETER VALIDATION EXPERIMENT")
    print("=" * 70)
    
    # Load configuration
    config_path = os.path.join(project_root, 'Simulation.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Define parameter ranges (reduced for computational feasibility)
    smoothness_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Key values
    gamma_thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # Key values
    min_nu_values = [3, 6, 9, 12, 15, 20, 25, 30]  # Key values
    tc_values = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25]  # Key values
    n_primary_states_to_test = 2

    print(f"Testing {len(smoothness_values)} smoothness values: {smoothness_values}")
    print(f"Testing {len(gamma_thresholds)} gamma thresholds: {gamma_thresholds}")
    print(f"Testing {len(min_nu_values)} min_nu values: {min_nu_values}")
    print(f"Testing {len(tc_values)} transition contribution values: {tc_values}")
    
    # Calculate total combinations
    total_combinations = len(smoothness_values) * len(gamma_thresholds) * len(min_nu_values) * len(tc_values)
    print(f"Total parameter combinations: {total_combinations}")
    
    temp_data_dir = '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/temp_validation_data_comprehensive'
    os.makedirs(temp_data_dir, exist_ok=True)

    # --- Step 1: Generate all required data files in parallel and record measured smoothness ---
    print(f"\n{'='*50}")
    print(f"PRE-GENERATING {len(smoothness_values)} DATA FILES IN PARALLEL")
    print(f"{'='*50}\n")

    def generate_data_if_needed(smoothness):
        data_file_path = os.path.join(temp_data_dir, f'sim_data_smooth_{smoothness}.csv')
        if not os.path.exists(data_file_path):
            print(f"  Generating data for smoothness = {smoothness}")
            return generate_simulated_data(smoothness, data_file_path, n_primary_states=n_primary_states_to_test)
        else:
            print(f"  Data for smoothness = {smoothness} already exists. Calculating measured smoothness...")
            # Calculate measured smoothness for existing data
            csv_splitter_instance = csv_splitter(data_file_path)
            df = csv_splitter_instance.read_CSV()
            feature_names = ['MetaAwareness', 'Presence', 'PhysicalEffort','MentalEffort','Boredom','Receptivity',
                            'EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience']
            measured_smoothness = CustomHMMClustering.calculate_smoothness(df, feature_names)
            print(f"  Existing data measured smoothness: {measured_smoothness:.6f}")
            return data_file_path, measured_smoothness

    generation_results = Parallel(n_jobs=-1)(
        delayed(generate_data_if_needed)(s) for s in smoothness_values
    )
    
    # Create smoothness mapping from simulation parameter to measured smoothness
    smoothness_mapping = {}
    for i, smoothness in enumerate(smoothness_values):
        _, measured_smoothness = generation_results[i]
        smoothness_mapping[smoothness] = measured_smoothness
        print(f"Simulation smoothness {smoothness} -> Measured smoothness {measured_smoothness:.6f}")
    
    print(f"\nSmootness mapping complete:")
    for sim_smooth, measured_smooth in smoothness_mapping.items():
        print(f"  {sim_smooth:6.3f} -> {measured_smooth:.6f}")
    print()
    
    # Create a list of all parameter combinations
    param_combinations = list(itertools.product(smoothness_values, gamma_thresholds, min_nu_values, tc_values))
    print(f"Generated {len(param_combinations)} parameter combinations")

    # --- Step 2: Main experiment loop - PARALLELIZED ---
    print(f"\n{'='*50}")
    print(f"RUNNING {len(param_combinations)} VALIDATIONS IN PARALLEL")
    print(f"{'='*50}\n")

    # Use fewer parallel jobs for stability with large parameter space
    n_jobs = min(8, os.cpu_count())  # Limit to 8 cores or available cores
    print(f"Using {n_jobs} parallel jobs")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_validation)(
            smoothness, gamma, min_nu, tc, config, temp_data_dir, n_primary_states_to_test, smoothness_mapping
        ) for smoothness, gamma, min_nu, tc in param_combinations
    )

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE - PROCESSING RESULTS")
    print(f"{'='*60}")
    
    # Filter out potential None results if any error was not caught
    results = [r for r in results if r is not None]

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    valid_results = results_df.dropna(subset=['accuracy'])
    valid_acc = valid_results['accuracy'].values
    valid_nmi = valid_results['nmi'].values
    
    print(f"Valid results: {len(valid_acc)}/{len(results_df)}")
    if len(valid_acc) > 0:
        print(f"Accuracy range: {valid_acc.min():.4f} - {valid_acc.max():.4f} (mean: {valid_acc.mean():.4f})")
        print(f"NMI range: {valid_nmi.min():.4f} - {valid_nmi.max():.4f} (mean: {valid_nmi.mean():.4f})")
    
    # --- Find and print optimal parameter combinations for each smoothness ---
    if not valid_results.empty:
        optimal_indices = valid_results.loc[valid_results.groupby('measured_smoothness')['accuracy'].idxmax()]
        optimal_params = optimal_indices[['measured_smoothness', 'gamma_threshold', 'min_nu', 'transition_contributions', 'accuracy', 'nmi']].rename(
            columns={
                'measured_smoothness': 'smoothness',
                'gamma_threshold': 'Optimal_Gamma', 
                'min_nu': 'Optimal_Min_Nu',
                'transition_contributions': 'Optimal_TC',
                'accuracy': 'Max_Accuracy',
                'nmi': 'Corresponding_NMI'
            }
        ).sort_values('smoothness').reset_index(drop=True)
        
        print("\n" + "="*80)
        print("OPTIMAL PARAMETER COMBINATIONS FOR EACH SMOOTHNESS LEVEL")
        print("="*80)
        print(optimal_params.to_string(index=False))
        print("="*80 + "\n")
        
        # Create a summary of parameter trends
        print("PARAMETER TREND SUMMARY:")
        print("-" * 40)
        print(f"Gamma range across smoothness: {optimal_params['Optimal_Gamma'].min():.3f} - {optimal_params['Optimal_Gamma'].max():.3f}")
        print(f"Min Nu range across smoothness: {optimal_params['Optimal_Min_Nu'].min()} - {optimal_params['Optimal_Min_Nu'].max()}")
        print(f"TC range across smoothness: {optimal_params['Optimal_TC'].min()} - {optimal_params['Optimal_TC'].max()}")
        print(f"Accuracy range: {optimal_params['Max_Accuracy'].min():.4f} - {optimal_params['Max_Accuracy'].max():.4f}")
        
    else:
        optimal_params = pd.DataFrame()
        print("\nNo valid results to determine optimal parameters.\n")

    # --- Create comprehensive visualizations ---
    if not valid_results.empty:
        create_comprehensive_plots(valid_results, config, n_primary_states_to_test)
    
    # --- Save all results ---
    print("Saving comprehensive results...")
    
    # Save full results
    results_csv_path = os.path.join(config['savelocation_TET'], 
                                   f'comprehensive_multi_parameter_validation_data_{n_primary_states_to_test}states.csv')
    if os.path.exists(results_csv_path):
        existing_results = pd.read_csv(results_csv_path)
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Full results data saved to: {results_csv_path}")
    
    # Save optimal parameters lookup table
    optimal_csv_path = os.path.join(config['savelocation_TET'], 
                                    f'optimal_parameters_lookup_{n_primary_states_to_test}states.csv')
    if os.path.exists(optimal_csv_path):
        existing_optimal_params = pd.read_csv(optimal_csv_path)
        optimal_params = pd.concat([existing_optimal_params, optimal_params], ignore_index=True)
    optimal_params.to_csv(optimal_csv_path, index=False)
    print(f"Optimal parameters lookup table saved to: {optimal_csv_path}")
    
    # Create a simplified lookup function format
    lookup_dict = {}
    for _, row in optimal_params.iterrows():
        lookup_dict[row['smoothness']] = {
            'gamma_threshold': row['Optimal_Gamma'],
            'min_nu': int(row['Optimal_Min_Nu']),
            'transition_contribution': int(row['Optimal_TC']),
            'expected_accuracy': row['Max_Accuracy']
        }
    
    # Save as JSON for easy programmatic access
    import json
    json_path = os.path.join(config['savelocation_TET'], 
                            f'optimal_parameters_lookup_{n_primary_states_to_test}states.json')
    with open(json_path, 'w') as f:
        json.dump(lookup_dict, f, indent=2)
    print(f"Optimal parameters JSON lookup saved to: {json_path}")
    
    # Clean up temporary files
    import shutil
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)
        print("Cleaned up temporary data files")
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE MULTI-PARAMETER VALIDATION COMPLETE")
    print(f"{'='*70}")
    
    return results_df, optimal_params

if __name__ == '__main__':
    results_df, optimal_params = main()