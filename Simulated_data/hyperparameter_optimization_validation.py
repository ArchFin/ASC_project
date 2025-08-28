#!/usr/bin/env python3
"""
Hyperparameter Optimization Validation Script
=============================================
This script demonstrates the performance improvement achieved by using optimized
hyperparameters from the comprehensive validation experiment versus using default
parameters.

The script:
1. Loads the optimal hyperparameters from the comprehensive validation
2. Simulates test data across different smoothness levels
3. Runs HMM clustering with both optimized and default parameters
4. Compares and visualizes the accuracy improvements

This validates that the hyperparameter optimization actually improves performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import sys
import os
import yaml
from joblib import Parallel, delayed

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from HMM.HMM_methods import CustomHMMClustering, principal_component_finder, csv_splitter
from Simulated_data.TET_simulation import TETSimulator

def generate_test_data(smoothness_value, save_path, n_primary_states=2):
    """
    Generate test data for validation comparison.
    """
    print(f"Generating test data with smoothness = {smoothness_value}")
    
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

    # Generate test data structure - smaller for faster validation
    n_subjects = 5
    n_weeks = 2
    n_sessions = 3
    timepoints_per_session = 80
    subjects = [f"test{str(i+1).zfill(2)}" for i in range(n_subjects)]
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
    
    extra_cols = {
        'Subject': subject_col,
        'Week': week_col,
        'Session': session_col,
        'Condition': 'TestData'
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
    
    # Calculate measured smoothness
    measured_smoothness = CustomHMMClustering.calculate_smoothness(sim_data_full, feature_names)
    
    return save_path, measured_smoothness

def run_hmm_with_params(data_file_path, gamma_threshold, min_nu, tc_value, config, n_primary_states=2):
    """
    Run HMM clustering with specified parameters and return accuracy metrics.
    """
    # Load the data
    csv_splitter_instance = csv_splitter(data_file_path)
    df_csv_file_original = csv_splitter_instance.read_CSV()
    if df_csv_file_original is None:
        raise ValueError("CSV file could not be read.")
    
    feelings = config['feelings']
    pc_finder = principal_component_finder(df_csv_file_original, feelings,
                                           config['no_dimensions_PCA'],
                                           config['savelocation_TET'])
    principal_components, _, _ = pc_finder.PCA_TOT()

    # HMM Clustering
    clustering = CustomHMMClustering(
        data_file_path, config['savelocation_TET'],
        df_csv_file_original, feelings, principal_components,
        config['no_of_jumps'], tc_value
    )
    
    _, _, _, notransitions_df = clustering.run(
        num_base_states=n_primary_states, num_iterations=25, num_repetitions=1,
        gamma_threshold=gamma_threshold, min_nu=min_nu
    )

    # Validation
    sim_labels, _ = pd.factorize(notransitions_df['Cluster'])
    pred_labels = notransitions_df['labels'].astype(int).values

    # Hungarian alignment
    D = max(sim_labels.max(), pred_labels.max()) + 1
    cost_matrix = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = -np.sum((sim_labels == i) & (pred_labels == j))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {j: i for i, j in zip(row_ind, col_ind)}
    aligned_pred = np.array([mapping.get(lbl, lbl) for lbl in pred_labels])

    acc = accuracy_score(sim_labels, aligned_pred)
    nmi = normalized_mutual_info_score(sim_labels, aligned_pred)
    
    return acc, nmi

def compare_parameters_single(smoothness, optimal_params_df, config, temp_data_dir, n_primary_states_to_test):
    """
    Compare optimal vs default parameters for a single smoothness level.
    """
    print(f"Comparing parameters for smoothness = {smoothness}")
    
    # Generate test data
    data_file_path = os.path.join(temp_data_dir, f'test_data_smooth_{smoothness}.csv')
    _, measured_smoothness = generate_test_data(smoothness, data_file_path, n_primary_states=n_primary_states_to_test)
    
    # Find optimal parameters for this measured smoothness
    optimal_params = CustomHMMClustering.get_optimal_params_for_smoothness(
        measured_smoothness, optimal_params_df
    )
    
    optimal_gamma = optimal_params['gamma_threshold']
    optimal_min_nu = optimal_params['min_nu']
    optimal_tc = optimal_params['transition_contribution']
    
    # Default parameters
    default_gamma = 0.01
    default_min_nu = 12
    default_tc = 12
    
    try:
        # Run with optimal parameters
        opt_acc, opt_nmi = run_hmm_with_params(
            data_file_path, optimal_gamma, optimal_min_nu, optimal_tc, 
            config, n_primary_states_to_test
        )
        
        # Run with default parameters
        def_acc, def_nmi = run_hmm_with_params(
            data_file_path, default_gamma, default_min_nu, default_tc, 
            config, n_primary_states_to_test
        )
        
        result = {
            'sim_smoothness': smoothness,
            'measured_smoothness': measured_smoothness,
            'optimal_gamma': optimal_gamma,
            'optimal_min_nu': optimal_min_nu,
            'optimal_tc': optimal_tc,
            'default_gamma': default_gamma,
            'default_min_nu': default_min_nu,
            'default_tc': default_tc,
            'optimal_accuracy': opt_acc,
            'optimal_nmi': opt_nmi,
            'default_accuracy': def_acc,
            'default_nmi': def_nmi,
            'accuracy_improvement': opt_acc - def_acc,
            'nmi_improvement': opt_nmi - def_nmi
        }
        
        print(f"  Smoothness {smoothness} (measured: {measured_smoothness:.6f}):")
        print(f"    Optimal params: γ={optimal_gamma:.3f}, ν={optimal_min_nu}, tc={optimal_tc}")
        print(f"    Default params: γ={default_gamma:.3f}, ν={default_min_nu}, tc={default_tc}")
        print(f"    Accuracy: {opt_acc:.4f} (opt) vs {def_acc:.4f} (def) → Δ={opt_acc-def_acc:+.4f}")
        print(f"    NMI: {opt_nmi:.4f} (opt) vs {def_nmi:.4f} (def) → Δ={opt_nmi-def_nmi:+.4f}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR for smoothness {smoothness}: {e}")
        return None

def create_comparison_plots(comparison_results, config, n_primary_states_to_test):
    """
    Create comprehensive comparison plots showing the improvement from optimized parameters.
    """
    print("Creating comparison visualization plots...")
    
    df = pd.DataFrame(comparison_results)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Accuracy comparison bar plot
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['optimal_accuracy'], width, label='Optimized Parameters', 
                    color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df['default_accuracy'], width, label='Default Parameters', 
                    color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Test Cases (by Smoothness)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison: Optimized vs Default Parameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s:.3f}' for s in df['measured_smoothness']], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Accuracy improvement plot
    ax2 = fig.add_subplot(2, 3, 2)
    improvement_colors = ['green' if x > 0 else 'red' for x in df['accuracy_improvement']]
    bars = ax2.bar(x, df['accuracy_improvement'], color=improvement_colors, alpha=0.7)
    ax2.set_xlabel('Test Cases (by Smoothness)')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('Accuracy Improvement from Optimization')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s:.3f}' for s in df['measured_smoothness']], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height > 0 else height - 0.005,
                f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 3. NMI comparison
    ax3 = fig.add_subplot(2, 3, 3)
    bars1 = ax3.bar(x - width/2, df['optimal_nmi'], width, label='Optimized Parameters', 
                    color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, df['default_nmi'], width, label='Default Parameters', 
                    color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Test Cases (by Smoothness)')
    ax3.set_ylabel('NMI')
    ax3.set_title('NMI Comparison: Optimized vs Default Parameters')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{s:.3f}' for s in df['measured_smoothness']], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter values used
    ax4 = fig.add_subplot(2, 3, 4)
    ax4_twin1 = ax4.twinx()
    ax4_twin2 = ax4.twinx()
    ax4_twin2.spines['right'].set_position(('outward', 60))
    
    line1 = ax4.plot(x, df['optimal_gamma'], 'o-', color='red', label='Optimal Gamma', linewidth=2, markersize=6)
    line2 = ax4_twin1.plot(x, df['optimal_min_nu'], 's-', color='blue', label='Optimal Min Nu', linewidth=2, markersize=6)
    line3 = ax4_twin2.plot(x, df['optimal_tc'], '^-', color='green', label='Optimal TC', linewidth=2, markersize=6)
    
    ax4.set_xlabel('Test Cases (by Smoothness)')
    ax4.set_ylabel('Optimal Gamma', color='red')
    ax4_twin1.set_ylabel('Optimal Min Nu', color='blue')
    ax4_twin2.set_ylabel('Optimal TC', color='green')
    ax4.set_title('Optimal Parameters Used')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{s:.3f}' for s in df['measured_smoothness']], rotation=45)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    # 5. Smoothness vs improvement scatter
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(df['measured_smoothness'], df['accuracy_improvement'], 
                         s=80, alpha=0.7, c=df['accuracy_improvement'], 
                         cmap='RdYlGn', edgecolors='black')
    ax5.set_xlabel('Measured Data Smoothness')
    ax5.set_ylabel('Accuracy Improvement')
    ax5.set_title('Smoothness vs Accuracy Improvement')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Accuracy Improvement')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate summary statistics
    mean_improvement = df['accuracy_improvement'].mean()
    std_improvement = df['accuracy_improvement'].std()
    max_improvement = df['accuracy_improvement'].max()
    min_improvement = df['accuracy_improvement'].min()
    positive_improvements = (df['accuracy_improvement'] > 0).sum()
    total_cases = len(df)
    
    summary_text = f"""
    HYPERPARAMETER OPTIMIZATION SUMMARY
    ═══════════════════════════════════════
    
    Test Cases: {total_cases}
    
    Accuracy Improvement:
    • Mean: {mean_improvement:+.4f} ± {std_improvement:.4f}
    • Range: {min_improvement:+.4f} to {max_improvement:+.4f}
    • Cases Improved: {positive_improvements}/{total_cases} ({100*positive_improvements/total_cases:.1f}%)
    
    Average Parameters:
    • Optimal Gamma: {df['optimal_gamma'].mean():.3f}
    • Optimal Min Nu: {df['optimal_min_nu'].mean():.1f}
    • Optimal TC: {df['optimal_tc'].mean():.1f}
    
    Default Parameters:
    • Gamma: {df['default_gamma'].iloc[0]:.3f}
    • Min Nu: {df['default_min_nu'].iloc[0]}
    • TC: {df['default_tc'].iloc[0]}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comparison plot
    save_path = os.path.join(config['savelocation_TET'], 
                            f'hyperparameter_optimization_comparison_{n_primary_states_to_test}states.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {save_path}")

def main():
    """
    Main function to run the hyperparameter optimization validation.
    """
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION VALIDATION")
    print("=" * 80)
    
    # Load configuration
    config_path = os.path.join(project_root, 'Simulation.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    n_primary_states_to_test = 2
    
    # Check for optimal parameters file
    optimal_params_path = os.path.join(config['savelocation_TET'], 
                                      f'optimal_parameters_lookup_{n_primary_states_to_test}states.csv')
    
    if not os.path.exists(optimal_params_path):
        print(f"ERROR: Optimal parameters file not found at: {optimal_params_path}")
        print("Please run the comprehensive validation experiment first!")
        return None, None
    
    # Load optimal parameters
    optimal_params_df = pd.read_csv(optimal_params_path)
    print(f"Loaded optimal parameters for {len(optimal_params_df)} smoothness levels")
    
    # Define test smoothness values (subset of original range for faster validation)
    test_smoothness_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    
    print(f"Testing hyperparameter optimization on {len(test_smoothness_values)} smoothness levels:")
    print(f"Test smoothness values: {test_smoothness_values}")
    
    temp_data_dir = '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/temp_comparison_data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Run comparisons
    print(f"\n{'='*60}")
    print("RUNNING PARAMETER COMPARISONS")
    print(f"{'='*60}\n")
    
    comparison_results = []
    for smoothness in test_smoothness_values:
        result = compare_parameters_single(
            smoothness, optimal_params_path, config, temp_data_dir, n_primary_states_to_test
        )
        if result is not None:
            comparison_results.append(result)
    
    if not comparison_results:
        print("No valid comparison results obtained!")
        return None, None
    
    # Create comparison plots
    create_comparison_plots(comparison_results, config, n_primary_states_to_test)
    
    # Save comparison results
    results_df = pd.DataFrame(comparison_results)
    results_csv_path = os.path.join(config['savelocation_TET'], 
                                   f'hyperparameter_optimization_comparison_data_{n_primary_states_to_test}states.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Comparison results saved to: {results_csv_path}")
    
    # Print summary
    mean_improvement = results_df['accuracy_improvement'].mean()
    positive_improvements = (results_df['accuracy_improvement'] > 0).sum()
    total_cases = len(results_df)
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER OPTIMIZATION VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Average accuracy improvement: {mean_improvement:+.4f}")
    print(f"Cases with improvement: {positive_improvements}/{total_cases} ({100*positive_improvements/total_cases:.1f}%)")
    print(f"Results demonstrate {'SUCCESSFUL' if mean_improvement > 0 else 'UNSUCCESSFUL'} hyperparameter optimization")
    print(f"{'='*80}")
    
    # Clean up temporary files
    import shutil
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)
        print("Cleaned up temporary comparison data files")
    
    return results_df, comparison_results

if __name__ == '__main__':
    results_df, comparison_results = main()