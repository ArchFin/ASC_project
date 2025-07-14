#!/usr/bin/env python3
"""
Comprehensive HMM Validation Script

This script tests the HMM's ability to recover transition matrices from simulated data
with different configurations and provides detailed scientific validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from HMM_methods import csv_splitter, principal_component_finder, CustomHMMClustering, HMMModel

def test_hmm_recovery(config_file="Simulation.yaml", test_configs=None):
    """
    Test HMM transition matrix recovery with multiple configurations
    """
    
    # Load configuration
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    # Default test configurations if none provided
    if test_configs is None:
        test_configs = [
            {"num_iterations": 50, "num_repetitions": 5, "constraint_lim": 1.0, "contributions": 1.0},
            {"num_iterations": 100, "num_repetitions": 3, "constraint_lim": 0.8, "contributions": 1.0},
            {"num_iterations": 30, "num_repetitions": 10, "constraint_lim": 1.0, "contributions": 0.8},
        ]
    
    # Load data
    csv_splitter_instance = csv_splitter(config['filelocation_TET'])
    df_csv_file_original = csv_splitter_instance.read_CSV()
    if df_csv_file_original is None:
        raise ValueError("CSV file could not be read. Check the file path and try again.")

    # Extract principal components
    feelings = config['feelings']
    principal_component_finder_instance = principal_component_finder(
        df_csv_file_original, feelings, config['no_dimensions_PCA'], config['savelocation_TET']
    )
    principal_components, _, _ = principal_component_finder_instance.PCA_TOT()
    
    # True transition matrix from simulation
    true_transition_matrix = np.array([
        [0.90, 0.00, 0.10],  # state 0: mostly stays, can go to metastable
        [0.00, 0.90, 0.10],  # state 1: mostly stays, can go to metastable
        [0.15, 0.15, 0.70]   # metastable: likely to go to 0 or 1, rarely stays
    ])
    
    true_states = df_csv_file_original['Cluster'].values
    
    results = []
    
    print("="*80)
    print("COMPREHENSIVE HMM TRANSITION MATRIX RECOVERY VALIDATION")
    print("="*80)
    
    for i, test_config in enumerate(test_configs):
        print(f"\nTest Configuration {i+1}:")
        print(f"  Iterations: {test_config['num_iterations']}")
        print(f"  Repetitions: {test_config['num_repetitions']}")
        print(f"  Constraint Limit: {test_config['constraint_lim']}")
        print(f"  Transition Contributions: {test_config['contributions']}")
        print("-" * 60)
        
        # Create clustering instance
        clustering = CustomHMMClustering(
            config['filelocation_TET'], config['savelocation_TET'],
            df_csv_file_original, feelings, principal_components, 
            config['no_of_jumps'], test_config['contributions'],
            base_prior=config['base_prior'], 
            extra_prior=config['extra_prior'], 
            extra_self_prior=config['extra_self_prior'], 
            transition_temp=config['transition_temp'], 
            transition_constraint_lim=test_config['constraint_lim']
        )
        
        # Run clustering
        clustering.preprocess_data()
        clustering.perform_clustering(
            num_base_states=2, 
            num_iterations=test_config['num_iterations'], 
            num_repetitions=test_config['num_repetitions']
        )
        
        # Validate transition matrix
        if hasattr(clustering, 'avg_trans_prob'):
            print("Learned transition matrix:")
            print(np.round(clustering.avg_trans_prob, 3))
            
            # Compare full 3x3 matrix
            validation_full = clustering.validate_transition_matrix(
                clustering.avg_trans_prob, true_transition_matrix, align_states=True
            )
            
            # Compare only base states (2x2)
            learned_base = clustering.avg_trans_prob[:2, :2]
            learned_base = learned_base / learned_base.sum(axis=1, keepdims=True)
            true_base = true_transition_matrix[:2, :2]
            true_base = true_base / true_base.sum(axis=1, keepdims=True)
            
            validation_base = clustering.validate_transition_matrix(
                learned_base, true_base, align_states=True
            )
            
            # Validate state sequence
            seq_validation = clustering.validate_state_sequence(clustering.avg_state_seq, true_states)
            
            # Store results
            result = {
                'config': test_config,
                'validation_full': validation_full,
                'validation_base': validation_base,
                'seq_validation': seq_validation,
                'learned_matrix': clustering.avg_trans_prob.copy(),
                'log_likelihood': clustering.avg_log_lik
            }
            results.append(result)
            
            # Print metrics
            print(f"Full matrix - Frobenius error: {validation_full['frob_norm']:.4f}")
            print(f"Full matrix - Max error: {validation_full['max_err']:.4f}")
            print(f"Base states - Frobenius error: {validation_base['frob_norm']:.4f}")
            print(f"Base states - Max error: {validation_base['max_err']:.4f}")
            print(f"State sequence accuracy: {seq_validation['accuracy']:.4f}")
            print(f"State sequence NMI: {seq_validation['nmi']:.4f}")
            print(f"Log likelihood: {clustering.avg_log_lik:.3f}")
        
        else:
            print("ERROR: avg_trans_prob not found!")
    
    # Create summary plot
    if results:
        create_validation_summary_plot(results, config['savelocation_TET'])
        
    return results

def create_validation_summary_plot(results, save_dir):
    """Create a comprehensive validation summary plot"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract metrics
    base_frob_errors = [r['validation_base']['frob_norm'] for r in results]
    base_max_errors = [r['validation_base']['max_err'] for r in results]
    full_frob_errors = [r['validation_full']['frob_norm'] for r in results]
    accuracies = [r['seq_validation']['accuracy'] for r in results]
    nmis = [r['seq_validation']['nmi'] for r in results]
    log_liks = [r['log_likelihood'] for r in results]
    
    config_labels = [f"Config {i+1}" for i in range(len(results))]
    
    # Plot metrics
    axes[0, 0].bar(config_labels, base_frob_errors, color='skyblue')
    axes[0, 0].set_title('Base States Frobenius Error')
    axes[0, 0].set_ylabel('Error')
    
    axes[0, 1].bar(config_labels, base_max_errors, color='lightcoral')
    axes[0, 1].set_title('Base States Max Error')
    axes[0, 1].set_ylabel('Error')
    
    axes[0, 2].bar(config_labels, full_frob_errors, color='lightgreen')
    axes[0, 2].set_title('Full Matrix Frobenius Error')
    axes[0, 2].set_ylabel('Error')
    
    axes[1, 0].bar(config_labels, accuracies, color='gold')
    axes[1, 0].set_title('State Sequence Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].bar(config_labels, nmis, color='plum')
    axes[1, 1].set_title('Normalized Mutual Information')
    axes[1, 1].set_ylabel('NMI')
    axes[1, 1].set_ylim([0, 1])
    
    axes[1, 2].bar(config_labels, log_liks, color='orange')
    axes[1, 2].set_title('Log Likelihood')
    axes[1, 2].set_ylabel('Log Likelihood')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_validation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmaps of best and worst performing matrices
    best_idx = np.argmin(base_frob_errors)
    worst_idx = np.argmax(base_frob_errors)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Best performing
    best_learned = results[best_idx]['validation_base']['learned_aligned']
    best_true = results[best_idx]['validation_base']['true_aligned']
    
    sns.heatmap(best_learned, annot=True, fmt='.3f', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title(f'Best Learned Matrix (Config {best_idx+1})')
    
    sns.heatmap(best_true, annot=True, fmt='.3f', ax=axes[0, 1], cmap='Blues')
    axes[0, 1].set_title('True Matrix')
    
    # Worst performing  
    worst_learned = results[worst_idx]['validation_base']['learned_aligned']
    worst_true = results[worst_idx]['validation_base']['true_aligned']
    
    sns.heatmap(worst_learned, annot=True, fmt='.3f', ax=axes[1, 0], cmap='Blues')
    axes[1, 0].set_title(f'Worst Learned Matrix (Config {worst_idx+1})')
    
    sns.heatmap(worst_true, annot=True, fmt='.3f', ax=axes[1, 1], cmap='Blues')
    axes[1, 1].set_title('True Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_worst_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nValidation plots saved to: {save_dir}")

if __name__ == "__main__":
    # Run comprehensive validation
    results = test_hmm_recovery()
    
    # Find best configuration
    base_errors = [r['validation_base']['frob_norm'] for r in results]
    best_config_idx = np.argmin(base_errors)
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Best performing configuration: Config {best_config_idx + 1}")
    print(f"Best base states Frobenius error: {base_errors[best_config_idx]:.4f}")
    print(f"Best state sequence accuracy: {results[best_config_idx]['seq_validation']['accuracy']:.4f}")
    
    # Recommendations
    print("\nRECOMMENDations:")
    if base_errors[best_config_idx] < 0.1:
        print("✓ EXCELLENT: HMM successfully recovers transition matrix")
    elif base_errors[best_config_idx] < 0.3:
        print("? MODERATE: HMM partially recovers transition matrix")
    else:
        print("✗ POOR: HMM fails to recover transition matrix")
        print("  - Try increasing number of iterations")
        print("  - Reduce transition constraints")
        print("  - Check data quality and simulation parameters")