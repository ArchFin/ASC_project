#!/usr/bin/env python3
"""
rnn_stable_states_validation.py

Validation script to test RNN model discrimination between stable breathing states only.
This script filters the data to include only cluster 1 and cluster 2 (stable states),
excluding transition states, to validate the model's ability to distinguish between
the two stable breathing patterns.

Usage:
    python rnn_stable_states_validation.py [--sequence-length 25] [--test-subject s01]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# TensorFlow and Keras
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Input, Dropout, LSTM, Bidirectional, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.utils import to_categorical

# Scikit-learn
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import seaborn as sns
import warnings

# Import functions from the main RNN script
sys.path.append(os.path.dirname(__file__))
try:
    from rnn_decoder_simple import (
        reshape_for_rnn, select_important_features, clean_impute, 
        preprocess_for_rnn, build_rnn_model, train_and_evaluate_rnn
    )
except ImportError:
    print("Warning: Could not import from rnn_decoder_simple.py. Using local implementations.")
    
    def reshape_for_rnn(X, sequence_length=25, overlap=0.8):
        """Create temporal sequences for RNN input with enhanced overlap for EEG."""
        n_samples, n_features = X.shape
        step_size = max(1, int(sequence_length * (1 - overlap)))
        n_sequences = max(0, (n_samples - sequence_length) // step_size + 1)
        
        if n_sequences == 0:
            # Fallback for small datasets
            if n_samples < sequence_length:
                padding_needed = sequence_length - n_samples
                X_padded = np.vstack([X, np.tile(X[-1:], (padding_needed, 1))])
                X_sequences = X_padded.reshape(1, sequence_length, n_features)
                indices = list(range(n_samples)) + [n_samples-1] * padding_needed
            else:
                X_sequences = X[:sequence_length].reshape(1, sequence_length, n_features)
                indices = list(range(sequence_length))
            return X_sequences, [indices]
        
        X_sequences = np.zeros((n_sequences, sequence_length, n_features))
        indices = []
        
        for i in range(n_sequences):
            start_idx = i * step_size
            end_idx = start_idx + sequence_length
            X_sequences[i] = X[start_idx:end_idx]
            indices.append(list(range(start_idx, end_idx)))
        
        return X_sequences, indices

    def select_important_features(X_train, y_train, X_test, n_features=21):
        """Enhanced feature selection with better breathing-specific features."""
        le_temp = LabelEncoder()
        y_encoded = le_temp.fit_transform(y_train)
        
        # Add temporal derivative features for better RNN performance
        X_train_enhanced = np.copy(X_train)
        X_test_enhanced = np.copy(X_test)
        
        # Add first-order differences (temporal derivatives)
        if X_train.shape[0] > 1:
            train_diff = np.diff(X_train, axis=0, prepend=X_train[:1])
            test_diff = np.diff(X_test, axis=0, prepend=X_test[:1])
            X_train_enhanced = np.hstack([X_train, train_diff])
            X_test_enhanced = np.hstack([X_test, test_diff])
            
            # Update y_encoded to match enhanced features
            if len(y_encoded) != X_train_enhanced.shape[0]:
                y_encoded = y_encoded[:X_train_enhanced.shape[0]]
        
        # Univariate selection
        selector_f = SelectKBest(score_func=f_classif, k=min(n_features, X_train_enhanced.shape[1]))
        selector_f.fit(X_train_enhanced, y_encoded)
        f_scores = selector_f.scores_
        f_selected = np.argsort(f_scores)[-n_features:]
        
        # Mutual information
        mi_scores = mutual_info_classif(X_train_enhanced, y_encoded, random_state=42)
        mi_selected = np.argsort(mi_scores)[-n_features:]
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=12)
        rf.fit(X_train_enhanced, y_encoded)
        rf_importances = rf.feature_importances_
        rf_selected = np.argsort(rf_importances)[-n_features:]
        
        # Combine scores with better weighting for temporal features
        combined_scores = np.zeros(X_train_enhanced.shape[1])
        f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
        rf_scores_norm = (rf_importances - rf_importances.min()) / (rf_importances.max() - rf_importances.min() + 1e-8)
        
        # Weight temporal derivatives higher for RNN
        combined_scores = 0.3 * f_scores_norm + 0.4 * mi_scores_norm + 0.3 * rf_scores_norm
        
        selected_features = np.argsort(combined_scores)[-n_features:]
        return X_train_enhanced[:, selected_features], X_test_enhanced[:, selected_features], selected_features, {
            'f_scores': f_scores,
            'mi_scores': mi_scores,
            'rf_importances': rf_importances,
            'combined_scores': combined_scores,
            'selected_indices': selected_features,
            'feature_names': [f"feature_{i}" for i in range(X_train_enhanced.shape[1])]
        }

    def clean_impute(X, mu=None, sigma=None, threshold=2.5):
        """Clean outliers and impute missing values."""
        mask_allnan = X.isna().all(axis=1)
        X_clean = X[~mask_allnan]
        
        if mu is None:
            mu = X_clean.mean()
        if sigma is None:
            sigma = X_clean.std()

        X_clean = X_clean.where(np.abs(X_clean - mu) <= threshold * sigma)
        X_clean = X_clean.fillna(method='ffill').fillna(method='bfill').dropna()
        return X_clean, mu, sigma

    def preprocess_for_rnn(X_train, X_test, y_train, y_test, sequence_length=10):
        """Complete preprocessing pipeline for RNN."""
        # Clean and impute
        X_train_clean, mu, sigma = clean_impute(X_train)
        y_train_clean = y_train.loc[X_train_clean.index]
        
        X_test_clean, _, _ = clean_impute(X_test, mu=mu, sigma=sigma)
        if X_test_clean.empty:
            return None
        y_test_clean = y_test.loc[X_test_clean.index]
        
        # Ensure indices are aligned and reset
        X_train_clean = X_train_clean.reset_index(drop=True)
        y_train_clean = y_train_clean.reset_index(drop=True)
        X_test_clean = X_test_clean.reset_index(drop=True)
        y_test_clean = y_test_clean.reset_index(drop=True)
        
        # Shuffle training data
        X_train_shuffled, y_train_shuffled = shuffle(X_train_clean, y_train_clean, random_state=42)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_shuffled)
        X_test_scaled = scaler.transform(X_test_clean)
        
        # Feature selection with enhanced features
        X_train_selected, X_test_selected, selected_features, feature_info = select_important_features(
            X_train_scaled, y_train_shuffled, X_test_scaled, n_features=21
        )
        
        # Label encoding
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_shuffled)
        y_test_enc = le.transform(y_test_clean)
        
        # Reshape for RNN
        X_train_sequences, train_indices = reshape_for_rnn(X_train_selected, sequence_length=sequence_length)
        X_test_sequences, test_indices = reshape_for_rnn(X_test_selected, sequence_length=sequence_length)
        
        # Adjust labels for sequences (use last sample label in each sequence)
        y_train_sequences = []
        for seq_indices in train_indices:
            last_idx = seq_indices[-1] if isinstance(seq_indices, list) else seq_indices
            label_idx = min(last_idx, len(y_train_enc) - 1)
            y_train_sequences.append(y_train_enc[label_idx])
        
        y_test_sequences = []
        for seq_indices in test_indices:
            last_idx = seq_indices[-1] if isinstance(seq_indices, list) else seq_indices
            label_idx = min(last_idx, len(y_test_enc) - 1)
            y_test_sequences.append(y_test_enc[label_idx])
        
        y_train_sequences = np.array(y_train_sequences)
        y_test_sequences = np.array(y_test_sequences)
        
        # Convert to one-hot
        Y_train_onehot = to_categorical(y_train_sequences)
        Y_test_onehot = to_categorical(y_test_sequences, num_classes=len(le.classes_))
        
        return X_train_sequences, X_test_sequences, Y_train_onehot, Y_test_onehot, le, le.classes_, feature_info

    def build_rnn_model(input_shape, num_classes, model_type='bidirectional'):
        """Build enhanced RNN model."""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                              kernel_regularizer=l2(0.01))),
            LayerNormalization(),
            Bidirectional(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2,
                              kernel_regularizer=l2(0.01))),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.4),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))
        ])
        
        optimizer = Adam(learning_rate=8e-4, clipnorm=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )
        
        return model

    def train_and_evaluate_rnn(X_train, Y_train, X_test, Y_test, model_type='bidirectional'):
        """Train and evaluate RNN model."""
        # Compute class weights
        y_train_labels = Y_train.argmax(axis=1)
        class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_rnn_model(input_shape, Y_train.shape[1], model_type)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7)
        ]
        
        history = model.fit(
            X_train, Y_train,
            epochs=40,
            batch_size=32,
            validation_data=(X_test, Y_test),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate
        probs = model.predict(X_test, verbose=1)
        preds = probs.argmax(axis=1)
        true = Y_test.argmax(axis=1)
        
        # Comprehensive metrics
        n_classes = Y_test.shape[1]
        cm = confusion_matrix(true, preds, labels=list(range(n_classes)))
        
        # Calculate additional metrics
        balanced_acc = balanced_accuracy_score(true, preds)
        f1_macro = f1_score(true, preds, average='macro')
        f1_weighted = f1_score(true, preds, average='weighted')
        
        # AUC per class
        aucs = {}
        for i in range(Y_test.shape[1]):
            if len(np.unique(true)) > 1 and len(np.unique(true == i)) > 1:
                try:
                    aucs[str(i)] = roc_auc_score((true == i).astype(int), probs[:, i])
                except ValueError:
                    aucs[str(i)] = 0.5
            else:
                aucs[str(i)] = 0.5
        
        metrics = {
            'aucs': aucs,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'cm': cm,
            'predictions': preds,
            'probabilities': probs,
            'true_labels': true
        }
        
        return model, metrics, history


def analyze_stable_states_data(df, exclude_subjects=None):
    """Analyze the stable states data distribution."""
    print("===== STABLE STATES DATA ANALYSIS =====")
    
    # Filter to stable states only (clusters 1 and 2)
    stable_df = df[df['transition_label'].isin([1, 2])].copy()
    
    if exclude_subjects:
        stable_df = stable_df[~stable_df['subject'].isin(exclude_subjects)]
    
    print(f"Original dataset: {len(df)} samples")
    print(f"Stable states only: {len(stable_df)} samples ({len(stable_df)/len(df)*100:.1f}%)")
    
    # Class distribution
    class_dist = stable_df['transition_label'].value_counts().sort_index()
    print(f"\nStable states distribution:")
    for label, count in class_dist.items():
        print(f"  Cluster {label}: {count} samples ({count/len(stable_df)*100:.1f}%)")
    
    # Subject distribution
    subject_dist = stable_df.groupby(['subject', 'transition_label']).size().unstack(fill_value=0)
    print(f"\nPer-subject stable states distribution:")
    print(subject_dist)
    
    # Check for subjects with only one class
    subjects_single_class = []
    for subject in subject_dist.index:
        non_zero_classes = (subject_dist.loc[subject] > 0).sum()
        if non_zero_classes < 2:
            subjects_single_class.append(subject)
    
    if subjects_single_class:
        print(f"\n⚠️  Subjects with only one stable state class: {subjects_single_class}")
        print("These subjects will be excluded from binary classification.")
    
    return stable_df, subjects_single_class


def process_stable_states_fold(fold_data, sequence_length=25):
    """Process a single fold for stable states validation."""
    fold_num, train_idx, test_idx, X, y, df = fold_data
    
    # Split data - ensure indices are aligned
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    # Check if both classes are present in train and test
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    
    if len(train_classes) < 2:
        print(f"    ⚠️  Training set only has {len(train_classes)} class(es): {train_classes}")
        return fold_num, None, None, None
    
    if len(test_classes) < 2:
        print(f"    ⚠️  Test set only has {len(test_classes)} class(es): {test_classes}")
        return fold_num, None, None, None
    
    # Preprocess
    result = preprocess_for_rnn(X_train, X_test, y_train, y_test, sequence_length=sequence_length)
    if result is None:
        return fold_num, None, None, None
    
    X_train_seq, X_test_seq, Y_train_oh, Y_test_oh, le, classes, feature_info = result
    
    # Train and evaluate
    try:
        model, metrics, history = train_and_evaluate_rnn(
            X_train_seq, Y_train_oh, X_test_seq, Y_test_oh, 'bidirectional'
        )
        
        fold_auc = np.mean(list(metrics['aucs'].values()))
        print(f"    Binary AUC: {fold_auc:.3f}, Balanced Acc: {metrics['balanced_accuracy']:.3f}")
        
        return fold_num, metrics, feature_info, history
        
    except Exception as e:
        print(f"    Fold failed: {e}")
        return fold_num, None, None, None


def create_stable_states_visualizations(all_metrics, results_dir):
    """Create visualizations specific to stable states validation."""
    print("\n📊 Creating stable states visualizations...")
    
    # 1. Binary classification performance plot
    plt.figure(figsize=(15, 10))
    
    # AUC scores across folds
    plt.subplot(2, 3, 1)
    fold_aucs = []
    for metrics in all_metrics:
        if metrics and 'aucs' in metrics:
            fold_auc = np.mean(list(metrics['aucs'].values()))
            fold_aucs.append(fold_auc)
    
    if fold_aucs:
        plt.plot(range(1, len(fold_aucs)+1), fold_aucs, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance Level')
        plt.axhline(y=np.mean(fold_aucs), color='green', linestyle='-', alpha=0.7, 
                   label=f'Mean: {np.mean(fold_aucs):.3f}')
        plt.xlabel('Fold Number')
        plt.ylabel('Binary AUC')
        plt.title('Stable States Binary Classification AUC')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 1)
    
    # 2. Balanced accuracy across folds
    plt.subplot(2, 3, 2)
    balanced_accs = [m['balanced_accuracy'] for m in all_metrics if m]
    if balanced_accs:
        plt.plot(range(1, len(balanced_accs)+1), balanced_accs, 'go-', linewidth=2, markersize=8)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance Level')
        plt.axhline(y=np.mean(balanced_accs), color='orange', linestyle='-', alpha=0.7,
                   label=f'Mean: {np.mean(balanced_accs):.3f}')
        plt.xlabel('Fold Number')
        plt.ylabel('Balanced Accuracy')
        plt.title('Stable States Balanced Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 1)
    
    # 3. F1 scores
    plt.subplot(2, 3, 3)
    f1_scores = [m['f1_macro'] for m in all_metrics if m]
    if f1_scores:
        plt.plot(range(1, len(f1_scores)+1), f1_scores, 'ro-', linewidth=2, markersize=8)
        plt.axhline(y=np.mean(f1_scores), color='purple', linestyle='-', alpha=0.7,
                   label=f'Mean: {np.mean(f1_scores):.3f}')
        plt.xlabel('Fold Number')
        plt.ylabel('F1-Score (Macro)')
        plt.title('Stable States F1-Score')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 1)
    
    # 4. Overall confusion matrix
    plt.subplot(2, 3, 4)
    if all_metrics:
        total_cm = np.sum([m['cm'] for m in all_metrics if m], axis=0)
        total_cm_percent = total_cm.astype('float') / total_cm.sum(axis=1, keepdims=True) * 100
        
        im = plt.imshow(total_cm_percent, cmap='Blues', vmin=0, vmax=100)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        class_names = ['Cluster 1', 'Cluster 2']
        plt.xticks([0, 1], class_names)
        plt.yticks([0, 1], class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Stable States Confusion Matrix (%)')
        
        # Add percentage labels
        for i in range(2):
            for j in range(2):
                color = 'white' if total_cm_percent[i, j] > 50 else 'black'
                plt.text(j, i, f"{total_cm_percent[i, j]:.1f}%", 
                        ha='center', va='center', color=color, fontweight='bold')
    
    # 5. Performance distribution
    plt.subplot(2, 3, 5)
    if fold_aucs and balanced_accs and f1_scores:
        metrics_data = [fold_aucs, balanced_accs, f1_scores]
        metrics_labels = ['AUC', 'Balanced Acc', 'F1-Score']
        
        box_plot = plt.boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel('Score')
        plt.title('Performance Metrics Distribution')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1)
    
    # 6. Statistical significance
    plt.subplot(2, 3, 6)
    if fold_aucs:
        # One-sample t-test against chance level
        t_stat, p_value = stats.ttest_1samp(fold_aucs, 0.5)
        
        # Create a text box with results
        stats_text = f"""Statistical Significance Test
        
H₀: AUC = 0.5 (chance level)
H₁: AUC > 0.5

Mean AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}
t-statistic: {t_stat:.3f}
p-value: {p_value:.4f}

95% CI: [{stats.t.interval(0.95, len(fold_aucs)-1, loc=np.mean(fold_aucs), scale=stats.sem(fold_aucs))[0]:.3f}, 
         {stats.t.interval(0.95, len(fold_aucs)-1, loc=np.mean(fold_aucs), scale=stats.sem(fold_aucs))[1]:.3f}]

Result: {'Significant' if p_value < 0.05 else 'Not Significant'}
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        plt.title('Statistical Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'stable_states_validation_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed performance comparison
    plt.figure(figsize=(12, 8))
    
    # Create comparison with chance level
    plt.subplot(2, 2, 1)
    if fold_aucs:
        x = np.arange(len(fold_aucs))
        plt.bar(x, fold_aucs, alpha=0.7, color='steelblue', label='Actual AUC')
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance Level')
        plt.xlabel('Fold Number')
        plt.ylabel('AUC')
        plt.title('Binary AUC vs Chance Level')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(fold_aucs):
            color = 'green' if v > 0.6 else 'orange' if v > 0.55 else 'red'
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', color=color)
    
    # Effect size analysis
    plt.subplot(2, 2, 2)
    if fold_aucs:
        effect_sizes = [(auc - 0.5) / 0.5 for auc in fold_aucs]  # Relative improvement over chance
        plt.bar(range(len(effect_sizes)), effect_sizes, alpha=0.7, color='orange')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Fold Number')
        plt.ylabel('Effect Size (Relative to Chance)')
        plt.title('Effect Size Analysis')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(effect_sizes):
            plt.text(i, v + 0.01 if v > 0 else v - 0.01, f'{v*100:.1f}%', 
                    ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')
    
    # Learning curves (if available)
    plt.subplot(2, 2, 3)
    plt.text(0.5, 0.5, 'Learning curves would be\nplotted here if training\nhistory is available', 
            ha='center', va='center', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.title('Training Dynamics')
    plt.axis('off')
    
    # Performance summary
    plt.subplot(2, 2, 4)
    if fold_aucs and balanced_accs:
        summary_text = f"""STABLE STATES VALIDATION SUMMARY

Binary Classification Results:
• Number of folds: {len(fold_aucs)}
• Mean AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}
• Mean Balanced Acc: {np.mean(balanced_accs):.3f} ± {np.std(balanced_accs):.3f}
• Best fold AUC: {max(fold_aucs):.3f}
• Worst fold AUC: {min(fold_aucs):.3f}

Performance Assessment:
{'✅ EXCELLENT' if np.mean(fold_aucs) > 0.80 else 
 '✅ GOOD' if np.mean(fold_aucs) > 0.70 else
 '⚠️ MODERATE' if np.mean(fold_aucs) > 0.60 else
 '🔍 WEAK' if np.mean(fold_aucs) > 0.55 else
 '❌ POOR'} discrimination between stable states

Consistency: {'High' if np.std(fold_aucs) < 0.05 else 'Moderate' if np.std(fold_aucs) < 0.10 else 'Low'}
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        plt.axis('off')
        plt.title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'stable_states_detailed_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main validation pipeline for stable breathing states."""
    parser = argparse.ArgumentParser(description='Validate RNN discrimination of stable breathing states')
    parser.add_argument('--sequence-length', type=int, default=25, help='Sequence length for RNN')
    parser.add_argument('--test-subject', type=str, help='Test specific subject only')
    parser.add_argument('--exclude-transitions', action='store_true', default=True, 
                       help='Exclude transition states (keep only clusters 1 and 2)')
    args = parser.parse_args()
    
    print("===== STABLE BREATHING STATES VALIDATION =====")
    print(f"Sequence length: {args.sequence_length}")
    
    # Load data
    csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete_2.csv'
    print(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Load exclusion list from main results
    exclude_subjects = []#'s03', 's19', 's17', 's10', 's02', 's21'
    results_json_path = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/Neural Decoder/rnn_results/rnn_results_summary.json'
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as f:
            try:
                prev_results = json.load(f)
                flagged_subjects = prev_results.get('excluded_subjects', [])
                exclude_subjects.extend(flagged_subjects)
                exclude_subjects = list(set(exclude_subjects))
            except Exception as e:
                print(f"Warning: Could not load excluded subjects: {e}")
    
    # Analyze stable states data
    stable_df, single_class_subjects = analyze_stable_states_data(df, exclude_subjects)
    
    # Add single-class subjects to exclusion list
    exclude_subjects.extend(single_class_subjects)
    exclude_subjects = list(set(exclude_subjects))
    
    # Filter dataset
    if exclude_subjects:
        print(f"\nExcluding subjects: {exclude_subjects}")
        stable_df = stable_df[~stable_df['subject'].isin(exclude_subjects)]
    
    # Test specific subject if requested
    if args.test_subject:
        if args.test_subject in stable_df['subject'].values:
            stable_df = stable_df[stable_df['subject'] == args.test_subject]
            print(f"\nTesting single subject: {args.test_subject}")
        else:
            print(f"Subject {args.test_subject} not found in data. Available subjects: {sorted(stable_df['subject'].unique())}")
            return
    
    # Prepare features
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number', 'transition_label']
    X_raw = stable_df.drop(columns=drop_cols, errors='ignore')
    glob_cols = [col for col in X_raw.columns if 'glob_chans' in col]
    X = X_raw[glob_cols]
    y = stable_df['transition_label']
    groups = stable_df['subject']
    
    print(f"\nFinal dataset for validation:")
    print(f"  Samples: {len(stable_df)}")
    print(f"  Subjects: {len(stable_df['subject'].unique())}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {y.value_counts().to_dict()}")
    
    # Remap labels to 0 and 1 for binary classification
    label_mapping = {1: 0, 2: 1}  # Cluster 1 -> 0, Cluster 2 -> 1
    y_binary = y.map(label_mapping)
    
    print(f"\nBinary mapping: Cluster 1 -> 0, Cluster 2 -> 1")
    print(f"Binary distribution: {y_binary.value_counts().to_dict()}")
    
    # Cross-validation setup
    subjects = stable_df['subject'].unique()
    n_splits = len(subjects)
    
    if args.test_subject:
        print("Single subject test - no cross-validation")
        return
    
    gkf = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(X, y_binary, groups=groups))
    
    print(f"\nCross-validation: {n_splits} folds (leave-one-subject-out)")
    
    # Run validation
    all_metrics = []
    all_feature_info = []
    failed_folds = []
    
    for fold_num, (train_idx, test_idx) in enumerate(cv_splits, 1):
        # Convert indices to actual DataFrame indices
        train_indices = X.index[train_idx]
        test_indices = X.index[test_idx]
        
        fold_data = (fold_num, train_indices, test_indices, X, y_binary, stable_df)
        test_subject = stable_df.loc[test_indices]['subject'].iloc[0]
        
        print(f"\nFold {fold_num}/{n_splits} - Test subject: {test_subject}")
        
        fold_result, metrics, feature_info, history = process_stable_states_fold(
            fold_data, sequence_length=args.sequence_length
        )
        
        if metrics is not None:
            all_metrics.append(metrics)
            all_feature_info.append(feature_info)
            fold_auc = np.mean(list(metrics['aucs'].values()))
            print(f"  ✅ Success: AUC={fold_auc:.3f}, Balanced Acc={metrics['balanced_accuracy']:.3f}")
        else:
            failed_folds.append(fold_num)
            print(f"  ❌ Failed")
    
    # Results analysis
    print(f"\n===== STABLE STATES VALIDATION RESULTS =====")
    
    if all_metrics:
        # Calculate statistics
        fold_aucs = [np.mean(list(m['aucs'].values())) for m in all_metrics]
        balanced_accs = [m['balanced_accuracy'] for m in all_metrics]
        f1_scores = [m['f1_macro'] for m in all_metrics]
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_balanced_acc = np.mean(balanced_accs)
        std_balanced_acc = np.std(balanced_accs)
        
        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(fold_aucs, 0.5)
        confidence_interval = stats.t.interval(0.95, len(fold_aucs)-1, 
                                             loc=mean_auc, scale=stats.sem(fold_aucs))
        
        print(f"\n🎯 BINARY CLASSIFICATION PERFORMANCE:")
        print(f"   Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"   95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        print(f"   p-value vs chance: {p_value:.4f}")
        print(f"   Mean Balanced Accuracy: {mean_balanced_acc:.3f} ± {std_balanced_acc:.3f}")
        print(f"   Mean F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        # Performance assessment
        print(f"\n📊 ASSESSMENT:")
        if mean_auc > 0.80:
            print("   ✅ EXCELLENT: Very strong discrimination between stable states!")
        elif mean_auc > 0.70:
            print("   ✅ GOOD: Strong discrimination between stable states")
        elif mean_auc > 0.60:
            print("   ⚠️  MODERATE: Some discrimination ability")
        elif mean_auc > 0.55:
            print("   🔍 WEAK: Limited discrimination ability")
        else:
            print("   ❌ POOR: No meaningful discrimination")
        
        if p_value < 0.05:
            print(f"   📈 SIGNIFICANT: Performance significantly above chance (p={p_value:.4f})")
        else:
            print(f"   📊 NOT SIGNIFICANT: Performance not significantly above chance (p={p_value:.4f})")
        
        # Consistency assessment
        cv = std_auc / mean_auc if mean_auc > 0 else float('inf')
        if cv < 0.10:
            print("   🎯 HIGH CONSISTENCY: Low variance across subjects")
        elif cv < 0.20:
            print("   📊 MODERATE CONSISTENCY: Some variance across subjects")
        else:
            print("   ⚠️  LOW CONSISTENCY: High variance across subjects")
        
        # Create results directory
        results_dir = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/Neural Decoder/stable_states_validation'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results
        validation_results = {
            'validation_type': 'stable_states_binary_classification',
            'dataset_info': {
                'total_samples': len(stable_df),
                'cluster_1_samples': len(stable_df[stable_df['transition_label'] == 1]),
                'cluster_2_samples': len(stable_df[stable_df['transition_label'] == 2]),
                'subjects_tested': len(subjects),
                'excluded_subjects': exclude_subjects,
                'sequence_length': args.sequence_length
            },
            'performance_metrics': {
                'mean_auc': float(mean_auc),
                'std_auc': float(std_auc),
                'auc_confidence_interval': [float(confidence_interval[0]), float(confidence_interval[1])],
                'mean_balanced_accuracy': float(mean_balanced_acc),
                'std_balanced_accuracy': float(std_balanced_acc),
                'mean_f1_score': float(np.mean(f1_scores)),
                'std_f1_score': float(np.std(f1_scores)),
                'fold_aucs': [float(auc) for auc in fold_aucs],
                'fold_balanced_accs': [float(acc) for acc in balanced_accs],
                'fold_f1_scores': [float(f1) for f1 in f1_scores]
            },
            'statistical_analysis': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_above_chance': bool(p_value < 0.05),
                'coefficient_of_variation': float(cv)
            },
            'assessment': {
                'performance_level': ('EXCELLENT' if mean_auc > 0.80 else 
                                    'GOOD' if mean_auc > 0.70 else
                                    'MODERATE' if mean_auc > 0.60 else
                                    'WEAK' if mean_auc > 0.55 else 'POOR'),
                'consistency_level': ('HIGH' if cv < 0.10 else 'MODERATE' if cv < 0.20 else 'LOW'),
                'validates_main_results': bool(mean_auc > 0.60 and p_value < 0.05)
            },
            'fold_details': {
                'successful_folds': len(all_metrics),
                'failed_folds': failed_folds,
                'success_rate': len(all_metrics) / n_splits
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(results_dir, 'stable_states_validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Create visualizations
        create_stable_states_visualizations(all_metrics, results_dir)
        
        # Confusion matrix
        total_cm = np.sum([m['cm'] for m in all_metrics], axis=0)
        total_cm_percent = total_cm.astype('float') / total_cm.sum(axis=1, keepdims=True) * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(total_cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=['Cluster 1', 'Cluster 2'],
                   yticklabels=['Cluster 1', 'Cluster 2'],
                   cbar_kws={'label': 'Percentage (%)'})
        plt.title(f'Stable States Validation - Confusion Matrix\nBinary AUC: {mean_auc:.3f} ± {std_auc:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'stable_states_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n💾 RESULTS SAVED TO:")
        print(f"   📁 {results_dir}/")
        print(f"   📊 stable_states_validation_results.json")
        print(f"   📈 stable_states_validation_analysis.png")
        print(f"   📊 stable_states_detailed_performance.png")
        print(f"   📊 stable_states_confusion_matrix.png")
        
        # Final interpretation
        print(f"\n💡 INTERPRETATION:")
        if validation_results['assessment']['validates_main_results']:
            print("   ✅ VALIDATION SUCCESSFUL:")
            print("     • Model demonstrates significant discrimination between stable breathing states")
            print("     • Results support the validity of the main RNN decoder findings")
            print("     • The model is learning meaningful physiological patterns, not just artifacts")
        else:
            print("   ⚠️  VALIDATION CONCERNS:")
            print("     • Limited discrimination between stable breathing states")
            print("     • Main results may be driven by transition artifacts rather than stable patterns")
            print("     • Consider further investigation of feature selection and preprocessing")
        
        if mean_auc > 0.70:
            print("   🎯 CLINICAL RELEVANCE:")
            print("     • Strong binary discrimination suggests potential for real-time monitoring")
            print("     • Could be useful for breathing pattern assessment applications")
        
    else:
        print("❌ No successful folds - validation failed")
    
    print(f"\n===== STABLE STATES VALIDATION COMPLETE =====")


if __name__ == '__main__':
    main()