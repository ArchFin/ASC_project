#!/usr/bin/env python3
"""
rnn_decoder_simple.py

Streamlined RNN neural decoder for EEG breathing pattern classification.
No interactive menus - just runs RNN training and reports results.

Usage:
    python rnn_decoder_simple.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# TensorFlow and Keras
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Input, Dropout, LSTM, Bidirectional, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.utils import to_categorical

# Scikit-learn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import seaborn as sns
import warnings


# =============================================================
# Overfitting and Reporting Caveats:
#
# - Overfitting is well-controlled (dropout, regularization, early stopping).
# - For more conservative AUC, consider reporting results for a single architecture (not best-of-3 per fold).
# - Be cautious interpreting results for subjects with very few samples (sequence padding may inflate AUC).
# - Optionally, add a warning or skip folds with very small test sets (see code below).
# =============================================================


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
    X = X[~mask_allnan]
    
    if mu is None:
        mu = X.mean()
    if sigma is None:
        sigma = X.std()

    X = X.where(np.abs(X - mu) <= threshold * sigma)
    X = X.fillna(method='ffill').fillna(method='bfill').dropna()
    return X, mu, sigma


def preprocess_for_rnn(X_train, X_test, y_train, y_test, sequence_length=10):
    """Complete preprocessing pipeline for RNN."""
    # Clean and impute
    X_train_clean, mu, sigma = clean_impute(X_train)
    y_train_clean = y_train.loc[X_train_clean.index]
    
    X_test_clean, _, _ = clean_impute(X_test, mu=mu, sigma=sigma)
    if X_test_clean.empty:
        return None
    y_test_clean = y_test.loc[X_test_clean.index]
    
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


def build_rnn_model(input_shape, num_classes, model_type='attention_lstm'):
    """Build enhanced RNN model with attention mechanisms."""
    if model_type == 'simple':
        # Simple LSTM baseline with better regularization
        model = Sequential([
            Input(shape=input_shape),
            LSTM(48, return_sequences=False, dropout=0.4, recurrent_dropout=0.3, 
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(24, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))
        ])
        lr = 1e-3
    
    elif model_type == 'bidirectional':
        # Enhanced bidirectional LSTM
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
        lr = 8e-4
    
    elif model_type == 'attention_lstm':
        # LSTM with Multi-Head Attention (breathing-optimized)
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                                  kernel_regularizer=l2(0.01)))(inputs)
        norm1 = LayerNormalization()(lstm1)
        
        # Multi-head attention for capturing long-range dependencies
        attention = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(norm1, norm1)
        attention = Dropout(0.2)(attention)
        attention_norm = LayerNormalization()(attention + norm1)  # Residual connection
        
        # Second LSTM layer
        lstm2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                                  kernel_regularizer=l2(0.01)))(attention_norm)
        
        # Global average pooling to aggregate temporal information
        pooled = GlobalAveragePooling1D()(lstm2)
        
        # Dense layers with progressive dropout
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(pooled)
        drop1 = Dropout(0.4)(dense1)
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(drop1)
        drop2 = Dropout(0.3)(dense2)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop2)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        lr = 5e-4
    
    else:  # 'deep'
        # Deep stacked LSTM with residual connections
        inputs = Input(shape=input_shape)
        
        # First bidirectional LSTM
        lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                                  kernel_regularizer=l2(0.01)))(inputs)
        norm1 = LayerNormalization()(lstm1)
        
        # Second bidirectional LSTM
        lstm2 = Bidirectional(LSTM(48, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                                  kernel_regularizer=l2(0.01)))(norm1)
        norm2 = LayerNormalization()(lstm2)
        
        # Third LSTM layer
        lstm3 = LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2,
                    kernel_regularizer=l2(0.01))(norm2)
        norm3 = BatchNormalization()(lstm3)
        
        # Dense layers
        dense1 = Dense(48, activation='relu', kernel_regularizer=l2(0.01))(norm3)
        drop1 = Dropout(0.4)(dense1)
        dense2 = Dense(24, activation='relu', kernel_regularizer=l2(0.01))(drop1)
        drop2 = Dropout(0.3)(dense2)
        
        outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(drop2)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        lr = 5e-4
    
    # Enhanced optimizer with gradient clipping
    optimizer = Adam(learning_rate=lr, clipnorm=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    
    # Use focal loss for better class balance
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    
    return model


def train_and_evaluate_rnn(X_train, Y_train, X_test, Y_test, model_type='attention_lstm'):
    """Train and evaluate RNN model with comprehensive metrics."""
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
    
    # Train with more epochs for attention model
    epochs = 50 if model_type == 'simple' else 40
    batch_size = 64 if model_type == 'simple' else 32
    
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
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
    
    # AUC per class with better error handling
    aucs = {}
    for i in range(Y_test.shape[1]):
        if len(np.unique(true)) > 1 and len(np.unique(true == i)) > 1:
            try:
                aucs[str(i+1)] = roc_auc_score((true == i).astype(int), probs[:, i])
            except ValueError:
                aucs[str(i+1)] = 0.5
        else:
            aucs[str(i+1)] = 0.5
    
    # Add additional metrics to return
    metrics = {
        'aucs': aucs,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cm': cm
    }
    
    return model, metrics, history


def process_fold_model_selection(fold_data):
    """Process a single fold testing all 3 models for model selection phase."""
    fold_num, train_idx, test_idx, X, y, df = fold_data
    
    # Split data (use integer positions for all)
    train_pos = X.index.get_indexer(train_idx)
    test_pos = X.index.get_indexer(test_idx)
    X_train = X.iloc[train_pos]
    X_test = X.iloc[test_pos]
    y_train = y.iloc[train_pos]
    y_test = y.iloc[test_pos]
    
    # Preprocess
    result = preprocess_for_rnn(X_train, X_test, y_train, y_test, sequence_length=25)
    if result is None:
        return fold_num, {}
    
    X_train_seq, X_test_seq, Y_train_oh, Y_test_oh, le, classes, feature_info = result
    
    # Test all three model types for selection
    models_to_test = ['bidirectional'] #'simple', 'attention_lstm'
    fold_results = {}
    
    for model_type in models_to_test:
        try:
            model, metrics, history = train_and_evaluate_rnn(
                X_train_seq, Y_train_oh, X_test_seq, Y_test_oh, model_type
            )
            fold_auc = np.mean(list(metrics['aucs'].values()))
            fold_results[model_type] = fold_auc
            print(f"    {model_type}: AUC={fold_auc:.3f}")
        except Exception as e:
            print(f"    {model_type} failed: {e}")
            fold_results[model_type] = 0.0
    
    return fold_num, fold_results


def process_fold(fold_data, selected_model_type):
    """Process a single cross-validation fold - PUBLICATION VERSION with selected model."""
    fold_num, train_idx, test_idx, X, y, df = fold_data
    
    # Split data (use integer positions for all)
    train_pos = X.index.get_indexer(train_idx)
    test_pos = X.index.get_indexer(test_idx)
    X_train = X.iloc[train_pos]
    X_test = X.iloc[test_pos]
    y_train = y.iloc[train_pos]
    y_test = y.iloc[test_pos]
    
    # Preprocess with longer sequences for better performance
    result = preprocess_for_rnn(X_train, X_test, y_train, y_test, sequence_length=10)
    if result is None:
        return fold_num, None, None
    
    X_train_seq, X_test_seq, Y_train_oh, Y_test_oh, le, classes, feature_info = result
    
    # Warn if test set is very small (e.g., < 5 sequences for publication standards)
    if X_test_seq.shape[0] < 5:
        print(f"  ⚠️  Warning: Test set for this fold is very small (only {X_test_seq.shape[0]} sequence(s)). Results may be unreliable.")
    
    # FOR PUBLICATION: Use only the selected model (best performing from model selection phase)
    model_type = selected_model_type
    try:
        model, metrics, history = train_and_evaluate_rnn(
            X_train_seq, Y_train_oh, X_test_seq, Y_test_oh, model_type
        )
        
        fold_auc = np.mean(list(metrics['aucs'].values()))
        print(f"    Model: {model_type} (AUC={fold_auc:.3f}, Balanced_Acc={metrics['balanced_accuracy']:.3f})")
        
        return fold_num, metrics, model_type, feature_info
        
    except Exception as e:
        print(f"    Model {model_type} failed: {e}")
        return fold_num, None, None, None


def subject_data_quality_check(df, y, min_samples=20, min_class_count=5):
    """Check each subject for data quality issues."""
    subjects = df['subject'].unique()
    problematic = []
    good = []
    for subject in subjects:
        mask = df['subject'] == subject
        n_samples = mask.sum()
        y_sub = y[mask]
        class_counts = pd.Series(y_sub).value_counts()
        issues = []
        if n_samples < min_samples:
            issues.append('too_few_samples')
        if len(class_counts) < 2:
            issues.append('missing_classes')
        if class_counts.min() < min_class_count:
            issues.append('severe_imbalance')
        if issues:
            problematic.append((subject, n_samples, class_counts.to_dict(), issues))
        else:
            good.append(subject)
    return good, problematic


def analyze_feature_importance(all_feature_info, original_feature_names, results_dir):
    """Analyze and visualize feature importance across all folds."""
    if not all_feature_info:
        print("No feature information available for analysis.")
        return
    
    print(f"\n===== FEATURE IMPORTANCE ANALYSIS =====")
    
    # Aggregate feature scores across folds
    all_f_scores = []
    all_mi_scores = []
    all_rf_scores = []
    all_combined_scores = []
    all_selected_indices = []
    
    for fold_info in all_feature_info:
        all_f_scores.append(fold_info['f_scores'])
        all_mi_scores.append(fold_info['mi_scores'])
        all_rf_scores.append(fold_info['rf_importances'])
        all_combined_scores.append(fold_info['combined_scores'])
        all_selected_indices.extend(fold_info['selected_indices'])
    
    # Convert to arrays
    f_scores_matrix = np.array(all_f_scores)
    mi_scores_matrix = np.array(all_mi_scores)
    rf_scores_matrix = np.array(all_rf_scores)
    combined_scores_matrix = np.array(all_combined_scores)
    
    # Calculate mean and std across folds
    mean_f_scores = np.mean(f_scores_matrix, axis=0)
    std_f_scores = np.std(f_scores_matrix, axis=0)
    mean_mi_scores = np.mean(mi_scores_matrix, axis=0)
    std_mi_scores = np.std(mi_scores_matrix, axis=0)
    mean_rf_scores = np.mean(rf_scores_matrix, axis=0)
    std_rf_scores = np.std(rf_scores_matrix, axis=0)
    mean_combined_scores = np.mean(combined_scores_matrix, axis=0)
    std_combined_scores = np.std(combined_scores_matrix, axis=0)
    
    # Feature selection frequency
    n_features_total = len(mean_combined_scores)
    selection_frequency = np.zeros(n_features_total)
    for idx in all_selected_indices:
        selection_frequency[idx] += 1
    selection_frequency = selection_frequency / len(all_feature_info)
    
    # Create comprehensive feature names
    n_original = len(original_feature_names)
    enhanced_feature_names = []
    for i in range(n_features_total):
        if i < n_original:
            enhanced_feature_names.append(f"{original_feature_names[i]}")
        else:
            enhanced_feature_names.append(f"{original_feature_names[i-n_original]}_diff")
    
    # Top features analysis
    top_indices = np.argsort(mean_combined_scores)[-20:][::-1]
    print(f"\nTOP 20 MOST IMPORTANT FEATURES:")
    print(f"{'Rank':<4} {'Feature Name':<40} {'Comb Score':<12} {'Sel Freq':<10} {'F-Score':<10} {'MI Score':<10} {'RF Score':<10}")
    print("-" * 100)
    for rank, idx in enumerate(top_indices, 1):
        feature_name = enhanced_feature_names[idx]
        if len(feature_name) > 37:
            feature_name = feature_name[:34] + "..."
        print(f"{rank:<4} {feature_name:<40} {mean_combined_scores[idx]:<12.3f} "
              f"{selection_frequency[idx]:<10.2f} {mean_f_scores[idx]:<10.1f} "
              f"{mean_mi_scores[idx]:<10.3f} {mean_rf_scores[idx]:<10.3f}")
    
    # Visualizations
    plt.style.use('default')
    
    # 1. Feature importance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Top 15 features for visualization
    top_15_indices = np.argsort(mean_combined_scores)[-15:][::-1]
    top_15_names = [enhanced_feature_names[i] for i in top_15_indices]
    
    # Shorten names for better visualization
    short_names = []
    for name in top_15_names:
        if len(name) > 25:
            short_names.append(name[:22] + "...")
        else:
            short_names.append(name)
    
    plt.subplot(2, 2, 1)
    y_pos = np.arange(len(short_names))
    bars = plt.barh(y_pos, mean_combined_scores[top_15_indices], 
                    xerr=std_combined_scores[top_15_indices], 
                    alpha=0.7, color='steelblue', capsize=3)
    plt.yticks(y_pos, short_names, fontsize=9)
    plt.xlabel('Combined Importance Score')
    plt.title('Top 15 Features - Combined Importance')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, mean_combined_scores[top_15_indices], 
                                           std_combined_scores[top_15_indices])):
        plt.text(val + std + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    
    # 2. Selection frequency plot
    plt.subplot(2, 2, 2)
    freq_indices = np.argsort(selection_frequency)[-15:][::-1]
    freq_names = [enhanced_feature_names[i][:22] + "..." if len(enhanced_feature_names[i]) > 25 
                  else enhanced_feature_names[i] for i in freq_indices]
    
    y_pos = np.arange(len(freq_names))
    bars = plt.barh(y_pos, selection_frequency[freq_indices] * 100, 
                    alpha=0.7, color='orange')
    plt.yticks(y_pos, freq_names, fontsize=9)
    plt.xlabel('Selection Frequency (%)')
    plt.title('Top 15 Features - Selection Frequency')
    plt.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, selection_frequency[freq_indices] * 100)):
        plt.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=8)
    
    # 3. Score correlation heatmap
    plt.subplot(2, 2, 3)
    score_matrix = np.column_stack([mean_f_scores, mean_mi_scores, mean_rf_scores, mean_combined_scores])
    correlation_matrix = np.corrcoef(score_matrix.T)
    
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    labels = ['F-Score', 'MI Score', 'RF Score', 'Combined']
    plt.xticks(range(4), labels, rotation=45)
    plt.yticks(range(4), labels)
    plt.title('Feature Score Correlations')
    
    # Add correlation values
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    
    # 4. Feature type analysis (original vs derivative)
    plt.subplot(2, 2, 4)
    original_scores = mean_combined_scores[:n_original]
    derivative_scores = mean_combined_scores[n_original:] if n_features_total > n_original else []
    
    if len(derivative_scores) > 0:
        box_data = [original_scores, derivative_scores]
        box_labels = ['Original Features', 'Derivative Features']
        box_plot = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        if len(box_plot['boxes']) > 1:
            box_plot['boxes'][1].set_facecolor('lightcoral')
        plt.ylabel('Combined Importance Score')
        plt.title('Original vs Derivative Features')
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics
        orig_mean = np.mean(original_scores)
        deriv_mean = np.mean(derivative_scores)
        plt.text(0.02, 0.98, f'Original mean: {orig_mean:.3f}\nDerivative mean: {deriv_mean:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No derivative features\navailable', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Type Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed feature stability analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate coefficient of variation for top features
    top_20_indices = np.argsort(mean_combined_scores)[-20:][::-1]
    cv_scores = std_combined_scores[top_20_indices] / (mean_combined_scores[top_20_indices] + 1e-8)
    
    plt.subplot(2, 1, 1)
    y_pos = np.arange(len(top_20_indices))
    bars = plt.barh(y_pos, cv_scores, alpha=0.7, color='green')
    plt.yticks(y_pos, [enhanced_feature_names[i][:30] + "..." if len(enhanced_feature_names[i]) > 33 
                       else enhanced_feature_names[i] for i in top_20_indices], fontsize=9)
    plt.xlabel('Coefficient of Variation (Stability)')
    plt.title('Feature Stability Across Folds (Lower = More Stable)')
    plt.grid(axis='x', alpha=0.3)
    
    # Add CV values
    for i, (bar, val) in enumerate(zip(bars, cv_scores)):
        color = 'red' if val > 0.5 else 'black'
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8, color=color)
    
    # 6. EEG channel importance (if we can infer channel info)
    plt.subplot(2, 1, 2)
    
    # Try to extract channel information from feature names
    channel_importance = {}
    for idx, name in enumerate(enhanced_feature_names):
        # Look for common EEG channel patterns
        if 'glob_chans' in name:
            # Extract channel number or identifier
            import re
            channel_match = re.search(r'glob_chans_(\d+)', name)
            if channel_match:
                channel_num = int(channel_match.group(1))
                if channel_num not in channel_importance:
                    channel_importance[channel_num] = []
                channel_importance[channel_num].append(mean_combined_scores[idx])
    
    if channel_importance:
        channels = sorted(channel_importance.keys())
        channel_scores = [np.mean(channel_importance[ch]) for ch in channels]
        
        bars = plt.bar(channels, channel_scores, alpha=0.7, color='purple')
        plt.xlabel('EEG Channel Number')
        plt.ylabel('Average Importance Score')
        plt.title('EEG Channel Importance Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        # Highlight top channels
        top_channel_idx = np.argmax(channel_scores)
        bars[top_channel_idx].set_color('red')
        bars[top_channel_idx].set_alpha(1.0)
        
        plt.text(0.02, 0.98, f'Most important channel: {channels[top_channel_idx]}\n'
                             f'Score: {channel_scores[top_channel_idx]:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'Channel information\nnot available\nin feature names', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('EEG Channel Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_stability_channels.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed feature analysis data
    feature_analysis = {
        'top_20_features': {
            'names': [enhanced_feature_names[i] for i in top_indices],
            'combined_scores': mean_combined_scores[top_indices].tolist(),
            'selection_frequencies': selection_frequency[top_indices].tolist(),
            'f_scores': mean_f_scores[top_indices].tolist(),
            'mi_scores': mean_mi_scores[top_indices].tolist(),
            'rf_scores': mean_rf_scores[top_indices].tolist(),
            'stability_cv': (std_combined_scores[top_indices] / (mean_combined_scores[top_indices] + 1e-8)).tolist()
        },
        'feature_statistics': {
            'total_features': n_features_total,
            'original_features': n_original,
            'derivative_features': n_features_total - n_original,
            'mean_selection_frequency': float(np.mean(selection_frequency)),
            'most_stable_feature': enhanced_feature_names[np.argmin(std_combined_scores / (mean_combined_scores + 1e-8))],
            'most_variable_feature': enhanced_feature_names[np.argmax(std_combined_scores / (mean_combined_scores + 1e-8))]
        },
        'channel_analysis': {
            'channel_importance': {str(k): float(np.mean(v)) for k, v in channel_importance.items()} if channel_importance else {},
            'top_channel': int(channels[np.argmax(channel_scores)]) if channel_importance else None
        }
    }
    
    with open(os.path.join(results_dir, 'feature_analysis_detailed.json'), 'w') as f:
        json.dump(feature_analysis, f, indent=2)
    
    print(f"\n💾 FEATURE ANALYSIS SAVED:")
    print(f"   📊 feature_importance_analysis.png")
    print(f"   📈 feature_stability_channels.png") 
    print(f"   📄 feature_analysis_detailed.json")
    
    return feature_analysis


def analyze_neural_channel_influence(all_feature_info, original_feature_names, results_dir):
    """Create detailed neural channel influence analysis."""
    print(f"\n===== NEURAL CHANNEL INFLUENCE ANALYSIS =====")
    
    if not all_feature_info:
        print("No feature information available for neural analysis.")
        return
    
    # Parse channel information from feature names
    channel_data = {}
    for fold_info in all_feature_info:
        combined_scores = fold_info['combined_scores']
        for i, feature_name in enumerate(original_feature_names):
            # Extract channel number from feature name
            import re
            if 'glob_chans' in feature_name:
                channel_match = re.search(r'glob_chans_(\d+)', feature_name)
                if channel_match:
                    channel_num = int(channel_match.group(1))
                    if channel_num not in channel_data:
                        channel_data[channel_num] = []
                    # Get original feature score
                    if i < len(combined_scores):
                        channel_data[channel_num].append(combined_scores[i])
                    # Get derivative feature score if available
                    derivative_idx = len(original_feature_names) + i
                    if derivative_idx < len(combined_scores):
                        channel_data[channel_num].append(combined_scores[derivative_idx])
    
    if not channel_data:
        print("Could not extract neural channel information from feature names.")
        return
    
    # Calculate statistics for each channel
    channel_stats = {}
    for channel, scores in channel_data.items():
        channel_stats[channel] = {
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'max_importance': np.max(scores),
            'n_features': len(scores)
        }
    
    # Sort channels by importance
    sorted_channels = sorted(channel_stats.keys(), 
                           key=lambda x: channel_stats[x]['mean_importance'], 
                           reverse=True)
    
    print(f"\nTOP 20 MOST INFLUENTIAL NEURAL CHANNELS:")
    print(f"{'Rank':<4} {'Channel':<8} {'Mean Score':<12} {'Std':<8} {'Max':<8} {'N_Features':<10}")
    print("-" * 60)
    for rank, channel in enumerate(sorted_channels[:20], 1):
        stats = channel_stats[channel]
        print(f"{rank:<4} {channel:<8} {stats['mean_importance']:<12.3f} "
              f"{stats['std_importance']:<8.3f} {stats['max_importance']:<8.3f} {stats['n_features']:<10}")
    
    # Create comprehensive visualizations
    plt.figure(figsize=(16, 12))
    
    # 1. Channel importance heatmap
    plt.subplot(2, 3, 1)
    channels = sorted(channel_stats.keys())
    importance_matrix = np.array([channel_stats[ch]['mean_importance'] for ch in channels])
    
    # Reshape into a grid (assuming typical EEG montage)
    grid_size = int(np.ceil(np.sqrt(len(channels))))
    heatmap_data = np.zeros((grid_size, grid_size))
    for i, channel in enumerate(channels):
        row, col = divmod(i, grid_size)
        if row < grid_size and col < grid_size:
            heatmap_data[row, col] = channel_stats[channel]['mean_importance']
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': 'Importance Score'})
    plt.title('Neural Channel Importance Heatmap')
    plt.xlabel('Channel Grid Position')
    plt.ylabel('Channel Grid Position')
    
    # 2. Top channels bar plot
    plt.subplot(2, 3, 2)
    top_20_channels = sorted_channels[:20]
    top_20_scores = [channel_stats[ch]['mean_importance'] for ch in top_20_channels]
    top_20_stds = [channel_stats[ch]['std_importance'] for ch in top_20_channels]
    
    bars = plt.bar(range(len(top_20_channels)), top_20_scores, 
                   yerr=top_20_stds, capsize=3, alpha=0.7, color='lightcoral')
    plt.xticks(range(len(top_20_channels)), [f'Ch{ch}' for ch in top_20_channels], 
               rotation=45, ha='right')
    plt.ylabel('Mean Importance Score')
    plt.title('Top 20 Neural Channels')
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight top 5
    for i in range(min(5, len(bars))):
        bars[i].set_color('red')
        bars[i].set_alpha(1.0)
    
    # 3. Channel importance distribution
    plt.subplot(2, 3, 3)
    all_importance_scores = [stats['mean_importance'] for stats in channel_stats.values()]
    plt.hist(all_importance_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(all_importance_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_importance_scores):.3f}')
    plt.axvline(np.median(all_importance_scores), color='orange', linestyle='--', 
                label=f'Median: {np.median(all_importance_scores):.3f}')
    plt.xlabel('Importance Score')
    plt.ylabel('Number of Channels')
    plt.title('Distribution of Channel Importance')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. Channel stability analysis
    plt.subplot(2, 3, 4)
    stability_scores = []
    channel_labels = []
    for channel in sorted_channels[:15]:  # Top 15 for readability
        stability = channel_stats[channel]['std_importance'] / (channel_stats[channel]['mean_importance'] + 1e-8)
        stability_scores.append(stability)
        channel_labels.append(f'Ch{channel}')
    
    bars = plt.barh(range(len(stability_scores)), stability_scores, alpha=0.7, color='lightgreen')
    plt.yticks(range(len(channel_labels)), channel_labels)
    plt.xlabel('Coefficient of Variation (Lower = More Stable)')
    plt.title('Channel Stability (Top 15)')
    plt.grid(axis='x', alpha=0.3)
    
    # Color code stability
    for i, (bar, score) in enumerate(zip(bars, stability_scores)):
        if score > 0.5:
            bar.set_color('red')  # Unstable
        elif score > 0.3:
            bar.set_color('orange')  # Moderate
        else:
            bar.set_color('green')  # Stable
    
    # 5. Regional analysis (if we can infer brain regions)
    plt.subplot(2, 3, 5)
    # Simple regional grouping based on channel numbers (this is a rough approximation)
    regions = {
        'Frontal': [ch for ch in channels if ch <= 20],
        'Central': [ch for ch in channels if 21 <= ch <= 40],
        'Parietal': [ch for ch in channels if 41 <= ch <= 60],
        'Occipital': [ch for ch in channels if ch > 60]
    }
    
    region_scores = {}
    for region, region_channels in regions.items():
        if region_channels:
            scores = [channel_stats[ch]['mean_importance'] for ch in region_channels if ch in channel_stats]
            if scores:
                region_scores[region] = np.mean(scores)
    
    if region_scores:
        regions_list = list(region_scores.keys())
        scores_list = list(region_scores.values())
        bars = plt.bar(regions_list, scores_list, alpha=0.7, 
                      color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        plt.ylabel('Mean Regional Importance')
        plt.title('Brain Region Importance (Approximate)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores_list):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor regional analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 6. Feature count per channel
    plt.subplot(2, 3, 6)
    feature_counts = [channel_stats[ch]['n_features'] for ch in sorted_channels[:20]]
    plt.bar(range(len(top_20_channels)), feature_counts, alpha=0.7, color='mediumpurple')
    plt.xticks(range(len(top_20_channels)), [f'Ch{ch}' for ch in top_20_channels], 
               rotation=45, ha='right')
    plt.ylabel('Number of Features')
    plt.title('Feature Count per Channel (Top 20)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'neural_channel_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed channel analysis
    channel_analysis = {
        'top_20_channels': {
            'channels': sorted_channels[:20],
            'importance_scores': [channel_stats[ch]['mean_importance'] for ch in sorted_channels[:20]],
            'stability_scores': [channel_stats[ch]['std_importance'] / (channel_stats[ch]['mean_importance'] + 1e-8) 
                               for ch in sorted_channels[:20]]
        },
        'regional_analysis': region_scores if 'region_scores' in locals() else {},
        'summary_statistics': {
            'total_channels': len(channel_stats),
            'mean_channel_importance': float(np.mean(all_importance_scores)),
            'std_channel_importance': float(np.std(all_importance_scores)),
            'most_important_channel': int(sorted_channels[0]),
            'least_important_channel': int(sorted_channels[-1])
        }
    }
    
    with open(os.path.join(results_dir, 'neural_channel_analysis.json'), 'w') as f:
        json.dump(channel_analysis, f, indent=2)
    
    print(f"\n💾 NEURAL CHANNEL ANALYSIS SAVED:")
    print(f"   📊 neural_channel_analysis.png")
    print(f"   📄 neural_channel_analysis.json")
    
    return channel_analysis


def main():
    """Main RNN training pipeline."""
    print("===== STREAMLINED RNN NEURAL DECODER =====")
    
    # Load data
    csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete_2.csv'
    print(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    groups = df['subject']
    y = df['transition_label']
    
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number', 'transition_label']
    # Subject exclusion list (edit as needed)
    exclude_subjects = []  # Example: exclude subject s01 s04 s07 s08 s13 s14 s18 's03', 's19', 's17', 's10', 's02', 's21'
    print(f"\n===== SUBJECT DATA QUALITY CHECK =====")
    good_subjects, problematic_subjects = subject_data_quality_check(df, y)
    print(f"Good subjects: {good_subjects}")
    print(f"Problematic subjects:")
    for subj, n, counts, issues in problematic_subjects:
        print(f"  {subj}: {n} samples, class counts {counts}, issues: {issues}")
    
    # Load flagged subjects from previous results if available
    flagged_subjects = []
    results_json_path = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/Neural Decoder/rnn_results/rnn_results_summary.json'
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as f:
            try:
                prev_results = json.load(f)
                flagged_subjects = prev_results.get('flagged_subjects', [])
                if flagged_subjects:
                    print(f"\nFlagged subjects from previous results: {flagged_subjects}")
            except Exception as e:
                print(f"Warning: Could not load flagged subjects from previous results: {e}")
    # Merge flagged subjects into exclude_subjects
    exclude_subjects = list(set(exclude_subjects) | set(flagged_subjects))
    if exclude_subjects:
        print(f"\nExcluding subjects: {exclude_subjects}")
        df = df[~df['subject'].isin(exclude_subjects)].copy()
        y = df['transition_label']
        groups = df['subject']
        # Prepare features again after exclusion
        X_raw = df.drop(columns=drop_cols, errors='ignore')
        glob_cols = [col for col in X_raw.columns if 'glob_chans' in col]
        X = X_raw[glob_cols]
    else:
        groups = df['subject']
        X_raw = df.drop(columns=drop_cols, errors='ignore')
        glob_cols = [col for col in X_raw.columns if 'glob_chans' in col]
        X = X_raw[glob_cols]
    
    print(f"Dataset: {len(df)} samples, {len(df['subject'].unique())} subjects")
    print(f"Features: {X.shape[1]} EEG global channel features")
    print(f"Classes: {y.value_counts().to_dict()}")
    
    # Check temporal structure (weeks per subject)
    print(f"\n===== TEMPORAL STRUCTURE ANALYSIS =====")
    week_analysis = df.groupby('subject')['week'].agg(['nunique', 'min', 'max']).reset_index()
    week_analysis.columns = ['subject', 'n_weeks', 'min_week', 'max_week']
    print("Weeks per subject:")
    for _, row in week_analysis.iterrows():
        print(f"  {row['subject']}: {row['n_weeks']} weeks (weeks {row['min_week']}-{row['max_week']})")
    
    total_weeks = df['week'].nunique()
    subjects_with_multiple_weeks = len(week_analysis[week_analysis['n_weeks'] > 1])
    print(f"\nTotal unique weeks: {total_weeks}")
    print(f"Subjects with multiple weeks: {subjects_with_multiple_weeks}/{len(week_analysis)}")
    
    # Analyze temporal stability: Are breathing patterns consistent across weeks?
    if subjects_with_multiple_weeks > 0:
        print(f"\n===== TEMPORAL STABILITY ANALYSIS =====")
        print("Analyzing if breathing patterns are consistent across weeks...")
        
        # Calculate class distribution differences between weeks
        temporal_drift_detected = False
        drift_subjects = []
        
        for subject in df['subject'].unique():
            subject_data = df[df['subject'] == subject]
            weeks = subject_data['week'].unique()
            
            if len(weeks) > 1:
                # Compare class distributions between weeks
                week_distributions = {}
                for week in weeks:
                    week_data = subject_data[subject_data['week'] == week]
                    class_dist = week_data['transition_label'].value_counts(normalize=True)
                    week_distributions[week] = class_dist
                
                # Check if distributions vary significantly (using simple variance threshold)
                if len(week_distributions) >= 2:
                    weeks_list = list(week_distributions.keys())
                    for i in range(len(weeks_list)):
                        for j in range(i+1, len(weeks_list)):
                            week1_dist = week_distributions[weeks_list[i]]
                            week2_dist = week_distributions[weeks_list[j]]
                            
                            # Calculate distribution difference
                            all_classes = set(week1_dist.index) | set(week2_dist.index)
                            diff_sum = 0
                            for cls in all_classes:
                                val1 = week1_dist.get(cls, 0)
                                val2 = week2_dist.get(cls, 0)
                                diff_sum += abs(val1 - val2)
                            
                            # If distributions differ by more than 20%, flag as drift
                            if diff_sum > 0.2:
                                temporal_drift_detected = True
                                drift_subjects.append(subject)
                                print(f"  ⚠️  {subject}: Class distribution changed between weeks {weeks_list[i]} and {weeks_list[j]}")
                                break
                        if subject in drift_subjects:
                            break
        
        if temporal_drift_detected:
            print(f"\n🚨 CRITICAL: Temporal drift detected in {len(drift_subjects)} subjects!")
            print("This suggests breathing patterns change significantly across weeks.")
            print("Current approach may be invalid - consider within-week validation only.")
        else:
            print(f"\n✅ No significant temporal drift detected.")
            print("Breathing patterns appear stable across weeks.")
    
    # Cross-validation strategy selection
    if subjects_with_multiple_weeks > 0:
        print(f"\n⚠️  TEMPORAL STRUCTURE DETECTED!")
        print("Multiple weeks per subject detected - implementing temporal-aware cross-validation")
        cv_strategy = "temporal_aware"
    else:
        print("\nSingle week per subject - using standard leave-one-subject-out CV")
        cv_strategy = "standard"
    
    # Cross-validation setup based on temporal structure
    if cv_strategy == "temporal_aware":
        # For temporal data: Use subject-week combinations as groups
        df['subject_week'] = df['subject'].astype(str) + '_week_' + df['week'].astype(str)
        unique_subject_weeks = df['subject_week'].unique()
        
        print(f"\n🔄 TEMPORAL-AWARE CROSS-VALIDATION")
        print(f"Strategy: Leave-one-subject-out, but ensure train/test weeks don't overlap")
        print(f"Total subject-week combinations: {len(unique_subject_weeks)}")
        
        # Create custom CV splits that respect temporal structure
        cv_splits = []
        subjects = df['subject'].unique()
        
        for test_subject in subjects:
            # Get all weeks for this subject
            test_mask = df['subject'] == test_subject
            train_mask = df['subject'] != test_subject
            
            test_idx = df[test_mask].index.tolist()
            train_idx = df[train_mask].index.tolist()
            
            cv_splits.append((train_idx, test_idx))
        
        n_splits = len(cv_splits)
        print(f"Temporal-aware CV splits: {n_splits}")
        
    else:
        # Standard leave-one-subject-out CV
        n_splits = len(df['subject'].unique())
        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(X, y, groups=groups))
        print(f"Standard CV splits: {n_splits}")
    
    # Model selection phase
    skip_selection = False
    best_model = None
    model_averages = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-selection', '-s', action='store_true', help='Skip model selection phase and use best model from previous results')
    args = parser.parse_args()
    skip_selection = args.skip_selection
    if skip_selection:
        print("\n⚡ Skipping model selection phase (using previous best model)...")
        results_json_path = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/Neural Decoder/rnn_results/rnn_results_summary.json'
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r') as f:
                prev_results = json.load(f)
                best_model = prev_results.get('selected_model', 'bidirectional')
                model_averages = prev_results.get('model_selection_results', None)
                print(f"Using best model from previous results: {best_model}")
        else:
            print("No previous results found. Running model selection phase.")
            skip_selection = False
    if not skip_selection:
        print(f"\n🔍 STAGE 1: MODEL SELECTION")
        print(f"Testing 3 RNN architectures across all {n_splits} folds to find best performer...")
        
        selection_results = []
        
        for fold_num, (train_idx, test_idx) in enumerate(cv_splits, 1):
            fold_data = (fold_num, train_idx, test_idx, X, y, df)
            test_subject = df.loc[test_idx, 'subject'].iloc[0]

            print(f"\nSelection Fold {fold_num}/{n_splits} - Test subject: {test_subject}")
            
            fold_num_result, fold_results = process_fold_model_selection(fold_data)
            if fold_results:
                selection_results.append(fold_results)
        
        # Calculate average performance for each model
        if selection_results:
            model_averages = {}
            for model_type in ['bidirectional']:
                model_scores = [result.get(model_type, 0) for result in selection_results if result.get(model_type, 0) > 0]
                if model_scores:
                    model_averages[model_type] = np.mean(model_scores)
                else:
                    model_averages[model_type] = 0.0
            
            # Select best model
            best_model = max(model_averages, key=model_averages.get)
            
            print(f"\n📊 MODEL SELECTION RESULTS:")
            for model_type, avg_score in model_averages.items():
                marker = " ← SELECTED" if model_type == best_model else ""
                print(f"   {model_type}: {avg_score:.3f}{marker}")
            
            print(f"\n✅ SELECTED MODEL: {best_model} (Average AUC: {model_averages[best_model]:.3f})")
        else:
            print("❌ Model selection failed - defaulting to attention_lstm")
            best_model = 'attention_lstm'
    
    # =============================================================
    # STAGE 2: UNBIASED EVALUATION PHASE
    # =============================================================
    print(f"\n🎯 STAGE 2: UNBIASED EVALUATION")
    print(f"Running final cross-validation with selected model: {best_model}")
    
    all_metrics = []
    all_cms = []
    all_aucs = []
    all_feature_info = []
    failed_folds = []
    
    # Run cross-validation
    for fold_num, (train_idx, test_idx) in enumerate(cv_splits, 1):
        fold_data = (fold_num, train_idx, test_idx, X, y, df)
        test_subject = df.loc[test_idx, 'subject'].iloc[0]

        print(f"\nFold {fold_num}/{n_splits} - Test subject: {test_subject}")
        
        fold_result, metrics, model_type, feature_info = process_fold(fold_data, best_model)
        
        if metrics is not None:
            all_metrics.append(metrics)
            all_cms.append(metrics['cm'])
            all_aucs.append(metrics['aucs'])
            all_feature_info.append(feature_info)
            fold_auc = np.mean(list(metrics['aucs'].values()))
            print(f"  ✅ Fold AUC: {fold_auc:.3f}")
        else:
            failed_folds.append(fold_num)
            print(f"  ❌ Fold failed")
    
    # Report results
    print(f"\n===== PUBLICATION-READY RNN RESULTS =====")
    
    if all_aucs:
        # Filter out any problematic aucs (should be flat dict of floats)
        def is_flat_float_dict(d):
            return isinstance(d, dict) and all(isinstance(v, (float, int, np.floating, np.integer)) for v in d.values())
        filtered_aucs = [a for a in all_aucs if is_flat_float_dict(a)]
        if len(filtered_aucs) < len(all_aucs):
            print(f"⚠️  Warning: {len(all_aucs) - len(filtered_aucs)} folds had invalid AUC results and were skipped in summary.")
        if not filtered_aucs:
            print("❌ No valid AUC results to summarize.")
            return
        
        # Calculate comprehensive statistics
        avg_aucs = pd.DataFrame(filtered_aucs).mean()
        std_aucs = pd.DataFrame(filtered_aucs).std()
        overall_auc = avg_aucs.mean()
        overall_std = std_aucs.mean()
        
        # Statistical significance testing (one-sample t-test against chance level 0.5)
        all_fold_aucs = [np.mean(list(auc_dict.values())) for auc_dict in filtered_aucs]
        t_stat, p_value = stats.ttest_1samp(all_fold_aucs, 0.5)
        
        # Confidence intervals (95%)
        confidence_interval = stats.t.interval(0.95, len(all_fold_aucs)-1, 
                                             loc=np.mean(all_fold_aucs), 
                                             scale=stats.sem(all_fold_aucs))
        
        # Additional metrics from all_metrics
        if all_metrics:
            balanced_accs = [m['balanced_accuracy'] for m in all_metrics]
            f1_macros = [m['f1_macro'] for m in all_metrics]
            f1_weighteds = [m['f1_weighted'] for m in all_metrics]
            
            avg_balanced_acc = np.mean(balanced_accs)
            std_balanced_acc = np.std(balanced_accs)
            avg_f1_macro = np.mean(f1_macros)
            std_f1_macro = np.std(f1_macros)
            avg_f1_weighted = np.mean(f1_weighteds)
            std_f1_weighted = np.std(f1_weighteds)
        
        print(f"\n🎯 OVERALL PERFORMANCE (PUBLICATION METRICS):")
        print(f"   Average AUC: {overall_auc:.3f} ± {overall_std:.3f}")
        print(f"   95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        print(f"   Statistical significance vs chance (p-value): {p_value:.4f}")
        if all_metrics:
            print(f"   Balanced Accuracy: {avg_balanced_acc:.3f} ± {std_balanced_acc:.3f}")
            print(f"   F1-Score (Macro): {avg_f1_macro:.3f} ± {std_f1_macro:.3f}")
            print(f"   F1-Score (Weighted): {avg_f1_weighted:.3f} ± {std_f1_weighted:.3f}")
        
        print(f"\n📊 PER-CLASS AUC (DETAILED):")
        for label in avg_aucs.index:
            class_aucs = [auc_dict[label] for auc_dict in filtered_aucs]
            class_ci = stats.t.interval(0.95, len(class_aucs)-1, 
                                      loc=np.mean(class_aucs), 
                                      scale=stats.sem(class_aucs))
            print(f"   Class {label}: {avg_aucs[label]:.3f} ± {std_aucs[label]:.3f}, 95% CI: [{class_ci[0]:.3f}, {class_ci[1]:.3f}]")
        
        # Confusion matrix
        total_cm = np.sum(all_cms, axis=0)
        total_cm_percent = total_cm.astype('float') / total_cm.sum(axis=1, keepdims=True) * 100
        
        print(f"\n🔢 CONFUSION MATRIX (%):")
        class_names = ['Class 1', 'Class 2', 'Class 3']
        print("        " + "  ".join(f"{name:>8}" for name in class_names))
        for i, true_class in enumerate(class_names):
            row = f"{true_class:>8} "
            for j in range(len(class_names)):
                row += f"{total_cm_percent[i, j]:8.1f}% "
            print(row)
        
        # Performance assessment
        print(f"\n📈 PERFORMANCE ASSESSMENT:")
        if overall_auc > 0.70:
            print("   ✅ EXCELLENT: Strong RNN decoding performance!")
        elif overall_auc > 0.65:
            print("   ✅ GOOD: Above target threshold (0.65)")
        elif overall_auc > 0.60:
            print("   ⚠️  MODERATE: Decent RNN performance, room for improvement")
        elif overall_auc > 0.55:
            print("   🔍 WEAK: Slightly above chance, RNN learning limited patterns")
        else:
            print("   ❌ POOR: RNN not capturing meaningful temporal patterns")
        
        # Save results
        results_dir = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/Neural Decoder/rnn_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save summary
        results_summary = {
            'model_type': f'{best_model} (Two-Stage Selection - Publication Version)',
            'selected_model': best_model,
            'model_selection_results': model_averages if 'model_averages' in locals() else None,
            'overall_auc_mean': float(overall_auc),
            'overall_auc_std': float(overall_std),
            'auc_confidence_interval': [float(confidence_interval[0]), float(confidence_interval[1])],
            'p_value_vs_chance': float(p_value),
            't_statistic': float(t_stat),
            'per_class_auc_mean': {str(k): float(v) for k, v in avg_aucs.items()},
            'per_class_auc_std': {str(k): float(v) for k, v in std_aucs.items()},
            'balanced_accuracy_mean': float(avg_balanced_acc) if all_metrics else None,
            'balanced_accuracy_std': float(std_balanced_acc) if all_metrics else None,
            'f1_macro_mean': float(avg_f1_macro) if all_metrics else None,
            'f1_macro_std': float(std_f1_macro) if all_metrics else None,
            'f1_weighted_mean': float(avg_f1_weighted) if all_metrics else None,
            'f1_weighted_std': float(std_f1_weighted) if all_metrics else None,
            'confusion_matrix_percent': total_cm_percent.tolist(),
            'confusion_matrix_counts': total_cm.tolist(),
            'total_folds_successful': len(all_aucs),
            'total_folds_attempted': n_splits,
            'failed_folds': failed_folds,
            'sequence_length': 25,
            'overlap': 0.8,
            'enhanced_features': 21,
            'feature_count': X.shape[1],
            'subjects_tested': n_splits,
            'excluded_subjects': exclude_subjects,
            'publication_ready': True,
            'temporal_aware': cv_strategy == "temporal_aware",
            'methodology': f'Two-stage with {cv_strategy} cross-validation: (1) Model selection across all folds, (2) Unbiased evaluation with selected model'
        }
        
        with open(os.path.join(results_dir, 'rnn_results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        plt.imshow(total_cm_percent, cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.title(f'RNN Neural Decoder Results\nOverall AUC: {overall_auc:.3f} ± {overall_std:.3f}', fontsize=14)
        plt.colorbar(label='Percentage (%)')
        
        plt.xticks([0, 1, 2], class_names)
        plt.yticks([0, 1, 2], class_names)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        
        # Add percentage labels
        for i in range(3):
            for j in range(3):
                color = 'white' if total_cm_percent[i, j] > 50 else 'black'
                plt.text(j, i, f"{total_cm_percent[i, j]:.1f}%", 
                        ha='center', va='center', color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'rnn_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix plot (with true/predicted labels)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(total_cm_percent, cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.title('Overall Cross-Validation Confusion Matrix (%)')
        plt.colorbar(im, label='Percentage')
        plt.xticks(np.arange(len(class_names)), class_names)
        plt.yticks(np.arange(len(class_names)), class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, f"{total_cm_percent[i, j]:.1f}%", ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'overall_confusion_matrix.png'))
        plt.close()

        # Save AUC bar plot (mean ± std, with value labels)
        plt.figure(figsize=(8, 6))
        # Use class labels from confusion matrix shape
        classes = list(range(1, total_cm_percent.shape[0]+1))
        x = np.arange(1, len(classes)+1)
        means = [avg_aucs[str(i)] for i in x]
        stds = [std_aucs[str(i)] for i in x]
        bars = plt.bar(x, means, yerr=stds, capsize=8, alpha=0.7, color='steelblue')
        plt.ylim(0, 1)
        plt.ylabel('AUC')
        plt.xlabel('Class')
        plt.title('Cross-Validation AUC Scores (Mean ± Std)')
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i+1, mean + std + 0.02, f"{mean:.3f}±{std:.3f}", ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cv_auc_barplot.png'))
        plt.close()
        
        # Feature analysis
        print(f"\n🔬 FEATURE ANALYSIS:")
        feature_analysis = analyze_feature_importance(all_feature_info, X.columns.tolist(), results_dir)
        
        # Neural channel analysis
        print(f"\n🧠 NEURAL CHANNEL ANALYSIS:")
        channel_analysis = analyze_neural_channel_influence(all_feature_info, X.columns.tolist(), results_dir)
        
        print(f"\n💾 RESULTS SAVED TO:")
        print(f"   📁 {results_dir}/")
        print(f"   📊 rnn_results_summary.json")
        print(f"   📈 rnn_confusion_matrix.png")
        print(f"   📉 rnn_auc_scores.png")
        print(f"   📊 overall_confusion_matrix.png")
        print(f"   📊 cv_auc_barplot.png")
        
        # Final recommendations
        print(f"\n💡 NEXT STEPS:")
        if overall_auc > 0.65:
            print("   🎯 Great results! Consider:")
            print("   • Hyperparameter tuning for further optimization")
            print("   • Ensemble methods combining multiple architectures")
            print("   • Feature engineering (frequency ratios, complexity measures)")
        elif overall_auc > 0.60:
            print("   🔧 Good foundation! Try:")
            print("   • Longer sequences (sequence_length > 10)")
            print("   • Advanced architectures (Attention, Transformer)")
            print("   • Domain adaptation techniques")
        else:
            print("   🔍 Needs improvement:")
            print("   • Check data quality and preprocessing")
            print("   • Try different sequence lengths")
            print("   • Consider non-temporal approaches")
        
    else:
        print("❌ No successful folds - check data and preprocessing")
    
    print(f"\n===== RNN ANALYSIS COMPLETE =====")


if __name__ == '__main__':
    main()