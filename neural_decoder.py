#!/usr/bin/env python3
"""
neural_decoder.py

Standalone script for multi-state neural decoding using Keras.

Usage:
    python neural_decoder.py --input_csv path/to/data.csv --drop_cols Subject Week Session Condition Cluster

This script:
  1. Loads and filters your data
  2. Cleans, imputes, and scales features
  3. Encodes labels for multi-class classification
  4. Builds and trains a feed-forward neural network (MLPC)
  5. Outputs performance metrics and saves model + label encoder
"""
import tensorflow as tf
print(tf.__version__)  # Verify TensorFlow version

import sys
print(f"Python executable: {sys.executable}")
print(f"Version info: {sys.version_info}")

import argparse
import pandas as pd
import numpy as np

# Scikit-learn utilities for splitting, scaling, encoding labels, and metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix

# Keras (via TensorFlow) layers and utilities for building an MLPC
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Input, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.utils import to_categorical

# Keras Tuner for hyperparameter search
import keras_tuner as kt

# Utilities for handling imbalanced classes, shuffling, and saving objects
from sklearn.utils import resample, shuffle
import joblib

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_data(csv_path, drop_cols=None, target_col='transition_label'):
    """
    1. Loads the CSV into a pandas DataFrame.
    2. Drops any unwanted columns (metadata like Subject, Week, etc.).
    3. Splits into feature matrix X and label vector y.
    """
    df = pd.read_csv(csv_path)
    if drop_cols:
        df = df.drop(drop_cols, axis=1, errors='ignore')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def select_features(X, pattern_keep=None):
    """
    Optionally select only columns whose names contain one of the substrings in pattern_keep.
    Useful if you want to focus on features like 'glob' channels only.
    """
    if not pattern_keep:
        return X
    cols = [c for c in X.columns if any(p in c for p in pattern_keep)]
    return X[cols]


def clean_impute(X, threshold=2.5):
    """
    1. Remove rows where all features are NaN.
    2. Clip outliers beyond `threshold * std` for each column, setting them to NaN.
    3. Forward-fill then backward-fill to impute missing values.
    4. Drop any remaining NaNs.
    This ensures no extreme outliers remain and that each row has no missing entries.
    """
    mask_allnan = X.isna().all(axis=1)
    X = X[~mask_allnan]
    mu = X.mean()
    sigma = X.std()
    # Any value with |value - mean| > threshold * sigma becomes NaN
    X = X.where(np.abs(X - mu) <= threshold * sigma)
    X = X.fillna(method='ffill').fillna(method='bfill').dropna()
    return X


def preprocess(X, y, test_size=0.2, random_state=42):
    """
    1. Clean and impute X.
    2. Align y to the cleaned X indices.
    3. Shuffle data.
    4. Standardize features to zero mean, unit variance (using StandardScaler).
    5. Label-encode y, then convert to one-hot (to_categorical).
    6. Split into training and test sets (stratified to preserve class proportions).
    Returns: X_train, X_test, Y_train (one-hot), Y_test (one-hot), label encoder, classes.
    """
    X_clean = clean_impute(X)
    y_clean = y.loc[X_clean.index]
    Xs, ys = shuffle(X_clean, y_clean, random_state=random_state)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)  # Each feature now has mean=0, std=1

    le = LabelEncoder()
    y_enc = le.fit_transform(ys)   # Integer encode labels
    Y = to_categorical(y_enc)      # Convert to one-hot vectors

    X_train, X_test, Y_train, Y_test = train_test_split(
        Xs, Y, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    return X_train, X_test, Y_train, Y_test, le, le.classes_


def build_model(input_dim, num_classes, hp=None):
    """
    Build a feed-forward neural network (MLPC) with:
    - Input layer of size `input_dim`.
    - Two hidden Dense layers with nonlinear activation, BatchNormalization, Dropout, and L2 regularization.
      * Hidden layer 1: `units` neurons (hyperparameter between 32 and 256).
      * Hidden layer 2: half as many neurons as layer 1.
    - Output layer: `num_classes` neurons with softmax activation.
    If `hp` is provided, Keras Tuner will sample:
      * units: [32, 64, 96, ..., 256]
      * activation: 'relu' or 'tanh'
      * dropout_rate: between 0.2 and 0.6
      * learning rate: between 1e-4 and 1e-2 (log scale)
    Otherwise, defaults: units=64, lr=1e-3, activation='relu', dropout_rate=0.5.
    """
    if hp:
        # Hyperparameter search space
        units = hp.Int('units', 32, 256, step=32)
        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        activation = hp.Choice('activation', ['relu', 'tanh'])
        dropout_rate = hp.Float('dropout_rate', 0.2, 0.6, step=0.1)
    else:
        # Default hyperparameters
        units, lr, activation, dropout_rate = 64, 1e-3, 'relu', 0.5

    model = Sequential([
        # Input layer placeholder
        Input(shape=(input_dim,)),

        # Hidden layer 1: Dense computes z = W·x + b, then applies activation(·)
        Dense(units, activation=activation, kernel_regularizer=l2(1e-3)),
        BatchNormalization(),           # Normalize activations to improve stability
        Dropout(dropout_rate),          # Randomly zero a fraction to prevent overfitting

        # Hidden layer 2: half as many neurons
        Dense(units // 2, activation=activation, kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Dropout(dropout_rate / 2),

        # Output layer: one neuron per class, softmax → class probabilities
        Dense(num_classes, activation='softmax')
    ])

    # Compile model: categorical cross-entropy loss + Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def tune_hyperparameters(X_train, Y_train, X_val, Y_val):
    """
    Uses Keras Tuner to search for the best hyperparameters.
    - Splits provided X_train into training/validation via arguments already provided.
    - Builds a tuner that tries different units, activation, dropout_rate, and learning rates.
    - Objective: minimize validation loss.
    - Returns: best_model (built with best_hp) and best_hp itself.
    """
    def model_fn(hp):
        return build_model(X_train.shape[1], Y_train.shape[1], hp)

    tuner = kt.RandomSearch(
        model_fn,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='three_state'
    )
    tuner.search(
        X_train, Y_train,
        epochs=20,
        validation_data=(X_val, Y_val),
        verbose=0
    )
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    return best_model, best_hp


def train_and_evaluate(X_train, Y_train, X_test, Y_test, label_names,
                       class_weights=None, out_dir='results'):
    """
    1. Builds an MLPC with default hyperparameters by calling build_model().
    2. Uses EarlyStopping and ReduceLROnPlateau to regularize training.
       - EarlyStopping: stop when validation loss doesn't improve for 5 epochs.
       - ReduceLROnPlateau: halve learning rate when validation loss plateaus for 3 epochs.
    3. Trains on (X_train, Y_train), validates on (X_test, Y_test).
    4. After training:
       - Predict probabilities and derive predicted class indices.
       - Compute confusion matrix (counts + percentages).
       - Plot & save confusion matrix heatmap.
       - Compute AUC (one-vs-rest) per class.
       - Plot & save AUC bar chart.
    Returns trained model, AUC dictionary, and confusion matrix.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build a fresh model
    model = build_model(X_train.shape[1], Y_train.shape[1])

    # Callbacks for regularization
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    # Train (forward + backpropagation over epochs)
    model.fit(
        X_train, Y_train,
        epochs=250,
        batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        class_weight=class_weights,  # handle class imbalance
        verbose=1
    )

    # === Evaluation ===
    probs = model.predict(X_test)          # Softmax outputs: probability for each class
    preds = probs.argmax(axis=1)           # Choose class with highest probability
    true = Y_test.argmax(axis=1)

    # Print counts of true vs predicted labels
    print('True label counts in test set:', pd.Series(true).value_counts())
    print('Predicted label counts:', pd.Series(preds).value_counts())

    # Warn if any class was never predicted
    missing_pred = set(range(len(label_names))) - set(np.unique(preds))
    if missing_pred:
        print(f"Warning: The following classes were never predicted: {missing_pred}")

    # Confusion matrix (counts + percentages)
    cm = confusion_matrix(true, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    print("Confusion Matrix (counts):\n", cm)
    print("Confusion Matrix (percentages):\n", np.round(cm_percent, 2))

    # Plot & save confusion matrix (percentages)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Confusion Matrix')
    plt.colorbar(label='Percentage')
    ticks = range(len(label_names))
    plt.xticks(ticks, label_names, rotation=45)
    plt.yticks(ticks, label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Annotate each cell with the percentage value
    for i in ticks:
        for j in ticks:
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha='center', va='center', color='black')
    cm_path = os.path.join(out_dir, 'confusion_matrix_0.6.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Compute and report AUC per class (one-vs-rest)
    aucs = {}
    for i, label in enumerate(label_names):
        aucs[str(label)] = roc_auc_score((true == i).astype(int), probs[:, i])
    print("AUC per class:", aucs)

    # Plot & save AUC bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(list(aucs.keys()), list(aucs.values()))
    plt.title('One-vs-Rest AUC by Class')
    plt.ylabel('AUC')
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45)
    auc_path = os.path.join(out_dir, 'auc_per_class_0.6.png')
    plt.tight_layout()
    plt.savefig(auc_path)
    plt.close()
    print(f"Saved AUC bar chart to {auc_path}")

    return model, aucs, cm


def analyze_cluster_averages(X, model, label_names, le=None, out_dir=f'/Users/a_fin/Desktop/Year 4/Project/Data/'):
    """
    Perform a cluster-based analysis on all data points (not just train/test).
    1. Predict cluster labels for each row of X.
    2. Attach predicted cluster to X, then compute:
       - Mean, median, and std deviation of neural features per cluster.
       - Bar plots of those statistics.
       - Heatmap of relative feature means across clusters.
       - Identify top features with largest relative differences between clusters.
       - Boxplots showing distribution of features by cluster.
       - Correlation heatmaps within each cluster.
       - Pairwise confusion matrices & difference plots between every pair of clusters.
    3. Save all figures to `out_dir`.
    """
    # Ensure X still has column names
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with original feature names.")

    # Predict cluster indices
    preds = model.predict(X.values).argmax(axis=1)
    X_clustered = X.copy()

    # If a LabelEncoder is provided, invert transform to get original cluster names
    if le is not None:
        cluster_labels = le.inverse_transform(preds)
        X_clustered['Cluster'] = cluster_labels
    else:
        X_clustered['Cluster'] = preds

    # Display cluster counts and diagnostics
    print('Unique predicted indices:', np.unique(preds))
    print('Unique cluster labels (original):', np.unique(X_clustered['Cluster']))
    print('Cluster counts in X_clustered:')
    print(X_clustered['Cluster'].value_counts())

    # Compute summary statistics per cluster
    cluster_averages = X_clustered.groupby('Cluster').mean().sort_index()
    cluster_medians = X_clustered.groupby('Cluster').median().sort_index()
    cluster_std = X_clustered.groupby('Cluster').std().sort_index()

    print("\nCluster Averages:\n", cluster_averages)
    print("\nCluster Medians:\n", cluster_medians)
    print("\nCluster Standard Deviations:\n", cluster_std)

    # 1. Bar plot of average feature values per cluster
    cluster_averages.T.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Neural Features per Cluster')
    plt.xlabel('Neural Feature')
    plt.ylabel('Average Value')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'avg_neural_features.png'))
    plt.close()

    # 2. Bar plot of median feature values per cluster
    cluster_medians.T.plot(kind='bar', figsize=(12, 8), colormap='viridis')
    plt.title('Median Neural Features per Cluster')
    plt.xlabel('Neural Feature')
    plt.ylabel('Median Value')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'median_neural_features.png'))
    plt.close()

    # 3. Bar plot of feature standard deviations per cluster
    cluster_std.T.plot(kind='bar', figsize=(12, 8), colormap='coolwarm')
    plt.title('Standard Deviation of Neural Features per Cluster')
    plt.xlabel('Neural Feature')
    plt.ylabel('Std')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'std_neural_features.png'))
    plt.close()

    # 4. Heatmap of relative feature means (percent of overall mean) by cluster
    rel_cluster_averages = cluster_averages.T.div(cluster_averages.T.mean(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(rel_cluster_averages, annot=True, cmap='vlag', center=100, fmt='.1f')
    plt.title('Heatmap of Feature Means by Cluster (Relative to Feature Mean)')
    plt.xlabel('Cluster')
    plt.ylabel('Neural Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'heatmap_feature_means_by_cluster.png'))
    plt.close()

    # 5. Identify top  features that differ most between clusters (relative percent difference)
    feature_means = cluster_averages.mean(axis=1)
    max_vals = cluster_averages.max(axis=0)
    min_vals = cluster_averages.min(axis=0)
    mean_vals = cluster_averages.mean(axis=0).replace(0, np.nan)
    rel_diffs = ((max_vals - min_vals).abs() / mean_vals.abs()) * 100
    top_features = rel_diffs.sort_values(ascending=False)
    print("\nRelative difference between clusters (%):")
    print(top_features)

    save_path = os.path.join(out_dir, 'relative_feature_difference.png')
    top_features.plot(kind='bar', figsize=(12, 8), color='orange')
    plt.title('Features Differing Between Clusters (Relative %)')
    plt.xlabel('Neural Feature')
    plt.ylabel('Max Relative Difference (%)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved relative feature difference plot to {save_path}")

    # 6. Boxplot of feature distributions by cluster
    plt.figure(figsize=(14, 8))
    X_melt = pd.melt(X_clustered, id_vars=['Cluster'], var_name='Feature', value_name='Value')
    sns.boxplot(x='Cluster', y='Value', data=X_melt, palette='Set3')
    plt.title('Distribution of Neural Features by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Feature Value')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_distribution.png'))
    plt.close()

    # 7. Correlation heatmap per cluster
    for cluster in X_clustered['Cluster'].unique():
        plt.figure(figsize=(10, 8))
        corr = X_clustered[X_clustered['Cluster'] == cluster].drop(columns='Cluster').corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap - Cluster {cluster}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'correlation_cluster_{cluster}.png'))
        plt.close()

    # 8. Pairwise cluster comparisons: confusion matrix & difference plots
    from sklearn.metrics import confusion_matrix
    cluster_labels = list(X_clustered['Cluster'].unique())
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            clust_a = cluster_labels[i]
            clust_b = cluster_labels[j]
            mask = X_clustered['Cluster'].isin([clust_a, clust_b])
            X_pair = X_clustered[mask]
            preds_pair = X_pair['Cluster']
            y_true = preds_pair.values
            y_pred = preds_pair.values  # Labels serve as both true and predicted
            cm = confusion_matrix(y_true, y_pred, labels=[clust_a, clust_b])
            cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

            print(f"\nConfusion Matrix for {clust_a} vs {clust_b} (predicted clusters):\n", cm)
            # Plot confusion matrix (percentages)
            plt.figure(figsize=(5, 4))
            plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
            plt.title(f'Confusion Matrix: {clust_a} vs {clust_b}')
            plt.colorbar(label='Percentage')
            plt.xticks([0, 1], [clust_a, clust_b], rotation=45)
            plt.yticks([0, 1], [clust_a, clust_b])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            for m in range(2):
                for n in range(2):
                    plt.text(n, m, f"{cm_percent[m, n]:.1f}%", ha='center', va='center', color='black')
            plt.tight_layout()
            cm_pair_path = os.path.join(out_dir, f'confusion_matrix_{clust_a}_vs_{clust_b}.png')
            plt.savefig(cm_pair_path)
            plt.close()
            print(f"Saved pairwise confusion matrix to {cm_pair_path}")

            # Compute and plot absolute mean difference
            means = X_pair.groupby('Cluster').mean()
            diff = means.loc[clust_a] - means.loc[clust_b]
            rel_diff = (diff / means.loc[[clust_a, clust_b]].mean()) * 100

            plt.figure(figsize=(10, 6))
            diff.abs().sort_values(ascending=False).plot(kind='bar', color='purple')
            plt.title(f'Absolute Mean Difference: {clust_a} vs {clust_b}')
            plt.ylabel('Absolute Mean Difference')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'mean_diff_{clust_a}_vs_{clust_b}.png'))
            plt.close()

            # Plot relative mean difference (%)
            plt.figure(figsize=(10, 6))
            rel_diff.abs().sort_values(ascending=False).plot(kind='bar', color='teal')
            plt.title(f'Relative Mean Difference (%): {clust_a} vs {clust_b}')
            plt.ylabel('Relative Mean Difference (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'rel_mean_diff_{clust_a}_vs_{clust_b}.png'))
            plt.close()

    return cluster_averages, cluster_medians, cluster_std


def evaluate_and_save(model, X_test, Y_test, label_names, out_dir='results'):
    """
    Given a *trained* model, this function:
    1) Computes predictions on X_test.
    2) Prints confusion matrix (counts + percentages).
    3) Plots & saves the confusion matrix figure.
    4) Computes and prints AUC per class.
    5) Plots & saves the AUC bar chart.
    """
    import os
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1) Predictions
    probs = model.predict(X_test)       # Softmax probabilities
    preds = probs.argmax(axis=1)        # Predicted class indices
    true = Y_test.argmax(axis=1)        # True class indices

    # 2) Confusion matrix (counts + percentage)
    cm = confusion_matrix(true, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    print("Confusion Matrix (counts):\n", cm)
    print("Confusion Matrix (percentages):\n", np.round(cm_pct, 2))

    # Plot & save confusion matrix (%)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_pct, cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Confusion Matrix (%)')
    plt.colorbar(label='Percentage')
    ticks = range(len(label_names))
    plt.xticks(ticks, label_names, rotation=45)
    plt.yticks(ticks, label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Annotate with percentage
    for i in ticks:
        for j in ticks:
            plt.text(j, i, f"{cm_pct[i, j]:.1f}%", ha='center', va='center')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix figure → {cm_path}")

    # 3) AUC per class (one-vs-rest)
    aucs = {
        str(label): roc_auc_score((true == i).astype(int), probs[:, i])
        for i, label in enumerate(label_names)
    }
    print("AUC per class:", aucs)

    # Plot & save AUC bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(list(aucs.keys()), list(aucs.values()))
    plt.title('One-vs-Rest AUC')
    plt.ylabel('AUC')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    auc_path = os.path.join(out_dir, 'auc_per_class.png')
    plt.savefig(auc_path)
    plt.close()
    print(f"Saved AUC bar chart → {auc_path}")

    return cm, aucs


def main():
    """
    Orchestrates the full pipeline:
    1. Load raw data from CSV.
    2. Drop unwanted columns and select features.
    3. Rename columns to meaningful names (e.g., 'psd global offset').
    4. Preprocess (clean, impute, scale, one-hot encode, train/test split).
    5. Compute class weights for imbalanced labels.
    6. Hyperparameter tuning (RandomSearch with keras_tuner).
    7. Retrain best model on full training set.
    8. Evaluate & save confusion matrix + AUC.
    9. Perform cluster-based analysis on entire dataset.
    10. Save final model and label encoder.
    """
    csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete_2.csv'
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number']

    # 1. Load data, drop metadata columns
    X, y = load_data(csv_path=csv_path, drop_cols=drop_cols)

    # 2. Select only features containing 'glob' (global channel metrics)
    X = select_features(X, pattern_keep=['glob_chans'])

    # Keep a copy of original column names
    columns = X.columns

    # 3. Rename columns to shorter, more interpretable feature names
    X['psd global offset']    = X['psd_metrics_combined_avg__Offset_glob_chans']
    X['wSMI 2 global']        = X['wsmi_2_global_2__wSMI_glob_chans']
    X['psd global gamma']     = X['psd_metrics_combined_avg__Gamma_Power_glob_chans']
    X['wSMI 4 global']        = X['wsmi_4_global_2__wSMI_glob_chans']
    X['psd global beta']      = X['psd_metrics_combined_avg__Beta_Power_glob_chans']
    X['psd global exponent']  = X['psd_metrics_combined_avg__Exponent_glob_chans']
    X['wSMI 8 global']        = X['wsmi_8_global_2__wSMI_glob_chans']
    X['psd global delta']     = X['psd_metrics_combined_avg__Delta_Power_glob_chans']
    X['psd global alpha']     = X['psd_metrics_combined_avg__Alpha_Power_glob_chans']
    X['wSMI 1 global']        = X['wsmi_1_global_2__wSMI_glob_chans']
    X['LZC global']           = X['lz_metrics_combined_LZc__glob_chans']
    X['LZsum global']         = X['lz_metrics_combined_LZsum__glob_chans']
    # (Optionally drop SMI columns if not needed)
    # X['SMI 1 global'] = X['wsmi_1_global_2__SMI_glob_chans']
    # X['SMI 2 global'] = X['wsmi_2_global_2__SMI_glob_chans']
    # X['SMI 8 global'] = X['wsmi_8_global_2__SMI_glob_chans']
    # X['SMI 4 global'] = X['wsmi_4_global_2__SMI_glob_chans']
    X['psd global theta']     = X['psd_metrics_combined_avg__Theta_Power_glob_chans']
    X['pe 2 global']          = X['pe_metrics_combined_2_wide__glob_chans']
    X['pe 1 global']          = X['pe_metrics_combined_1_wide__glob_chans']
    X['pe 4 global']          = X['pe_metrics_combined_4_wide__glob_chans']
    X['pe 8 global']          = X['pe_metrics_combined_8_wide__glob_chans']

    # Drop original long column names, keep only the renamed ones
    X.drop(columns=columns, inplace=True)

    # 4. Preprocess: clean, impute, scale, encode labels, split into train/test
    X_train, X_test, Y_train, Y_test, le, label_names = preprocess(X, y)

    # 5. Compute class weights to address imbalanced classes
    from sklearn.utils.class_weight import compute_class_weight
    y_train_labels = np.argmax(Y_train, axis=1)
    class_weights_array = compute_class_weight(
        'balanced', classes=np.unique(y_train_labels), y=y_train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}

    # 6. Hyperparameter tuning
    # Split training set further into a training/validation split
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_train, Y_train, test_size=0.3, random_state=42, stratify=y_train_labels
    )
    best_model, best_hp = tune_hyperparameters(X_tr, Y_tr, X_val, Y_val)

    # 7. Retrain best model on the full training set
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    best_model.fit(
        X_train, Y_train,
        epochs=250,
        batch_size=256,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # 8. Evaluate & save confusion matrix + AUC plots for the tuned model
    cm, aucs = evaluate_and_save(
        best_model,
        X_test, Y_test,
        label_names,
        out_dir=f'/Users/a_fin/Desktop/Year 4/Project/Data/'
    )

    # 9. Cluster analysis on the full cleaned dataset
    #    - Re-scale entire cleaned dataset (to use consistent scaling as training)
    scaler = StandardScaler()
    X_clean = clean_impute(X)
    scaler.fit(X_clean)
    X_scaled = pd.DataFrame(
        scaler.transform(X_clean),
        columns=X_clean.columns, index=X_clean.index
    )
    analyze_cluster_averages(
        X_scaled,
        best_model,
        label_names,
        le=le,
        out_dir=f'/Users/a_fin/Desktop/Year 4/Project/Data/'
    )

    # 10. Save the final tuned model and label encoder for future inference
    best_model.save('neural_decoder_model.h5')
    joblib.dump(le, 'label_encoder.pkl')
    print('Saved tuned model → neural_decoder_model.h5')
    print('Saved label encoder → label_encoder.pkl')


if __name__ == '__main__':
    main()
    # Uncomment the following line to run the script directly