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

from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Scikit-learn utilities for splitting, scaling, encoding labels, and metrics
from sklearn.model_selection import train_test_split, GroupKFold
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


def preprocess(X_train, X_test, y_train, y_test, random_state=42):
    """
    1. Clean and impute X_train and X_test separately.
    2. Align y to the cleaned X indices.
    3. Shuffle training data.
    4. Standardize features: fit on X_train, transform both X_train and X_test.
    5. Label-encode y: fit on y_train, transform both y_train and y_test.
    6. Convert y to one-hot vectors.
    Returns: X_train_scaled, X_test_scaled, Y_train_onehot, Y_test_onehot, label_encoder, classes.
    """
    # Clean and impute training data
    X_train_clean = clean_impute(X_train)
    y_train_clean = y_train.loc[X_train_clean.index]

    # Clean and impute test data
    X_test_clean = clean_impute(X_test)
    if X_test_clean.empty:
        print("Warning: Test set became empty after cleaning. Skipping this fold.")
        # Return empty arrays and a flag to signal skipping
        return None, None, None, None, None, None

    y_test_clean = y_test.loc[X_test_clean.index]

    # Shuffle training data
    X_train_shuffled, y_train_shuffled = shuffle(X_train_clean, y_train_clean, random_state=random_state)

    # Fit scaler ONLY on training data, then transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_shuffled)
    X_test_scaled = scaler.transform(X_test_clean)

    # Fit label encoder ONLY on training labels, then transform both
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_shuffled)
    y_test_enc = le.transform(y_test_clean)

    # Convert to one-hot vectors
    Y_train_onehot = to_categorical(y_train_enc)
    Y_test_onehot = to_categorical(y_test_enc, num_classes=len(le.classes_))

    # Return all processed sets and the fitted encoder
    return X_train_scaled, X_test_scaled, Y_train_onehot, Y_test_onehot, le, le.classes_


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
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
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

    tuner = kt.Hyperband(  # Use Hyperband for more efficient search
        model_fn,
        objective='val_auc',
        max_epochs=50, # max_epochs instead of epochs for Hyperband
        factor=3,
        directory='tuner_dir',
        project_name='three_state_hyperband'
    )

    # Add EarlyStopping to the tuner search
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(
        X_train, Y_train,
        epochs=50,
        validation_data=(X_val, Y_val),
        verbose=1,
        callbacks=[stop_early] # Add callback here
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
    labels = range(len(label_names))
    cm = confusion_matrix(true, preds, labels=labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cm_pct = np.nan_to_num(cm_pct) # Replace NaNs (from division by zero) with 0

    print("Confusion Matrix (counts):\n", cm)
    print("Confusion Matrix (percentages):\n", np.round(cm_pct, 2))

    # Plot & save confusion matrix (%)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_pct, cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Confusion Matrix of MLP-C True label vs Predicted label (%)')
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
    Orchestrates a full cross-validation pipeline:
    1. Load raw data from CSV.
    2. Define features and prepare data.
    3. Use GroupKFold to split data by subject, ensuring no subject leakage.
    4. For each fold:
        a. Preprocess (clean, impute, scale, one-hot encode).
        b. Tune hyperparameters on a validation set split from the training data.
        c. Train the best model on the full training data for that fold.
        d. Evaluate on the test set for that fold and store metrics.
    5. Aggregate and report the average performance (AUC, confusion matrix) across all folds.
    6. Retrain the best overall model on the entire dataset and save it.
    """
    csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete_2.csv'
    
    # 1. Load data and define groups for cross-validation
    df = pd.read_csv(csv_path)
    # Exclude subjects s10 and s19
    subjects_to_exclude = ['s10', 's19']
    df = df[~df['subject'].isin(subjects_to_exclude)]
    print(f"Excluding subjects: {subjects_to_exclude}. Remaining subjects: {df['subject'].unique()}")

    groups = df['subject']  # Use subject IDs for grouped splitting
    y = df['transition_label']
    
    # Define features and drop metadata
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number', 'transition_label']
    X_raw = df.drop(columns=drop_cols, errors='ignore')
    X_raw = select_features(X_raw, pattern_keep=['glob_chans'])

    # Define a function to rename features (for reusability)
    def rename_features(X):
        original_columns = X.columns.tolist()
        # Create new columns with shorter names
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
        X['psd global theta']     = X['psd_metrics_combined_avg__Theta_Power_glob_chans']
        X['pe 2 global']          = X['pe_metrics_combined_2_wide__glob_chans']
        X['pe 1 global']          = X['pe_metrics_combined_1_wide__glob_chans']
        X['pe 4 global']          = X['pe_metrics_combined_4_wide__glob_chans']
        X['pe 8 global']          = X['pe_metrics_combined_8_wide__glob_chans']
        X = X.drop(columns=original_columns)
        return X

    X = rename_features(X_raw)

    # 2. Set up GroupKFold Cross-Validation
    n_splits = len(df['subject'].unique())
    gkf = GroupKFold(n_splits=n_splits)
    
    all_aucs = []
    all_cms = []
    fold_num = 0

    print(f"Starting {n_splits}-fold cross-validation...")

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        fold_num += 1
        print(f"\n===== FOLD {fold_num}/{n_splits} =====")
        
        # 3. Split data for the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        test_subjects = df['subject'].iloc[test_idx].unique()
        print(f"Testing on subjects: {test_subjects}")

        # 4. Preprocess data for the fold
        X_train_p, X_test_p, Y_train_p, Y_test_p, le, label_names = preprocess(
            X_train, X_test, y_train, y_test
        )

        # If preprocessing fails (e.g., empty test set), skip fold
        if X_train_p is None:
            continue

        # 5. Hyperparameter Tuning for the fold
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_train_p, Y_train_p, test_size=0.2, random_state=42, stratify=np.argmax(Y_train_p, axis=1)
        )
        best_model, best_hp = tune_hyperparameters(X_tr, Y_tr, X_val, Y_val)
        print(f"Best hyperparameters for fold {fold_num}: {best_hp.values}")

        # 6. Train the best model on the full training set for this fold
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        best_model.fit(
            X_train_p, Y_train_p,
            epochs=250,
            batch_size=256,
            validation_data=(X_test_p, Y_test_p),
            callbacks=callbacks,
            verbose=0  # Keep training output clean
        )

        # 7. Evaluate and store results for the fold
        cm, aucs = evaluate_and_save(
            best_model, X_test_p, Y_test_p, label_names, out_dir=f'results/fold_{fold_num}'
        )
        all_cms.append(cm)
        all_aucs.append(aucs)

    # 8. Aggregate and report final results
    print("\n===== CROSS-VALIDATION COMPLETE =====")
    
    # Sum confusion matrices for an overall view
    total_cm = np.sum(all_cms, axis=0)
    total_cm_percent = total_cm.astype('float') / total_cm.sum(axis=1, keepdims=True) * 100
    print("Overall Confusion Matrix (counts):\n", total_cm)
    print("Overall Confusion Matrix (percentages):\n", np.round(total_cm_percent, 2))

    # Average AUC scores
    avg_aucs = pd.DataFrame(all_aucs).mean()
    std_aucs = pd.DataFrame(all_aucs).std()
    print("\nAverage AUC per class across all folds:")
    for label in avg_aucs.index:
        print(f"  Class {label}: {avg_aucs[label]:.3f} ± {std_aucs[label]:.3f}")

    # 9. Retrain final model on all data (optional, for deployment)
    print("\nRetraining final model on all data with best hyperparameters from first fold...")
    # Preprocess all data
    X_all_p, _, Y_all_p, _, le, final_label_names = preprocess(X, X, y, y)
    
    # Build model with best HPs found in the first fold (or you could average them)
    final_model = build_model(X_all_p.shape[1], Y_all_p.shape[1], hp=best_hp)
    final_model.fit(X_all_p, Y_all_p, epochs=100, batch_size=256, verbose=0)

    # Evaluate the final model on the full dataset it was trained on
    print("\n===== FINAL MODEL PERFORMANCE (ON ALL DATA) =====")
    evaluate_and_save(
        final_model, X_all_p, Y_all_p, final_label_names, out_dir='results/final_model'
    )
    
    # Save the final model and encoder
    final_model.save('final_neural_decoder_model.h5')
    joblib.dump(le, 'final_label_encoder.pkl')

    # # —————————————————————————————— 8.5 MANUAL PERMUTATION‐IMPORTANCE ——————————————————————————————

    # # Compute baseline accuracy on X_test once (no shuffling).
    # # 1) True labels as integers 0…(num_classes−1)
    # y_test_labels = Y_test.argmax(axis=1)

    # # 2) Baseline predictions & accuracy
    # probs_baseline = best_model.predict(X_test)
    # preds_baseline = probs_baseline.argmax(axis=1)
    # baseline_acc = (preds_baseline == y_test_labels).mean()
    # print(f"Baseline test accuracy (no shuffling): {baseline_acc:.4f}")

    # # 3) Prepare DataFrame form of X_test (so we can shuffle columns by name)
    # feature_names = X_full_renamed.columns.tolist()
    # X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # # 4) For each feature, shuffle it n_repeats times and measure accuracy drop
    # n_repeats = 10
    # perm_importances = np.zeros(len(feature_names))
    # perm_std       = np.zeros(len(feature_names))

    # # Loop over feature indices
    # for j, feat in enumerate(feature_names):
    #     acc_drops = []
    #     for _ in range(n_repeats):
    #         # Make a copy of X_test_df and shuffle **only** column `feat`
    #         X_shuffled = X_test_df.copy()
    #         X_shuffled[feat] = np.random.permutation(X_shuffled[feat].values)

    #         # Predict with the model on the shuffled data
    #         probs_perm = best_model.predict(X_shuffled.values)
    #         preds_perm = probs_perm.argmax(axis=1)
    #         acc_perm   = (preds_perm == y_test_labels).mean()

    #         # Accuracy drop = baseline_acc − acc_perm
    #         acc_drops.append(baseline_acc - acc_perm)

    #     # Store the mean and std of the drops
    #     perm_importances[j] = np.mean(acc_drops)
    #     perm_std[j] = np.std(acc_drops)

    # # 5) Sort features by mean accuracy drop (descending)
    # idx_sorted = np.argsort(perm_importances)[::-1]
    # sorted_feats = [feature_names[i] for i in idx_sorted]
    # sorted_means = perm_importances[idx_sorted]
    # sorted_stds  = perm_std[idx_sorted]

    # # 6) Plot a bar chart of (mean drop ± std) for each feature
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(sorted_feats)), sorted_means, yerr=sorted_stds, align='center')
    # plt.xticks(range(len(sorted_feats)), sorted_feats, rotation=90)
    # plt.ylabel("Mean accuracy drop (± std) after shuffling")
    # plt.title("Permutation‐Importance of Each Feature")
    # plt.tight_layout()

    # perm_path = os.path.join(f'/Users/a_fin/Desktop/Year 4/Project/Data/', 'permutation_importance.png')
    # os.makedirs(os.path.dirname(perm_path), exist_ok=True)
    # plt.savefig(perm_path)
    # plt.close()
    # print(f"Saved manual permutation‐importance plot → {perm_path}")

    # # 7) Optionally, print top 5 features
    # print("Top 5 features by permutation‐importance (manual):")
    # for i in range(min(5, len(sorted_feats))):
    #     print(f"  {i+1}. {sorted_feats[i]} (mean drop = {sorted_means[i]:.4f} ± {sorted_stds[i]:.4f})")

    # # —————————————————————————————————————————————————————————————— 

    #     # —————————————————————————————— 8.75 SHAP EXPLANATIONS (FIXED) ——————————————————————————————

    # # Pick a small background set (e.g. up to 100 random training samples)
    # np.random.seed(42)
    # bg_indices = np.random.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
    # X_background = X_train[bg_indices]

    # # 1) Create the DeepExplainer
    # # If you see a “DeepExplainer with TensorFlow eager mode” error, you can uncomment:
    # #    tf.compat.v1.disable_eager_execution()
    # explainer = shap.DeepExplainer(best_model, X_background)

    # # 2) Select a modest subset of X_test to explain (up to 100 points)
    # test_indices = np.random.choice(X_test.shape[0], size=min(100, X_test.shape[0]), replace=False)
    # X_shap = X_test[test_indices]

    # # 3) Compute SHAP values
    # shap_values = explainer.shap_values(X_shap)

    # # 4) Convert shap_values into a single “mean |SHAP| per feature” vector of length = n_features
    # if isinstance(shap_values, list):
    #     # shap_values is a list of length = n_classes; each entry is (n_samples, n_features)
    #     # Compute per-class mean(|shap|)
    #     mean_abs_per_class = [np.abs(cls_shap).mean(axis=0) for cls_shap in shap_values]
    #     # Now average over classes → result shape = (n_features,)
    #     shap_abs_mean = np.mean(mean_abs_per_class, axis=0)

    # else:
    #     # shap_values is a single ndarray. Possible shapes:
    #     #  - (n_samples, n_features)        (binary or single-output)
    #     #  - (n_classes, n_samples, n_features)   (multi-output packed into one array)
    #     arr = np.array(shap_values)
    #     if arr.ndim == 2:
    #         # Only one output dimension: arr.shape = (n_samples, n_features)
    #         shap_abs_mean = np.abs(arr).mean(axis=0)

    #     elif arr.ndim == 3:
    #         # Either (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes).
    #         # We check which axis corresponds to n_classes by comparing to label_names.
    #         n_classes = len(label_names)
    #         if arr.shape[0] == n_classes and arr.shape[2] != n_classes:
    #             # arr.shape = (n_classes, n_samples, n_features)
    #             # → take absolute, then mean over samples (axis=1), then mean over classes (axis=0)
    #             shap_abs_mean = np.mean(np.abs(arr), axis=(0, 1))
    #         elif arr.shape[2] == n_classes and arr.shape[0] != n_classes:
    #             # arr.shape = (n_samples, n_features, n_classes)
    #             # → take absolute, then mean over classes (axis=2), then mean over samples (axis=0)
    #             temp = np.abs(arr).mean(axis=2)   # shape = (n_samples, n_features)
    #             shap_abs_mean = temp.mean(axis=0) # shape = (n_features,)
    #         else:
    #             raise ValueError(f"Unexpected shap_values.shape = {arr.shape}. Cannot infer class axis.")
    #     else:
    #         raise ValueError(f"Unexpected shap_values.ndim = {arr.ndim}. Expected 2 or 3.")

    # # 5) Now shap_abs_mean is guaranteed to have length = n_features
    # feature_names = X_full_renamed.columns.tolist()
    # if shap_abs_mean.shape[0] != len(feature_names):
    #     raise ValueError(
    #         f"SHAP‐output size mismatch: got {shap_abs_mean.shape[0]} features but expected {len(feature_names)}."
    #     )

    # # 6) Sort features by average |SHAP|
    # idx_shap = np.argsort(shap_abs_mean)[::-1]
    # sorted_feats_shap = [feature_names[i] for i in idx_shap]
    # sorted_shap_vals   = shap_abs_mean[idx_shap]

    # # 7) Plot a bar chart of average |SHAP| per feature
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(feature_names)), sorted_shap_vals, align='center')
    # plt.xticks(range(len(feature_names)), sorted_feats_shap, rotation=90)
    # plt.ylabel("Mean(|SHAP value|)")
    # plt.title("Global SHAP‐Importance (average over all classes)")
    # plt.tight_layout()

    # shap_path = os.path.join(f'/Users/a_fin/Desktop/Year 4/Project/Data/', 'shap_global_importance.png')
    # os.makedirs(os.path.dirname(shap_path), exist_ok=True)
    # plt.savefig(shap_path)
    # plt.close()
    # print(f"Saved SHAP global importance bar chart → {shap_path}")

    # # ——————————————————————————————————————————————————————————————

    # # 9. Cluster analysis on the full cleaned dataset
    # scaler = StandardScaler()
    # X_clean = clean_impute(X_full_renamed)
    # scaler.fit(X_clean)
    # X_scaled = pd.DataFrame(
    #     scaler.transform(X_clean),
    #     columns=X_clean.columns, index=X_clean.index
    # )
    # analyze_cluster_averages(
    #     X_scaled,
    #     best_model,
    #     label_names,
    #     le=le,
    #     out_dir=f'/Users/a_fin/Desktop/Year 4/Project/Data/'
    # )

    # # 10. Save the final tuned model and label encoder for future inference
    # best_model.save('neural_decoder_model.h5')
    # joblib.dump(le, 'label_encoder.pkl')
    # print('Saved tuned model → neural_decoder_model.h5')
    # print('Saved label encoder → label_encoder.pkl')


if __name__ == '__main__':
    main()
    # Uncomment the following line to run the script directly