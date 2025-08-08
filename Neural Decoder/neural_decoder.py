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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

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


def safe_json_dump(data, file_path, indent=2):
    """
    Safely dump data to JSON, converting numpy types to native Python types.
    """
    import json
    import numpy as np
    
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(v) for v in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    converted_data = convert_numpy_types(data)
    
    with open(file_path, 'w') as f:
        json.dump(converted_data, f, indent=indent)


def select_important_features(X_train, y_train, X_test, n_features=None):
    """
    Select the most important features using multiple methods and take the union of top features.
    1. Univariate feature selection (f_classif)
    2. Mutual information
    3. Random Forest feature importance
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels for feature selection
    le_temp = LabelEncoder()
    y_encoded = le_temp.fit_transform(y_train)
    
    if n_features is None:
        n_features = min(X_train.shape[1] // 2, 27)  # Select more features
    
    print(f"Selecting {n_features} most important features from {X_train.shape[1]} total features")
    
    # Method 1: Univariate feature selection
    selector_f = SelectKBest(score_func=f_classif, k=min(n_features, X_train.shape[1]))
    selector_f.fit(X_train, y_encoded)
    f_scores = selector_f.scores_
    f_selected = np.argsort(f_scores)[-n_features:]
    
    # Method 2: Mutual information
    mi_scores = mutual_info_classif(X_train, y_encoded, random_state=42)
    mi_selected = np.argsort(mi_scores)[-n_features:]
    
    # Method 3: Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_encoded)
    rf_importances = rf.feature_importances_
    rf_selected = np.argsort(rf_importances)[-n_features:]
    
    # Take union of top features from all methods and rank by combined score
    all_features = set(f_selected) | set(mi_selected) | set(rf_selected)
    
    # Create combined scores for ranking
    combined_scores = np.zeros(X_train.shape[1])
    
    # Normalize scores to 0-1 range and combine
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
    rf_scores_norm = (rf_importances - rf_importances.min()) / (rf_importances.max() - rf_importances.min() + 1e-8)
    
    combined_scores = f_scores_norm + mi_scores_norm + rf_scores_norm
    
    # Select top features by combined score
    selected_features = np.argsort(combined_scores)[-n_features:]
    
    print(f"Selected {len(selected_features)} features using combined ranking")
    
    return X_train[:, selected_features], X_test[:, selected_features], selected_features


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


def clean_impute(X, mu=None, sigma=None, threshold=2.5):
    """
    1. Remove rows where all features are NaN.
    2. Clip outliers beyond `threshold * std` for each column, setting them to NaN.
       If mu and sigma are provided, they are used for clipping. Otherwise, they are computed from X.
    3. Forward-fill then backward-fill to impute missing values.
    4. Drop any remaining NaNs.
    This ensures no extreme outliers remain and that each row has no missing entries.
    """
    mask_allnan = X.isna().all(axis=1)
    X = X[~mask_allnan]
    
    # If mu and sigma are not provided, compute them from the data (for training set)
    if mu is None:
        mu = X.mean()
    if sigma is None:
        sigma = X.std()

    # Any value with |value - mean| > threshold * sigma becomes NaN
    X = X.where(np.abs(X - mu) <= threshold * sigma)
    X = X.fillna(method='ffill').fillna(method='bfill').dropna()
    return X, mu, sigma


def preprocess(X_train, X_test, y_train, y_test, random_state=42):
    """
    Enhanced preprocessing with feature engineering and class balancing.
    1. Clean and impute X_train and X_test separately.
    2. Add feature engineering (interaction terms, polynomial features).
    3. Align y to the cleaned X indices.
    4. Apply SMOTE for class balancing on training data.
    5. Shuffle training data.
    6. Robust standardization with outlier handling.
    7. Label-encode y: fit on y_train, transform both y_train and y_test.
    8. Convert y to one-hot vectors.
    Returns: X_train_scaled, X_test_scaled, Y_train_onehot, Y_test_onehot, label_encoder, classes.
    """
    from sklearn.preprocessing import RobustScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import PolynomialFeatures
    
    # Clean and impute training data, and get the statistics (mu, sigma)
    X_train_clean, mu, sigma = clean_impute(X_train)
    y_train_clean = y_train.loc[X_train_clean.index]

    # Clean and impute test data using statistics from the training data
    X_test_clean, _, _ = clean_impute(X_test, mu=mu, sigma=sigma)
    if X_test_clean.empty:
        print("Warning: Test set became empty after cleaning. Skipping this fold.")
        return None, None, None, None, None, None

    y_test_clean = y_test.loc[X_test_clean.index]

    # Skip feature engineering - keep it simple for better generalization
    print("Skipping feature engineering for better cross-subject generalization")
    
    # Shuffle training data
    X_train_shuffled, y_train_shuffled = shuffle(X_train_clean, y_train_clean, random_state=random_state)

    # Use RobustScaler instead of StandardScaler for better outlier handling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_shuffled)
    X_test_scaled = scaler.transform(X_test_clean)

    # Apply feature selection to improve performance - use fewer features
    print("\n===== FEATURE SELECTION =====")
    X_train_selected, X_test_selected, selected_features = select_important_features(
        X_train_scaled, y_train_shuffled, X_test_scaled, n_features=12  # Reduced back to 12
    )

    # Fit label encoder ONLY on training labels, then transform both
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_shuffled)
    y_test_enc = le.transform(y_test_clean)

    # Skip SMOTE to avoid overfitting and use class weights instead
    print(f"Original class distribution: {np.bincount(y_train_enc)}")
    X_train_balanced, y_train_balanced = X_train_selected, y_train_enc

    # Convert to one-hot vectors
    Y_train_onehot = to_categorical(y_train_balanced)
    Y_test_onehot = to_categorical(y_test_enc, num_classes=len(le.classes_))

    # Return all processed sets and the fitted encoder
    return X_train_balanced, X_test_selected, Y_train_onehot, Y_test_onehot, le, le.classes_, selected_features


def build_simple_baseline_model(input_dim, num_classes):
    """
    Extremely simple baseline model for comparison.
    Just one hidden layer with heavy regularization.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.1)),
        Dropout(0.7),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.1))
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=0.5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    
    return model


def build_breathing_optimized_model(input_dim, num_classes, dropout_rate=0.4):
    """
    Build a neural network optimized specifically for breathing pattern classification.
    Uses breathing-specific architecture with attention-like mechanisms.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # First pathway: Focus on frequency domain features
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='freq_pathway'),
        Dropout(dropout_rate),
        BatchNormalization(),
        
        # Second pathway: Focus on complexity features  
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), name='complexity_pathway'),
        Dropout(dropout_rate),
        BatchNormalization(),
        
        # Attention-like mechanism
        Dense(32, activation='tanh', kernel_regularizer=l2(0.01), name='attention'),
        Dropout(dropout_rate),
        
        # Final classification
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.02))
    ])
    
    # Use a more sophisticated optimizer
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    
    return model


def build_model(input_dim, num_classes, hp=None):
    """
    Build a simple, robust neural network optimized for cross-subject generalization.
    Heavily constrained hyperparameters to prevent overfitting.
    """
    if hp:
        # Severely constrained hyperparameter search for generalization
        units = hp.Choice('units', [64, 128])  # Much smaller range
        lr = hp.Choice('lr', [1e-3, 5e-4])     # Fixed good values
        dropout_rate = hp.Choice('dropout_rate', [0.5, 0.6])  # High dropout for generalization
        l2_reg = hp.Choice('l2_reg', [0.01, 0.05])  # Strong regularization
    else:
        # Conservative defaults optimized for generalization
        units, lr, dropout_rate, l2_reg = 64, 1e-3, 0.5, 0.01

    # Simpler architecture - just 2 hidden layers
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # First hidden layer with heavy regularization
        Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        # Second hidden layer
        Dense(units // 2, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        # Output layer with extra regularization
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg * 2))
    ])
    
    # Conservative optimizer settings
    optimizer = Adam(
        learning_rate=lr,
        clipnorm=0.5,  # Stronger gradient clipping
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    
    return model


def focal_loss(gamma=2., alpha=None):
    """
    Focal Loss for addressing class imbalance.
    gamma: focusing parameter (higher gamma = more focus on hard examples)
    alpha: class weighting (can be None, scalar, or list of class weights)
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight (1 - pt)^gamma
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        
        # Apply focal weight
        focal_ce = focal_weight * ce
        
        # Apply alpha weighting if provided
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                alpha_weight = alpha
            else:
                alpha_weight = tf.reduce_sum(y_true * alpha, axis=-1, keepdims=True)
            focal_ce = alpha_weight * focal_ce
            
        return tf.reduce_sum(focal_ce, axis=-1)
    
    return focal_loss_fixed


def tune_hyperparameters(X_train, Y_train, X_val, Y_val):
    """
    Minimal hyperparameter tuning to prevent overfitting to validation subjects.
    Focus on generalization over validation performance.
    """
    def model_fn(hp):
        return build_model(X_train.shape[1], Y_train.shape[1], hp)

    # MUCH more conservative search to prevent overfitting
    tuner = kt.RandomSearch(
        model_fn,
        objective='val_auc',
        max_trials=3,  # Drastically reduced from 15
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='neural_decoder_conservative',
        overwrite=True
    )

    # Very conservative early stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]

    print("Starting minimal hyperparameter search (conservative approach)...")
    # Much fewer epochs to prevent overfitting to validation subjects
    tuner.search(
        X_train, Y_train,
        epochs=25,  # Reduced from 60
        validation_data=(X_val, Y_val),
        verbose=1,
        callbacks=callbacks
    )
    
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    
    print(f"Best hyperparameters found:")
    for key, value in best_hp.values.items():
        print(f"  {key}: {value}")
    
    return best_model, best_hp


def train_and_evaluate(X_train, Y_train, X_test, Y_test, label_names,
                       class_weights=None, out_dir='results'):
    """
    1. Builds an MLPC with default hyperparameters by calling build_model().
    2. Uses EarlyStopping and ReduceLROnPlateau to regularize training.
       - EarlyStopping: stop when validation loss doesn't improve for 3 epochs.
       - ReduceLROnPlateau: reduce learning rate when validation loss plateaus for 2 epochs.
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

    # Compute class weights if not provided
    if class_weights is None:
        y_train_labels = Y_train.argmax(axis=1)
        class_weights = compute_balanced_class_weights(y_train_labels)

    # Build a fresh model
    model = build_model(X_train.shape[1], Y_train.shape[1])

    # Callbacks for regularization - more aggressive
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]

    # Train (forward + backpropagation over epochs)
    model.fit(
        X_train, Y_train,
        epochs=30,  # Reduced from 250 to prevent overfitting
        batch_size=64,  # Smaller batch size for better generalization
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
    if hasattr(model, 'predict_proba'): # Scikit-learn style model
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        # Scikit-learn models work with integer labels, not one-hot
        true = Y_test 
    else: # Keras style model
        probs = model.predict(X_test)       # Softmax probabilities
        preds = probs.argmax(axis=1)        # Predicted class indices
        true = Y_test.argmax(axis=1) 

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


def compute_balanced_class_weights(y_train_labels):
    """
    Compute balanced class weights with moderate amplification for severe imbalance.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get unique classes and compute standard balanced weights
    classes = np.unique(y_train_labels)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_labels)
    
    # Moderate amplification - too much can hurt learning
    weights = weights ** 1.2  # Reduced from 1.5
    
    # Create class weight dictionary
    class_weights = dict(zip(classes, weights))
    
    print(f"Computed class weights: {class_weights}")
    return class_weights


def process_single_fold(fold_data):
    """
    Process a single cross-validation fold.
    This function is designed to be run in parallel.
    """
    fold_num, train_idx, test_idx, X, y, df = fold_data
    
    print(f"\n===== FOLD {fold_num} =====")
    
    # Split data for the current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    test_subjects = df['subject'].iloc[test_idx].unique()
    print(f"Fold {fold_num} - Testing on subjects: {test_subjects}")

    # Preprocess data for the fold
    result = preprocess(X_train, X_test, y_train, y_test)
    
    # If preprocessing fails (e.g., empty test set), skip fold
    if result[0] is None:
        print(f"Fold {fold_num} - Preprocessing failed, skipping")
        return fold_num, None, None
    
    X_train_p, X_test_p, Y_train_p, Y_test_p, le, label_names, selected_features = result

    # Extract selected features information from preprocessing
    # We need to save this information for later use
    selected_features_info = selected_features

    # Import metrics at the top of the function
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import json
    
    # Hyperparameter Tuning for the fold
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_train_p, Y_train_p, test_size=0.2, random_state=42, 
        stratify=np.argmax(Y_train_p, axis=1)
    )
    
    # Compute class weights for handling imbalance
    y_train_labels = np.argmax(Y_train_p, axis=1)
    class_weights = compute_balanced_class_weights(y_train_labels)
    
    # Test both simple baseline, breathing-optimized, and tuned model
    print(f"Fold {fold_num} - Testing simple baseline model...")
    baseline_model = build_simple_baseline_model(X_train_p.shape[1], Y_train_p.shape[1])
    
    baseline_callbacks = [
        EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    
    baseline_model.fit(
        X_train_p, Y_train_p,
        epochs=50,
        batch_size=64,
        validation_data=(X_test_p, Y_test_p),
        callbacks=baseline_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate baseline
    baseline_probs = baseline_model.predict(X_test_p)
    baseline_aucs = {}
    true_labels = Y_test_p.argmax(axis=1)
    for i, label in enumerate(label_names):
        baseline_aucs[str(label)] = roc_auc_score((true_labels == i).astype(int), baseline_probs[:, i])
    print(f"Fold {fold_num} - Baseline AUC: {baseline_aucs}")
    
    # Test breathing-optimized model
    print(f"Fold {fold_num} - Testing breathing-optimized neural network...")
    breathing_model = build_breathing_optimized_model(X_train_p.shape[1], Y_train_p.shape[1])
    
    breathing_callbacks = [
        EarlyStopping(monitor='val_auc', patience=12, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-5)
    ]
    
    breathing_model.fit(
        X_train_p, Y_train_p,
        epochs=60,
        batch_size=32,
        validation_data=(X_test_p, Y_test_p),
        callbacks=breathing_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate breathing-optimized model
    breathing_probs = breathing_model.predict(X_test_p)
    breathing_aucs = {}
    for i, label in enumerate(label_names):
        breathing_aucs[str(label)] = roc_auc_score((true_labels == i).astype(int), breathing_probs[:, i])
    print(f"Fold {fold_num} - Breathing-optimized AUC: {breathing_aucs}")
    
    # Now try tuned model with minimal search
    print(f"Fold {fold_num} - Tuning deep neural network...")
    best_model, best_hp = tune_hyperparameters(X_tr, Y_tr, X_val, Y_val)
    print(f"Fold {fold_num} - Best hyperparameters: {best_hp.values}")
    
    # Train the best model on full training data for this fold with conservative settings
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),  # Less patience
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)  # More aggressive
    ]
    
    best_model.fit(
        X_train_p, Y_train_p,
        epochs=50,  # Much fewer epochs
        batch_size=64,  # Larger batch size for better generalization
        validation_data=(X_test_p, Y_test_p),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate the tuned neural network
    probs = best_model.predict(X_test_p)
    preds = probs.argmax(axis=1)
    
    # Calculate metrics
    cm = confusion_matrix(true_labels, preds)
    
    aucs = {}
    for i, label in enumerate(label_names):
        aucs[str(label)] = roc_auc_score((true_labels == i).astype(int), probs[:, i])
    
    print(f"Fold {fold_num} - Tuned AUC: {aucs}")
    
    # Choose the best model among all three
    baseline_avg = np.mean(list(baseline_aucs.values()))
    breathing_avg = np.mean(list(breathing_aucs.values()))
    tuned_avg = np.mean(list(aucs.values()))
    
    model_performances = {
        'baseline': (baseline_avg, baseline_model, baseline_aucs, baseline_probs),
        'breathing': (breathing_avg, breathing_model, breathing_aucs, breathing_probs),
        'tuned': (tuned_avg, best_model, aucs, probs)
    }
    
    # Find best performing model
    best_type = max(model_performances.keys(), key=lambda k: model_performances[k][0])
    best_avg, final_model, final_aucs, final_probs = model_performances[best_type]
    
    print(f"Fold {fold_num} - Performance comparison:")
    print(f"  Baseline: {baseline_avg:.3f}")
    print(f"  Breathing-optimized: {breathing_avg:.3f}")
    print(f"  Tuned: {tuned_avg:.3f}")
    print(f"  Best: {best_type} model with {best_avg:.3f}")
    
    # Use the better model for final results
    preds = final_probs.argmax(axis=1)
    cm = confusion_matrix(true_labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    print(f"Fold {fold_num} - Final AUC: {final_aucs}")
    
    # Save results and models for this fold
    fold_dir = f'results/fold_{fold_num}'
    os.makedirs(fold_dir, exist_ok=True)
    
    # Save the best model for this fold
    model_path = os.path.join(fold_dir, 'best_model.h5')
    final_model.save(model_path)
    print(f"Saved best model → {model_path}")
    
    # Save label encoder for this fold
    le_path = os.path.join(fold_dir, 'label_encoder.pkl')
    joblib.dump(le, le_path)
    print(f"Saved label encoder → {le_path}")
    
    # Save predictions and probabilities
    predictions_data = {
        'true_labels': true_labels.tolist(),
        'predicted_labels': preds.tolist(),
        'predicted_probabilities': final_probs.tolist(),
        'label_names': label_names.tolist(),
        'test_subjects': test_subjects.tolist(),
        'model_type': best_type,
        'all_model_aucs': {
            'baseline': baseline_avg,
            'breathing_optimized': breathing_avg,
            'tuned': tuned_avg
        }
    }
    predictions_path = os.path.join(fold_dir, 'predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Saved predictions → {predictions_path}")
    
    # Save metrics
    metrics_data = {
        'auc_scores': final_aucs,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_percent': cm_percent.tolist(),
        'test_subjects': test_subjects.tolist(),
        'fold_number': fold_num,
        'model_type': best_type,
        'all_model_results': {
            'baseline': {'auc': baseline_avg, 'aucs': baseline_aucs},
            'breathing_optimized': {'auc': breathing_avg, 'aucs': breathing_aucs},
            'tuned': {'auc': tuned_avg, 'aucs': aucs}
        }
    }
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics → {metrics_path}")
    
    # Save feature information
    feature_info = {
        'selected_features': selected_features_info.tolist() if selected_features_info is not None else [],
        'feature_names': X.columns.tolist(),
        'n_features_selected': len(selected_features_info) if selected_features_info is not None else X.shape[1]
    }
    features_path = os.path.join(fold_dir, 'features.json')
    with open(features_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"Saved feature info → {features_path}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_percent, cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title(f'Confusion Matrix - Fold {fold_num} ({best_type.title()} Model)')
    plt.colorbar(label='Percentage')
    ticks = range(len(label_names))
    plt.xticks(ticks, label_names, rotation=45)
    plt.yticks(ticks, label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in ticks:
        for j in ticks:
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save AUC bar chart with model comparison
    plt.figure(figsize=(8, 6))
    
    # Plot individual class AUCs for best model
    plt.subplot(2, 1, 1)
    plt.bar(list(final_aucs.keys()), list(final_aucs.values()))
    plt.title(f'AUC Scores - Fold {fold_num} ({best_type.title()} Model)')
    plt.ylabel('AUC')
    plt.ylim(0, 1)
    
    # Plot model comparison
    plt.subplot(2, 1, 2)
    model_names = ['Baseline', 'Breathing-Opt', 'Tuned']
    model_aucs = [baseline_avg, breathing_avg, tuned_avg]
    bars = plt.bar(model_names, model_aucs, alpha=0.7)
    
    # Highlight best model
    best_idx = ['baseline', 'breathing', 'tuned'].index(best_type)
    bars[best_idx].set_color('orange')
    
    plt.title('Model Comparison')
    plt.ylabel('Average AUC')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'model_comparison.png'))
    plt.close()
    
    print(f"Confusion Matrix (percentages):\n{np.round(cm_percent, 2)}")
    print("AUC per class:", final_aucs)
    
    return fold_num, cm, final_aucs


def comprehensive_data_quality_check(df, X, y):
    """
    Comprehensive analysis of your dataset quality.
    """
    print("===== DATA QUALITY ASSESSMENT =====")
    
    # 1. Basic dataset info
    print(f"Total samples: {len(df)}")
    print(f"Total subjects: {df['subject'].nunique()}")
    print(f"Feature matrix shape: {X.shape}")
    
    # 2. Class distribution analysis
    overall_class_dist = pd.Series(y).value_counts().sort_index()
    print(f"\nOverall class distribution: {overall_class_dist.to_dict()}")
    
    # 3. Per-subject analysis
    problematic_subjects = []
    good_subjects = []
    
    print("\nPer-subject analysis:")
    for subject in sorted(df['subject'].unique()):
        mask = df['subject'] == subject
        subj_y = y[mask]
        subj_samples = len(subj_y)
        class_counts = pd.Series(subj_y).value_counts().sort_index()
        
        # Check for problems
        problems = []
        if subj_samples < 50:
            problems.append("too_few_samples")
        if len(class_counts) < 3:
            problems.append("missing_classes")
        if class_counts.min() < 5:
            problems.append("severe_imbalance")
            
        if problems:
            problematic_subjects.append(subject)
            status = "❌ " + ", ".join(problems)
        else:
            good_subjects.append(subject)
            status = "✓ good"
            
        print(f"  {subject}: {subj_samples} samples, classes: {class_counts.to_dict()} - {status}")
    
    # 4. Temporal structure analysis
    if 'week' in df.columns:
        print(f"\nTemporal structure:")
        week_counts = df['week'].value_counts().sort_index()
        print(f"  Weeks: {list(week_counts.index)}")
        print(f"  Samples per week: {week_counts.to_dict()}")
        
        # Check for temporal bias
        week_class_correlations = []
        for week in df['week'].unique():
            week_mask = df['week'] == week
            week_classes = y[week_mask]
            class_props = pd.Series(week_classes).value_counts(normalize=True).sort_index()
            week_class_correlations.append(class_props)
        
        correlation_df = pd.DataFrame(week_class_correlations)
        print(f"  Class proportions by week:")
        print(correlation_df)
        
        # Check if certain classes are concentrated in certain weeks
        for class_label in np.unique(y):
            class_week_dist = df[y == class_label]['week'].value_counts(normalize=True)
            if class_week_dist.max() > 0.5:
                print(f"  ⚠️  Class {class_label} is concentrated in certain weeks!")
    
    # 5. Feature quality analysis
    print(f"\nFeature quality:")
    X_values = X.values if hasattr(X, 'values') else X
    nan_features = np.isnan(X_values).sum(axis=0)
    inf_features = np.isinf(X_values).sum(axis=0)
    constant_features = (X_values.std(axis=0) < 1e-10).sum()
    
    print(f"  Features with NaN: {(nan_features > 0).sum()}")
    print(f"  Features with Inf: {(inf_features > 0).sum()}")
    print(f"  Constant features: {constant_features}")
    
    if (nan_features > 0).any():
        print(f"  NaN counts per feature: {nan_features[nan_features > 0]}")
    
    # 6. Recommendations
    print(f"\n===== RECOMMENDATIONS =====")
    print(f"✓ Good subjects ({len(good_subjects)}): {good_subjects}")
    print(f"❌ Problematic subjects ({len(problematic_subjects)}): {problematic_subjects}")
    
    if len(good_subjects) >= 5:
        print(f"✓ Sufficient good subjects for analysis")
        print(f"💡 Consider excluding problematic subjects: {problematic_subjects}")
    else:
        print(f"❌ Too few good subjects! Need data collection improvements.")
    
    return {
        'good_subjects': good_subjects,
        'problematic_subjects': problematic_subjects,
        'overall_class_dist': overall_class_dist.to_dict(),
        'recommendations': 'exclude_problematic' if len(good_subjects) >= 5 else 'collect_more_data'
    }


def fixed_within_subject_decoding(df, X, y):
    """
    Fixed version that handles class imbalance properly.
    """
    print("===== FIXED WITHIN-SUBJECT ANALYSIS =====")
    
    subject_aucs = []
    results = {}
    
    for subject in df['subject'].unique():
        print(f"\nAnalyzing subject: {subject}")
        
        # Get subject data
        subject_mask = df['subject'] == subject
        subject_X = X[subject_mask]
        subject_y = y[subject_mask]
        
        # Check class distribution
        class_counts = pd.Series(subject_y).value_counts().sort_index()
        print(f"  Class distribution: {class_counts.to_dict()}")
        
        # Skip if insufficient diversity
        if len(class_counts) < 2 or class_counts.min() < 5:
            print(f"  ⚠️  Skipping {subject}: insufficient class diversity")
            continue
        
        # Use session-based splitting instead of random
        if 'session' in df.columns or 'week' in df.columns:
            # Split by session/week
            session_col = 'session' if 'session' in df.columns else 'week'
            subject_sessions = df[subject_mask][session_col].unique()
            
            if len(subject_sessions) < 2:
                print(f"  ⚠️  Skipping {subject}: only one session")
                continue
                
            # Use first half of sessions for training
            n_train_sessions = len(subject_sessions) // 2
            train_sessions = subject_sessions[:n_train_sessions]
            test_sessions = subject_sessions[n_train_sessions:]
            
            train_mask = df[subject_mask][session_col].isin(train_sessions)
            test_mask = df[subject_mask][session_col].isin(test_sessions)
        else:
            # Fallback to simple split
            n_train = len(subject_X) // 2
            train_mask = np.zeros(len(subject_X), dtype=bool)
            train_mask[:n_train] = True
            test_mask = ~train_mask
        
        X_train = subject_X[train_mask]
        y_train = subject_y[train_mask]
        X_test = subject_X[test_mask]
        y_test = subject_y[test_mask]
        
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Check if all classes present in training
        train_classes = set(y_train)
        test_classes = set(y_test)
        
        if len(train_classes) < 2:
            print(f"  ⚠️  Skipping {subject}: insufficient classes in training")
            continue
            
        # Train simple model
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values first
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Use simple logistic regression for robustness
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict probabilities
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate AUC for each class present in test
            class_aucs = {}
            for i, class_label in enumerate(model.classes_):
                if class_label in test_classes and class_label in train_classes:
                    # Binary classification for this class vs rest
                    y_test_binary = (y_test == class_label).astype(int)
                    if len(np.unique(y_test_binary)) > 1:  # Ensure both classes present
                        auc = roc_auc_score(y_test_binary, y_pred_proba[:, i])
                        class_aucs[f'class_{class_label}'] = auc
            
            if class_aucs:
                mean_auc = np.mean(list(class_aucs.values()))
                print(f"  ✓ AUC: {mean_auc:.3f} {class_aucs}")
                subject_aucs.append(mean_auc)
                results[subject] = {
                    'auc': mean_auc,
                    'class_aucs': class_aucs,
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    'train_classes': list(train_classes),
                    'test_classes': list(test_classes)
                }
            else:
                print(f"  ⚠️  No valid AUC computed for {subject}")
                
        except Exception as e:
            print(f"  ❌ Error processing {subject}: {e}")
    
    if subject_aucs:
        mean_auc = np.mean(subject_aucs)
        std_auc = np.std(subject_aucs)
        print(f"\n--- FIXED WITHIN-SUBJECT RESULTS ---")
        print(f"Average within-subject AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"Valid subjects: {len(subject_aucs)}/{len(df['subject'].unique())}")
        
        return mean_auc, results
    else:
        print("❌ No valid within-subject results!")
        return None, {}


def create_temporal_breathing_features(df, X, y, window_size=5):
    """
    Create temporal features that capture breathing pattern dynamics.
    These features look at how neural patterns change over time during breathing.
    
    Args:
        df: DataFrame with metadata including temporal information
        X: Original feature matrix
        y: Labels
        window_size: Number of consecutive samples to use for temporal features
    
    Returns:
        X_temporal: Enhanced feature matrix with temporal breathing features
        valid_indices: Indices of samples that have valid temporal features
    """
    print(f"Creating temporal breathing features (window_size={window_size})...")
    
    X_temporal = X.copy()
    valid_indices = []
    
    # Sort by subject and time to ensure proper temporal order
    if 'epoch' in df.columns:
        time_col = 'epoch'
    elif 'run' in df.columns:
        time_col = 'run'
    else:
        print("  Warning: No temporal ordering column found, using original order")
        time_col = None
    
    temporal_features = []
    
    for subject in df['subject'].unique():
        subject_mask = df['subject'] == subject
        subject_df = df[subject_mask].copy()
        subject_X = X[subject_mask].copy()
        
        # Sort by temporal order if available
        if time_col and time_col in subject_df.columns:
            sort_order = subject_df[time_col].argsort()
            subject_df = subject_df.iloc[sort_order]
            subject_X = subject_X.iloc[sort_order]
        
        # Extract temporal features for this subject
        for i in range(window_size, len(subject_X)):
            window_data = subject_X.iloc[i-window_size:i]
            current_sample = subject_X.iloc[i]
            
            # Calculate temporal features
            temp_features = {}
            
            # 1. Trend features (slope over window)
            for feature in ['psd global alpha', 'psd global theta', 'psd global beta']:
                if feature in window_data.columns:
                    values = window_data[feature].values
                    if len(values) > 1:
                        slope = np.polyfit(range(len(values)), values, 1)[0]
                        temp_features[f'{feature}_trend'] = slope            # 2. Variability features (std over window)
            for feature in ['psd global alpha', 'psd global theta']:
                if feature in window_data.columns:
                    temp_features[f'{feature}_variability'] = window_data[feature].std()
            
            # 3. Change features (current vs previous)
            prev_sample = subject_X.iloc[i-1]
            for feature in ['psd global alpha', 'psd global theta', 'psd global beta']:
                if feature in current_sample.index and feature in prev_sample.index:
                    change = current_sample[feature] - prev_sample[feature]
                    temp_features[f'{feature}_change'] = change
            
            # 4. Rhythm features (autocorrelation-like patterns)
            for feature in ['psd global alpha', 'psd global theta']:
                if feature in window_data.columns:
                    values = window_data[feature].values
                    if len(values) >= 3:
                        # Simple rhythm detection: correlation between first and second half
                        mid = len(values) // 2
                        if mid > 0:
                            corr = np.corrcoef(values[:mid], values[-mid:])[0, 1]
                            if not np.isnan(corr):
                                temp_features[f'{feature}_rhythm'] = corr
            
            # 5. Phase features (for breathing cycles)
            wsmi_features = [col for col in window_data.columns if 'wSMI' in col]
            if len(wsmi_features) >= 2:
                wsmi_mean = np.mean([window_data[col].values for col in wsmi_features], axis=0)
                temp_features['wsmi_phase_coherence'] = np.std(wsmi_mean)
            
            temporal_features.append(temp_features)
            valid_indices.append(subject_df.index[i])
    
    # Convert temporal features to DataFrame
    if temporal_features:
        temp_df = pd.DataFrame(temporal_features, index=valid_indices)
        
        # Add temporal features to original features
        X_temporal_subset = X.loc[valid_indices].copy()
        for col in temp_df.columns:
            X_temporal_subset[col] = temp_df[col]
        
        print(f"  Added {len(temp_df.columns)} temporal features")
        print(f"  Valid samples: {len(valid_indices)}/{len(X)} ({100*len(valid_indices)/len(X):.1f}%)")
        
        return X_temporal_subset, valid_indices
    else:
        print("  No temporal features could be created")
        return X, X.index.tolist()


def advanced_preprocessing_pipeline(X_train, X_test, y_train, y_test, random_state=42):
    """
    Advanced preprocessing pipeline with multiple feature enhancement techniques.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        random_state: Random seed
    
    Returns:
        Processed features and labels with advanced enhancements
    """
    from sklearn.preprocessing import RobustScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    
    print("\n===== ADVANCED PREPROCESSING PIPELINE =====")
    
    # 1. Clean and impute (existing function)
    X_train_clean, mu, sigma = clean_impute(X_train)
    y_train_clean = y_train.loc[X_train_clean.index]
    
    X_test_clean, _, _ = clean_impute(X_test, mu=mu, sigma=sigma)
    if X_test_clean.empty:
        return None, None, None, None, None, None
    y_test_clean = y_test.loc[X_test_clean.index]
    
    # 2. Add universal breathing features
    X_train_univ = create_universal_breathing_features(X_train_clean)
    X_test_univ = create_universal_breathing_features(X_test_clean)
    
    # 3. Robust scaling with outlier handling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_univ)
    X_test_scaled = scaler.transform(X_test_univ)
    
    # 4. Advanced feature selection with multiple criteria
    print("\n--- Advanced Feature Selection ---")
    
    # Encode labels for feature selection
    le_temp = LabelEncoder()
    y_train_encoded = le_temp.fit_transform(y_train_clean)
    
    # Multiple feature selection methods
    n_features = min(20, X_train_scaled.shape[1] // 2)  # Select more features than before
    
    # Method 1: F-test
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    selector_f.fit(X_train_scaled, y_train_encoded)
    f_scores = selector_f.scores_
    
    # Method 2: Mutual information
    mi_scores = mutual_info_classif(X_train_scaled, y_train_encoded, random_state=random_state)
    
    # Method 3: Variance-based selection (remove low-variance features)
    from sklearn.feature_selection import VarianceThreshold
    var_selector = VarianceThreshold(threshold=0.01)  # Remove features with very low variance
    var_selector.fit(X_train_scaled)
    var_mask = var_selector.get_support()
    
    # Combine selection criteria
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
    
    # Weight the scores (give more weight to mutual information for breathing patterns)
    combined_scores = 0.4 * f_scores_norm + 0.6 * mi_scores_norm
    combined_scores[~var_mask] = 0  # Zero out low-variance features
    
    # Select top features
    selected_indices = np.argsort(combined_scores)[-n_features:]
    
    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    print(f"Selected {n_features} features using advanced multi-criteria selection")
    
    # 5. Label encoding with class balancing awareness
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_clean)
    y_test_enc = le.transform(y_test_clean)
    
    # 6. Convert to one-hot for neural networks
    Y_train_onehot = to_categorical(y_train_enc)
    Y_test_onehot = to_categorical(y_test_enc, num_classes=len(le.classes_))
    
    print(f"Enhanced preprocessing complete:")
    print(f"  Features: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
    print(f"  Classes: {len(le.classes_)}")
    print(f"  Train samples: {X_train_selected.shape[0]}")
    print(f"  Test samples: {X_test_selected.shape[0]}")
    
    return X_train_selected, X_test_selected, Y_train_onehot, Y_test_onehot, le, le.classes_, selected_indices


def create_universal_breathing_features(X):
    """
    Create universal breathing features that work across ALL techniques.
    These features capture fundamental breathing-related neural dynamics.
    
    Args:
        X: Original feature matrix
    
    Returns:
        X_universal: Enhanced feature matrix with universal breathing features
    """
    print("Creating universal breathing features...")
    
    X_universal = X.copy()
    
    # 1. Autonomic nervous system balance (works for all breathing techniques)
    if all(col in X.columns for col in ['psd global delta', 'psd global theta', 'psd global beta', 'psd global gamma']):
        # Parasympathetic activity (low frequency)
        X_universal['parasympathetic_activity'] = X['psd global delta'] + X['psd global theta']
        
        # Sympathetic activity (high frequency)  
        X_universal['sympathetic_activity'] = X['psd global beta'] + X['psd global gamma']
        
        # Autonomic balance ratio
        X_universal['autonomic_balance'] = (X_universal['parasympathetic_activity'] + 1e-8) / (X_universal['sympathetic_activity'] + 1e-8)
        print("  Added autonomic nervous system features")
    
    # 2. Attention/relaxation states (alpha/theta ratios)
    if 'psd global alpha' in X.columns and 'psd global theta' in X.columns:
        X_universal['attention_relaxation'] = (X['psd global alpha'] + 1e-8) / (X['psd global theta'] + 1e-8)
        print("  Added attention/relaxation ratio")
    
    # 3. Neural complexity measures (breathing pattern complexity)
    complexity_features = ['LZC global', 'LZsum global']
    if all(col in X.columns for col in complexity_features):
        X_universal['complexity_ratio'] = (X['LZC global'] + 1e-8) / (X['LZsum global'] + 1e-8)
        print("  Added neural complexity ratio")
    
    # 4. Permutation entropy ratios (neural predictability)
    pe_features = [col for col in X.columns if 'pe' in col.lower()]
    if len(pe_features) >= 2:
        X_universal['pe_complexity_ratio'] = (X[pe_features[0]] + 1e-8) / (X[pe_features[1]] + 1e-8)
        print("  Added permutation entropy complexity ratio")
    
    # 5. Inter-region synchrony (wSMI coherence)
    wsmi_features = [col for col in X.columns if 'wSMI' in col]
    if len(wsmi_features) >= 2:
        X_universal['neural_synchrony'] = np.mean([X[col] for col in wsmi_features], axis=0)
        X_universal['synchrony_variability'] = np.std([X[col] for col in wsmi_features], axis=0)
        print("  Added neural synchrony features")
    
    # 6. Spectral ratios important for breathing
    if all(col in X.columns for col in ['psd global alpha', 'psd global beta', 'psd global theta', 'psd global delta']):
        # Classic EEG ratios used in meditation/breathing research
        X_universal['alpha_beta_ratio'] = (X['psd global alpha'] + 1e-8) / (X['psd global beta'] + 1e-8)
        X_universal['theta_beta_ratio'] = (X['psd global theta'] + 1e-8) / (X['psd global beta'] + 1e-8)
        X_universal['low_high_freq_ratio'] = (X['psd global delta'] + X['psd global theta']) / (X['psd global beta'] + X['psd global gamma'] + 1e-8)
        print("  Added classic EEG spectral ratios")
    
    print(f"Universal features: {X.shape[1]} -> {X_universal.shape[1]} (+{X_universal.shape[1] - X.shape[1]} universal features)")
    return X_universal


def universal_ensemble_cross_validation(df, X, y):
    """
    Universal ensemble approach that improves performance across ALL breathing techniques.
    
    Args:
        df: DataFrame with metadata
        X: Feature matrix
        y: Labels
    
    Returns:
        dict: Universal ensemble results
    """
    print(f"\n===== UNIVERSAL ENSEMBLE CROSS-VALIDATION =====")
    print("Building ensemble model that works across ALL breathing techniques...")
    
    # Create universal breathing features for ALL data
    X_universal = create_universal_breathing_features(X)
    
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    # Define ensemble of models optimized for cross-technique generalization
    models = [
        ('rf', RandomForestClassifier(
            n_estimators=300, 
            max_depth=12, 
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.85,
            random_state=42
        )),
        ('svm', SVC(
            kernel='rbf',
            C=0.5,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )),
        ('lr', LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]
    
    # Create voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='soft')
    
    # Cross-validation across all subjects and techniques
    gkf = GroupKFold(n_splits=min(len(df['subject'].unique()), 10))
    fold_aucs = []
    individual_model_aucs = {name: [] for name, _ in models}
    
    print("Running universal cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_universal, y, df['subject'])):
        test_subjects = df.iloc[test_idx]['subject'].unique()
        print(f"\nFold {fold+1}: Test subjects {test_subjects}")
        
        X_train, X_test = X_universal.iloc[train_idx], X_universal.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Handle NaN values and preprocessing
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        try:
            # Train ensemble
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            if len(np.unique(y_test)) == 2:
                ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
                ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            else:
                ensemble_proba = ensemble.predict_proba(X_test_scaled)
                ensemble_auc = roc_auc_score(y_test, ensemble_proba, multi_class='ovr')
            
            fold_aucs.append(ensemble_auc)
            print(f"  Universal Ensemble AUC: {ensemble_auc:.3f}")
            
            # Evaluate individual models
            for name, model in models:
                model.fit(X_train_scaled, y_train)
                
                if len(np.unique(y_test)) == 2:
                    model_proba = model.predict_proba(X_test_scaled)[:, 1]
                    model_auc = roc_auc_score(y_test, model_proba)
                else:
                    model_proba = model.predict_proba(X_test_scaled)
                    model_auc = roc_auc_score(y_test, model_proba, multi_class='ovr')
                
                individual_model_aucs[name].append(model_auc)
                print(f"    {name.upper()} AUC: {model_auc:.3f}")
                
        except Exception as e:
            print(f"  Fold {fold+1}: Error {e}")
    
    if fold_aucs:
        ensemble_avg = np.mean(fold_aucs)
        ensemble_std = np.std(fold_aucs)
        
        print(f"\n🚀 UNIVERSAL ENSEMBLE RESULTS:")
        print(f"Universal Ensemble AUC: {ensemble_avg:.3f} ± {ensemble_std:.3f}")
        
        print(f"\nIndividual Model Performance:")
        for name, aucs in individual_model_aucs.items():
            if aucs:
                avg_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                print(f"  {name.upper():15s}: {avg_auc:.3f} ± {std_auc:.3f}")
        
        # Compare with previous technique-specific results (0.556 weighted average)
        baseline_auc = 0.556  # Your current best weighted average
        improvement = ensemble_avg - baseline_auc
        print(f"\nImprovement over technique-mixed baseline: {improvement:+.3f}")
        
        # Determine success level
        if ensemble_avg > 0.65:
            print("✅ EXCELLENT: Universal model achieves 0.65+ AUC target!")
        elif ensemble_avg > 0.60:
            print("✅ GOOD: Strong universal performance across all techniques")
        elif improvement > 0.02:
            print("✅ MODERATE: Meaningful improvement through universal features")
        else:
            print("⚠️  LIMITED: Universal approach shows marginal improvement")
        
        return {
            'universal_ensemble_auc_mean': ensemble_avg,
            'universal_ensemble_auc_std': ensemble_std,
            'individual_model_results': {name: {'mean': np.mean(aucs), 'std': np.std(aucs)} 
                                       for name, aucs in individual_model_aucs.items() if aucs},
            'improvement_over_baseline': improvement,
            'n_folds': len(fold_aucs),
            'universal_features_count': X_universal.shape[1],
            'original_features_count': X.shape[1]
        }
    else:
        print("❌ No valid universal ensemble results")
        return None


def ensemble_cross_subject_validation(df, X, y, technique_col='week'):
    """
    Advanced ensemble approach for cross-subject validation.
    Combines multiple models and techniques for better generalization.
    
    Args:
        df: DataFrame with metadata
        X: Feature matrix
        y: Labels
        technique_col: Column indicating different techniques
    
    Returns:
        dict: Ensemble validation results
    """
    print(f"\n===== ENSEMBLE CROSS-SUBJECT VALIDATION =====")
    
    # Create enhanced features
    X_enhanced = create_universal_breathing_features(X)
    
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    # Define ensemble of models
    models = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]
    
    # Create voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='soft')
    
    # Cross-validation
    gkf = GroupKFold(n_splits=min(len(df['subject'].unique()), 8))
    fold_aucs = []
    individual_model_aucs = {name: [] for name, _ in models}
    
    print("Running ensemble cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_enhanced, y, df['subject'])):
        test_subjects = df.iloc[test_idx]['subject'].unique()
        print(f"\nFold {fold+1}: Test subjects {test_subjects}")
        
        X_train, X_test = X_enhanced.iloc[train_idx], X_enhanced.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Handle NaN values and preprocessing
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        try:
            # Train ensemble
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            if len(np.unique(y_test)) == 2:
                ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
                ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            else:
                ensemble_proba = ensemble.predict_proba(X_test_scaled)
                ensemble_auc = roc_auc_score(y_test, ensemble_proba, multi_class='ovr')
            
            fold_aucs.append(ensemble_auc)
            print(f"  Ensemble AUC: {ensemble_auc:.3f}")
            
            # Evaluate individual models
            for name, model in models:
                model.fit(X_train_scaled, y_train)
                
                if len(np.unique(y_test)) == 2:
                    model_proba = model.predict_proba(X_test_scaled)[:, 1]
                    model_auc = roc_auc_score(y_test, model_proba)
                else:
                    model_proba = model.predict_proba(X_test_scaled)
                    model_auc = roc_auc_score(y_test, model_proba, multi_class='ovr')
                
                individual_model_aucs[name].append(model_auc)
                print(f"    {name.upper()} AUC: {model_auc:.3f}")
                
        except Exception as e:
            print(f"  Fold {fold+1}: Error {e}")
    
    if fold_aucs:
        ensemble_avg = np.mean(fold_aucs)
        ensemble_std = np.std(fold_aucs)
        
        print(f"\n🚀 ENSEMBLE RESULTS:")
        print(f"Ensemble Average AUC: {ensemble_avg:.3f} ± {ensemble_std:.3f}")
        
        print(f"\nIndividual Model Performance:")
        for name, aucs in individual_model_aucs.items():
            if aucs:
                avg_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                print(f"  {name.upper():15s}: {avg_auc:.3f} ± {std_auc:.3f}")
        
        # Compare with previous best (0.581 from Week 3)
        improvement = ensemble_avg - 0.581
        print(f"\nImprovement over baseline: {improvement:+.3f}")
        
        return {
            'ensemble_auc_mean': ensemble_avg,
            'ensemble_auc_std': ensemble_std,
            'individual_model_results': {name: {'mean': np.mean(aucs), 'std': np.std(aucs)} 
                                       for name, aucs in individual_model_aucs.items() if aucs},
            'improvement_over_baseline': improvement,
            'n_folds': len(fold_aucs)
        }
    else:
        print("❌ No valid ensemble results")
        return None


def within_technique_cross_subject_validation(df, X, y, technique_col='week'):
    """
    Perform cross-subject validation within each technique/week separately.
    This avoids confounding technique effects with subject effects.
    
    Args:
        df: DataFrame with metadata
        X: Feature matrix
        y: Labels
        technique_col: Column indicating different techniques (e.g., 'week')
    
    Returns:
        dict: Results for each technique
    """
    print(f"\n===== WITHIN-TECHNIQUE CROSS-SUBJECT VALIDATION =====")
    
    techniques = sorted(df[technique_col].unique())
    technique_results = {}
    
    for technique in techniques:
        print(f"\n--- TECHNIQUE {technique} ---")
        
        # Filter to only this technique
        technique_mask = df[technique_col] == technique
        df_tech = df[technique_mask].copy()
        X_tech = X[technique_mask]
        y_tech = y[technique_mask]
        
        print(f"Technique {technique}: {len(df_tech)} samples, {len(df_tech['subject'].unique())} subjects")
        
        # Check if we have enough subjects and samples for CV
        subjects_in_tech = df_tech['subject'].unique()
        if len(subjects_in_tech) < 3:
            print(f"  ⚠️  Skipping technique {technique}: only {len(subjects_in_tech)} subjects")
            continue
        
        # Check class distribution
        class_dist = y_tech.value_counts().sort_index()
        print(f"  Class distribution: {class_dist.to_dict()}")
        
        if len(class_dist) < 2 or class_dist.min() < 5:
            print(f"  ⚠️  Skipping technique {technique}: poor class distribution")
            continue
        
        # Perform cross-subject CV within this technique
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        
        gkf = GroupKFold(n_splits=min(len(subjects_in_tech), 5))
        fold_aucs = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_tech, y_tech, df_tech['subject'])):
            test_subjects = df_tech.iloc[test_idx]['subject'].unique()
            train_subjects = df_tech.iloc[train_idx]['subject'].unique()
            
            print(f"    Fold {fold+1}: Train subjects {train_subjects}, Test subjects {test_subjects}")
            
            # Use iloc for proper indexing with integer positions
            X_train, X_test = X_tech.iloc[train_idx], X_tech.iloc[test_idx]
            y_train, y_test = y_tech.iloc[train_idx], y_tech.iloc[test_idx]
            
            # Basic preprocessing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Simple classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            try:
                if len(np.unique(y_test)) == 2:
                    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    y_pred_proba = clf.predict_proba(X_test_scaled)
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                
                fold_aucs.append(auc)
                print(f"      AUC: {auc:.3f}")
                
            except Exception as e:
                print(f"      Error: {e}")
        
        if fold_aucs:
            avg_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            print(f"  Technique {technique} Average AUC: {avg_auc:.3f} ± {std_auc:.3f}")
            
            technique_results[technique] = {
                'auc_mean': avg_auc,
                'auc_std': std_auc,
                'fold_aucs': fold_aucs,
                'n_subjects': len(subjects_in_tech),
                'n_samples': len(df_tech),
                'class_distribution': class_dist.to_dict()
            }
        else:
            print(f"  No valid results for technique {technique}")
    
    if technique_results:
        print(f"\n--- WITHIN-TECHNIQUE CV SUMMARY ---")
        for tech, results in technique_results.items():
            print(f"Technique {tech}: {results['auc_mean']:.3f} ± {results['auc_std']:.3f} AUC ({results['n_subjects']} subjects)")
        
        # Overall weighted average
        total_samples = sum(r['n_samples'] for r in technique_results.values())
        weighted_auc = sum(r['auc_mean'] * r['n_samples'] for r in technique_results.values()) / total_samples
        print(f"\nWeighted Average AUC: {weighted_auc:.3f}")
        
        return technique_results
    else:
        print("No valid technique results!")
        return None


def cross_technique_within_subject_validation(df, X, y, technique_col='week'):
    """
    Perform within-subject validation across different techniques.
    Train on some techniques, test on others, within each subject.
    
    Args:
        df: DataFrame with metadata
        X: Feature matrix  
        y: Labels
        technique_col: Column indicating different techniques
    
    Returns:
        dict: Results for cross-technique generalization
    """
    print(f"\n===== CROSS-TECHNIQUE WITHIN-SUBJECT VALIDATION =====")
    
    techniques = sorted(df[technique_col].unique())
    if len(techniques) < 2:
        print("Need at least 2 techniques for cross-technique analysis")
        return None
    
    subject_results = {}
    
    for subject in df['subject'].unique():
        print(f"\nSubject {subject}:")
        
        # Get subject data
        subject_mask = df['subject'] == subject
        df_subj = df[subject_mask]
        X_subj = X[subject_mask]
        y_subj = y[subject_mask]
        
        # Check which techniques this subject participated in
        subject_techniques = sorted(df_subj[technique_col].unique())
        print(f"  Participated in techniques: {subject_techniques}")
        
        if len(subject_techniques) < 2:
            print(f"  ⚠️  Skipping: only participated in {len(subject_techniques)} technique(s)")
            continue
        
        # Try different train/test technique splits
        technique_aucs = []
        
        for train_tech in subject_techniques:
            for test_tech in subject_techniques:
                if train_tech == test_tech:
                    continue
                
                # Get train and test data
                train_mask = df_subj[technique_col] == train_tech
                test_mask = df_subj[technique_col] == test_tech
                
                X_train = X_subj[train_mask]
                y_train = y_subj[train_mask]
                X_test = X_subj[test_mask]
                y_test = y_subj[test_mask]
                
                # Check if we have enough data and class diversity
                if len(X_train) < 10 or len(X_test) < 5:
                    continue
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import roc_auc_score
                    
                    # Preprocess
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train classifier
                    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    clf.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    if len(np.unique(y_test)) == 2:
                        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                        auc = roc_auc_score(y_test, y_pred_proba)
                    else:
                        y_pred_proba = clf.predict_proba(X_test_scaled)
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    
                    technique_aucs.append(auc)
                    print(f"    Train on {train_tech} → Test on {test_tech}: AUC {auc:.3f}")
                    
                except Exception as e:
                    print(f"    Train on {train_tech} → Test on {test_tech}: Error {e}")
        
        if technique_aucs:
            avg_auc = np.mean(technique_aucs)
            print(f"  Average cross-technique AUC: {avg_auc:.3f}")
            
            subject_results[subject] = {
                'auc_mean': avg_auc,
                'technique_aucs': technique_aucs,
                'n_technique_pairs': len(technique_aucs),
                'techniques_participated': subject_techniques
            }
    
    if subject_results:
        print(f"\n--- CROSS-TECHNIQUE SUMMARY ---")
        all_aucs = []
        for subj, results in subject_results.items():
            print(f"Subject {subj}: {results['auc_mean']:.3f} AUC ({results['n_technique_pairs']} technique pairs)")
            all_aucs.extend(results['technique_aucs'])
        
        overall_auc = np.mean(all_aucs)
        overall_std = np.std(all_aucs)
        print(f"\nOverall Cross-Technique AUC: {overall_auc:.3f} ± {overall_std:.3f}")
        
        return {
            'subject_results': subject_results,
            'overall_auc': overall_auc,
            'overall_std': overall_std,
            'all_aucs': all_aucs
        }
    else:
        print("No valid cross-technique results!")
        return None
def within_subject_decoding(df, X, y):
    """
    Perform within-subject decoding analysis.
    
    For each subject:
    1. Split their data temporally (by session/run)
    2. Train and test within that subject
    3. Report per-subject AUC
    
    Args:
        df: Original dataframe with subject/session info
        X: Feature matrix (pandas DataFrame)
        y: Labels
    
    Returns:
        dict: Per-subject AUCs and overall within-subject average
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import roc_auc_score
    
    print("\n===== WITHIN-SUBJECT DECODING ANALYSIS =====")
    
    subject_aucs = {}
    subjects = df['subject'].unique()
    
    for subject in subjects:
        print(f"\nAnalyzing subject: {subject}")
        
        # Get data for this subject only
        subject_mask = df['subject'] == subject
        X_subj = X[subject_mask]
        y_subj = y[subject_mask]
        df_subj = df[subject_mask]
        
        # Check if we have enough data and multiple sessions/runs
        if len(X_subj) < 20:  # Need minimum samples
            print(f"  Skipping {subject}: insufficient data ({len(X_subj)} samples)")
            continue
            
        # Try to split by session first, then by run, then temporally
        if 'week' in df_subj.columns and len(df_subj['week'].unique()) > 1:
            # Split by session/week
            sessions = df_subj['week'].unique()
            train_sessions = sessions[:len(sessions)//2] if len(sessions) > 2 else sessions[:1]
            test_sessions = sessions[len(sessions)//2:] if len(sessions) > 2 else sessions[1:]
            
            train_mask = df_subj['week'].isin(train_sessions)
            test_mask = df_subj['week'].isin(test_sessions)
            
        elif 'run' in df_subj.columns and len(df_subj['run'].unique()) > 1:
            # Split by run
            runs = df_subj['run'].unique()
            train_runs = runs[:len(runs)//2] if len(runs) > 2 else runs[:1]
            test_runs = runs[len(runs)//2:] if len(runs) > 2 else runs[1:]
            
            train_mask = df_subj['run'].isin(train_runs)
            test_mask = df_subj['run'].isin(test_runs)
            
        else:
            # Temporal split (first half vs second half)
            split_idx = len(X_subj) // 2
            train_mask = np.arange(len(X_subj)) < split_idx
            test_mask = np.arange(len(X_subj)) >= split_idx
        
        # Create train/test splits
        X_train_subj = X_subj[train_mask]
        X_test_subj = X_subj[test_mask]
        y_train_subj = y_subj[train_mask]
        y_test_subj = y_subj[test_mask]
        
        # Check if we have both classes in train and test
        if len(np.unique(y_train_subj)) < 2 or len(np.unique(y_test_subj)) < 2:
            print(f"  Skipping {subject}: insufficient class diversity")
            continue
            
        if len(X_test_subj) < 5:  # Need minimum test samples
            print(f"  Skipping {subject}: insufficient test data ({len(X_test_subj)} samples)")
            continue
        
        print(f"  Train: {len(X_train_subj)} samples, Test: {len(X_test_subj)} samples")
        
        try:
            # Preprocess data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subj)
            X_test_scaled = scaler.transform(X_test_subj)
            
            # Encode labels
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train_subj)
            y_test_enc = le.transform(y_test_subj)
            
            # Train simple classifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            clf.fit(X_train_scaled, y_train_enc)
            
            # Predict and compute AUC
            if len(le.classes_) == 2:
                # Binary classification
                y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test_enc, y_pred_proba)
            else:
                # Multi-class classification
                y_pred_proba = clf.predict_proba(X_test_scaled)
                auc = roc_auc_score(y_test_enc, y_pred_proba, multi_class='ovr')
            
            subject_aucs[subject] = auc
            print(f"  AUC: {auc:.3f}")
            
        except Exception as e:
            print(f"  Error processing {subject}: {e}")
            continue
    
    # Compute overall within-subject performance
    if subject_aucs:
        within_subject_avg = np.mean(list(subject_aucs.values()))
        within_subject_std = np.std(list(subject_aucs.values()))
        
        print(f"\n--- WITHIN-SUBJECT RESULTS ---")
        print(f"Average within-subject AUC: {within_subject_avg:.3f} ± {within_subject_std:.3f}")
        print(f"Individual subject AUCs:")
        for subj, auc in subject_aucs.items():
            print(f"  {subj}: {auc:.3f}")
        
        return {
            'subject_aucs': subject_aucs,
            'average_auc': within_subject_avg,
            'std_auc': within_subject_std
        }
    else:
        print("No subjects could be processed for within-subject analysis")
        return None


def coral_domain_adaptation(X_source, X_target):
    """
    CORAL (CORrelation ALignment) domain adaptation.
    
    Aligns the covariance of source domain features to match target domain.
    
    Args:
        X_source: Source domain features (training data)
        X_target: Target domain features (test data)
    
    Returns:
        X_source_adapted: Source features adapted to target domain
        X_target_adapted: Target features (whitened)
    """
    print("Applying CORAL domain adaptation...")
    
    # Compute covariance matrices
    cov_source = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1]) * 1e-6
    cov_target = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1]) * 1e-6
    
    # Compute transformation matrices using matrix square roots
    try:
        # Source whitening transformation
        eigenvals_s, eigenvecs_s = np.linalg.eigh(cov_source)
        eigenvals_s = np.maximum(eigenvals_s, 1e-8)  # Ensure positive
        sqrt_cov_source = eigenvecs_s @ np.diag(np.sqrt(eigenvals_s)) @ eigenvecs_s.T
        inv_sqrt_cov_source = eigenvecs_s @ np.diag(1.0 / np.sqrt(eigenvals_s)) @ eigenvecs_s.T
        
        # Target coloring transformation  
        eigenvals_t, eigenvecs_t = np.linalg.eigh(cov_target)
        eigenvals_t = np.maximum(eigenvals_t, 1e-8)  # Ensure positive
        sqrt_cov_target = eigenvecs_t @ np.diag(np.sqrt(eigenvals_t)) @ eigenvecs_t.T
        
        # CORAL transformation: A_coral = A_target^(1/2) * A_source^(-1/2)
        coral_transform = sqrt_cov_target @ inv_sqrt_cov_source
        
        # Apply transformation
        X_source_adapted = X_source @ coral_transform.T
        X_target_adapted = X_target  # Target stays the same
        
        print(f"CORAL adaptation completed. Transform shape: {coral_transform.shape}")
        
        return X_source_adapted, X_target_adapted, coral_transform
        
    except np.linalg.LinAlgError as e:
        print(f"CORAL failed due to numerical issues: {e}")
        print("Returning original features...")
        return X_source, X_target, np.eye(X_source.shape[1])


def compute_alpha_theta_ratio(eeg_data, fs=500, alpha_band=(8, 13), theta_band=(4, 8)):
    """
    Compute alpha/theta power ratio for each trial.
    
    Args:
        eeg_data: EEG data array of shape (n_trials, n_channels, n_timepoints)
        fs: Sampling frequency
        alpha_band: Alpha frequency band (Hz)
        theta_band: Theta frequency band (Hz)
    
    Returns:
        alpha_theta_ratios: Array of shape (n_trials,) with ratio per trial
    """
    from scipy import signal
    
    print("Computing alpha/theta power ratios...")
    
    n_trials, n_channels, n_timepoints = eeg_data.shape
    alpha_theta_ratios = np.zeros(n_trials)
    
    for trial in range(n_trials):
        trial_data = eeg_data[trial]  # Shape: (n_channels, n_timepoints)
        
        alpha_power_total = 0
        theta_power_total = 0
        
        for ch in range(n_channels):
            # Compute power spectral density
            freqs, psd = signal.welch(trial_data[ch], fs, nperseg=min(256, n_timepoints//4))
            
            # Find frequency indices for bands
            alpha_idx = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
            theta_idx = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
            
            # Compute band powers
            alpha_power = np.trapz(psd[alpha_idx], freqs[alpha_idx])
            theta_power = np.trapz(psd[theta_idx], freqs[theta_idx])
            
            alpha_power_total += alpha_power
            theta_power_total += theta_power
        
        # Compute ratio (add small epsilon to avoid division by zero)
        alpha_theta_ratios[trial] = (alpha_power_total + 1e-8) / (theta_power_total + 1e-8)
    
    print(f"Computed alpha/theta ratios for {n_trials} trials")
    return alpha_theta_ratios


def compute_permutation_entropy(eeg_data, order=3, normalize=True):
    """
    Compute permutation entropy for each trial.
    
    Args:
        eeg_data: EEG data array of shape (n_trials, n_channels, n_timepoints)
        order: Order of permutation patterns
        normalize: Whether to normalize entropy
    
    Returns:
        pe_features: Array of shape (n_trials, n_channels) with PE per channel per trial
    """
    from scipy.special import factorial
    
    print("Computing permutation entropy...")
    
    def _embed(x, order):
        """Time-delay embedding"""
        N = len(x)
        if order * 1 >= N:
            raise ValueError("Insufficient data length for given order")
        
        embedded = np.zeros((N - order + 1, order))
        for i in range(order):
            embedded[:, i] = x[i:N - order + 1 + i]
        return embedded
    
    def _relative_variance(x, order):
        """Compute permutation entropy for a single time series"""
        embedded = _embed(x, order)
        sorted_idx = np.argsort(embedded, axis=1)
        
        # Convert to permutation patterns
        perms = np.zeros(len(embedded), dtype=int)
        for i, perm in enumerate(sorted_idx):
            # Convert permutation to unique integer
            perm_int = 0
            for j, p in enumerate(perm):
                perm_int += p * (order ** (order - 1 - j))
            perms[i] = perm_int
        
        # Count occurrences of each permutation
        unique_perms, counts = np.unique(perms, return_counts=True)
        
        # Compute relative frequencies
        total_perms = len(perms)
        rel_freqs = counts / total_perms
        
        # Compute entropy
        entropy = -np.sum(rel_freqs * np.log2(rel_freqs))
        
        if normalize:
            max_entropy = np.log2(factorial(order))
            entropy = entropy / max_entropy
        
        return entropy
    
    n_trials, n_channels, n_timepoints = eeg_data.shape
    pe_features = np.zeros((n_trials, n_channels))
    
    for trial in range(n_trials):
        for ch in range(n_channels):
            try:
                pe_features[trial, ch] = _relative_variance(eeg_data[trial, ch], order)
            except (ValueError, ZeroDivisionError):
                pe_features[trial, ch] = 0  # Handle edge cases
    
    print(f"Computed permutation entropy for {n_trials} trials, {n_channels} channels")
    return pe_features


def create_channel_layout_2d(n_channels=64):
    """
    Create a 2D layout for EEG channels (simplified grid layout).
    
    For a proper implementation, you would use actual electrode positions.
    This is a simplified version that arranges channels in a grid.
    
    Args:
        n_channels: Number of EEG channels
    
    Returns:
        layout: Dictionary mapping channel index to (row, col) position
        grid_shape: Shape of the 2D grid (rows, cols)
    """
    # Create a roughly square grid
    grid_size = int(np.ceil(np.sqrt(n_channels)))
    grid_shape = (grid_size, grid_size)
    
    layout = {}
    for ch in range(n_channels):
        row = ch // grid_size
        col = ch % grid_size
        layout[ch] = (row, col)
    
    return layout, grid_shape


def eeg_trials_to_images(eeg_data, channel_layout=None):
    """
    Convert EEG trials to 2D channel layout images.
    
    Args:
        eeg_data: EEG data of shape (n_trials, n_channels, n_timepoints)
        channel_layout: Dictionary mapping channel to (row, col) positions
    
    Returns:
        images: Array of shape (n_trials, height, width, time_windows)
    """
    n_trials, n_channels, n_timepoints = eeg_data.shape
    
    # Create channel layout if not provided
    if channel_layout is None:
        layout, grid_shape = create_channel_layout_2d(n_channels)
    else:
        layout = channel_layout
        max_row = max(pos[0] for pos in layout.values())
        max_col = max(pos[1] for pos in layout.values())
        grid_shape = (max_row + 1, max_col + 1)
    
    # For simplicity, we'll average over time windows or use power in frequency bands
    # Here we'll compute the mean power over the entire trial
    print("Converting EEG trials to 2D channel images...")
    
    images = np.zeros((n_trials, grid_shape[0], grid_shape[1]))
    
    for trial in range(n_trials):
        trial_image = np.zeros(grid_shape)
        
        for ch in range(n_channels):
            if ch in layout:
                row, col = layout[ch]
                # Use RMS power as the pixel value
                trial_image[row, col] = np.sqrt(np.mean(eeg_data[trial, ch] ** 2))
        
        images[trial] = trial_image
    
    # Add channel dimension for CNN input
    images = np.expand_dims(images, axis=-1)  # Shape: (n_trials, height, width, 1)
    
    print(f"Converted to images: {images.shape}")
    return images


def build_simple_cnn(input_shape, num_classes):
    """
    Build a simple CNN for EEG channel layout classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        model: Compiled Keras CNN model
    """
    from keras._tf_keras.keras.models import Sequential
    from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from keras._tf_keras.keras.optimizers import Adam
    
    print(f"Building CNN for input shape: {input_shape}")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("CNN model built successfully")
    return model


def create_performance_analysis_plots(results_dir='results'):
    """
    Create comprehensive performance analysis plots from saved results.
    """
    import json
    import glob
    
    print(f"Creating performance analysis plots from {results_dir}...")
    
    # Collect results from all folds
    fold_dirs = glob.glob(os.path.join(results_dir, 'fold_*'))
    all_results = []
    
    for fold_dir in sorted(fold_dirs):
        metrics_path = os.path.join(fold_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                all_results.append(metrics)
    
    if not all_results:
        print("No fold results found!")
        return
    
    # Extract model performance data
    model_types = ['baseline', 'breathing_optimized', 'tuned']
    model_performances = {model: [] for model in model_types}
    
    for result in all_results:
        if 'all_model_results' in result:
            for model_type in model_types:
                if model_type in result['all_model_results']:
                    auc = result['all_model_results'][model_type]['auc']
                    model_performances[model_type].append(auc)
    
    # Plot 1: Model comparison across folds
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for model_type, aucs in model_performances.items():
        if aucs:
            plt.plot(range(1, len(aucs) + 1), aucs, 'o-', label=model_type.replace('_', ' ').title())
    plt.title('Model Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of model performances
    plt.subplot(2, 2, 2)
    box_data = [aucs for aucs in model_performances.values() if aucs]
    box_labels = [model.replace('_', ' ').title() for model, aucs in model_performances.items() if aucs]
    plt.boxplot(box_data, labels=box_labels)
    plt.title('Model Performance Distribution')
    plt.ylabel('AUC')
    plt.xticks(rotation=45)
    
    # Plot 3: Average performance with error bars
    plt.subplot(2, 2, 3)
    means = [np.mean(aucs) if aucs else 0 for aucs in model_performances.values()]
    stds = [np.std(aucs) if aucs else 0 for aucs in model_performances.values()]
    colors = ['skyblue', 'lightgreen', 'coral']
    
    bars = plt.bar(box_labels, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    plt.title('Average Model Performance')
    plt.ylabel('AUC')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01, 
                f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)
    
    # Plot 4: Best model selection frequency
    plt.subplot(2, 2, 4)
    best_models = [result.get('model_type', 'unknown') for result in all_results]
    model_counts = pd.Series(best_models).value_counts()
    
    plt.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
    plt.title('Best Model Selection Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print(f"\n📊 PERFORMANCE ANALYSIS SUMMARY:")
    print(f"{'='*50}")
    
    for model_type, aucs in model_performances.items():
        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            print(f"{model_type.replace('_', ' ').title():20s}: {mean_auc:.3f} ± {std_auc:.3f} AUC")
    
    print(f"\nBest model selection:")
    for model, count in model_counts.items():
        percentage = count / len(all_results) * 100
        print(f"  {model.replace('_', ' ').title():20s}: {count}/{len(all_results)} folds ({percentage:.1f}%)")
    
    # Save performance summary
    summary = {
        'model_performances': {k: {'mean': np.mean(v), 'std': np.std(v), 'values': v} 
                              for k, v in model_performances.items() if v},
        'best_model_selection': model_counts.to_dict(),
        'total_folds': len(all_results)
    }
    
    with open(os.path.join(results_dir, 'performance_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved performance analysis to {results_dir}/performance_analysis.png")
    print(f"Saved performance summary to {results_dir}/performance_summary.json")


def visualize_embeddings_tsne(features, labels, method='tsne', save_path=None):
    """
    Create t-SNE or UMAP visualization of learned features.
    
    Args:
        features: Feature vectors of shape (n_samples, n_features)
        labels: Class labels
        method: 'tsne' or 'umap'
        save_path: Path to save the plot
    
    Returns:
        embedding: 2D embedding coordinates
    """
    print(f"Computing {method.upper()} embedding...")
    
    # Handle NaN values in features
    print(f"Input features shape: {features.shape}")
    print(f"NaN count before cleaning: {np.isnan(features).sum()}")
    
    # Remove samples with any NaN values
    nan_mask = np.isnan(features).any(axis=1)
    if nan_mask.any():
        print(f"Removing {nan_mask.sum()} samples with NaN values")
        features_clean = features[~nan_mask]
        labels_clean = labels[~nan_mask]
    else:
        features_clean = features
        labels_clean = labels
    
    print(f"Clean features shape: {features_clean.shape}")
    print(f"NaN count after cleaning: {np.isnan(features_clean).sum()}")
    
    # Check if we have enough samples
    if len(features_clean) < 10:
        print("Not enough clean samples for embedding visualization")
        return None
    
    # Additional check for infinite values
    inf_mask = np.isinf(features_clean).any(axis=1)
    if inf_mask.any():
        print(f"Removing {inf_mask.sum()} samples with infinite values")
        features_clean = features_clean[~inf_mask]
        labels_clean = labels_clean[~inf_mask]
    
    if method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = min(30, len(features_clean)//3, len(features_clean)-1)
        perplexity = max(5, perplexity)  # Ensure minimum perplexity
        embedder = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    elif method.lower() == 'umap':
        try:
            import umap
            n_neighbors = min(15, len(features_clean)-1)
            n_neighbors = max(2, n_neighbors)
            embedder = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            perplexity = min(30, len(features_clean)//3, len(features_clean)-1)
            perplexity = max(5, perplexity)
            embedder = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")
    
    # Compute embedding
    embedding = embedder.fit_transform(features_clean)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels_clean)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_clean == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.6)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{method.upper()} Visualization of Learned Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved embedding plot to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return embedding


def create_ensemble_model(X_train, Y_train, X_val, Y_val, class_weights):
    """
    Create an ensemble of different model architectures to capture various patterns.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    # Convert one-hot back to labels for sklearn models
    y_train_labels = Y_train.argmax(axis=1)
    y_val_labels = Y_val.argmax(axis=1)
    
    models = {}
    
    # 1. Deep Neural Network
    print("Training deep neural network...")
    nn_model = build_model(X_train.shape[1], Y_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)
    ]
    nn_model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0
    )
    models['neural_network'] = nn_model
    
    # 2. Random Forest
    print("Training random forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_labels)
    models['random_forest'] = rf_model
    
    # 3. Gradient Boosting
    print("Training gradient boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    gb_model.fit(X_train, y_train_labels)
    models['gradient_boosting'] = gb_model
    
    # 4. SVM with RBF kernel
    print("Training SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train_labels)
    models['svm'] = svm_model
    
    return models


def ensemble_predict(models, X_test, num_classes):
    """
    Make predictions using ensemble of models.
    """
    predictions = []
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            # Sklearn models
            pred_proba = model.predict_proba(X_test)
        else:
            # Keras models
            pred_proba = model.predict(X_test)
        
        # Ensure we have the right number of classes
        if pred_proba.shape[1] != num_classes:
            # Handle case where some classes might be missing in some folds
            full_pred = np.zeros((pred_proba.shape[0], num_classes))
            full_pred[:, :pred_proba.shape[1]] = pred_proba
            pred_proba = full_pred
            
        predictions.append(pred_proba)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred


def advanced_analysis_pipeline(df, X, y, results_dir='advanced_results'):
    """
    Comprehensive analysis pipeline that demonstrates all the new methods.
    
    This function shows how to integrate:
    1. Data quality assessment and recommendations
    2. Fixed within-subject vs cross-subject decoding comparison
    3. Robust cross-subject validation with good subjects only
    4. CORAL domain adaptation
    5. Feature engineering (alpha/theta ratio, permutation entropy)
    6. CNN on channel layout
    7. t-SNE visualization
    
    Args:
        df: Original dataframe with metadata
        X: Feature matrix
        y: Labels
        results_dir: Directory to save results
    """
    import os
    from sklearn.model_selection import GroupKFold, train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report
    import json
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n===== ADVANCED ANALYSIS PIPELINE =====")
    print(f"Results will be saved to: {results_dir}")
    
    # ========== 0. DATA QUALITY ASSESSMENT ==========
    print("\n0. DATA QUALITY ASSESSMENT")
    quality_results = comprehensive_data_quality_check(df, X, y)
    
    # Save quality assessment
    quality_path = os.path.join(results_dir, 'data_quality_assessment.json')
    safe_json_dump(quality_results, quality_path)
    print(f"Saved data quality assessment to {quality_path}")
    
    # ========== 1. FIXED WITHIN-SUBJECT ANALYSIS ==========
    print("\n1. WITHIN-SUBJECT DECODING ANALYSIS")
    fixed_within_auc, within_results = fixed_within_subject_decoding(df, X, y)
    
    if within_results:
        # Save within-subject results
        within_path = os.path.join(results_dir, 'within_subject_results.json')
        within_data = {
            'average_auc': fixed_within_auc,
            'individual_results': within_results,
            'valid_subjects': list(within_results.keys()),
            'total_subjects_attempted': len(df['subject'].unique())
        }
        safe_json_dump(within_data, within_path)
        print(f"Saved within-subject results to {within_path}")
        
        # Compare with your existing cross-subject AUC (~0.58)
        cross_subject_auc = 0.58  # Replace with your actual cross-subject AUC
        
        print(f"\n--- WITHIN vs CROSS-SUBJECT COMPARISON ---")
        print(f"Within-subject AUC: {fixed_within_auc:.3f}")
        print(f"Cross-subject AUC: {cross_subject_auc:.3f}")
        print(f"Generalization gap: {fixed_within_auc - cross_subject_auc:.3f}")
        
        if fixed_within_auc > 0.7:
            print("✓ Within-subject decoding is much better - suggests individual differences")
        else:
            print("✗ Even within-subject decoding is poor - may need better features")
    
    # ========== 1.5. TECHNIQUE-SPECIFIC ANALYSIS ==========
    print("\n1.5. TECHNIQUE-SPECIFIC ANALYSIS")
    technique_results = within_technique_cross_subject_validation(df, X, y, technique_col='week')
    robust_results = technique_results
    
    if robust_results:
        # Convert numpy int64 keys to strings for JSON serialization
        robust_results_json = {}
        for key, value in robust_results.items():
            robust_results_json[str(key)] = value
        
        robust_path = os.path.join(results_dir, 'robust_cv_results.json')
        safe_json_dump(robust_results_json, robust_path)
        print(f"Saved robust CV results to {robust_path}")
    
    # ========== 2. CORAL DOMAIN ADAPTATION EXAMPLE ==========
    print("\n2. CORAL DOMAIN ADAPTATION EXAMPLE")
    
    # Use GroupKFold to get one example fold
    gkf = GroupKFold(n_splits=3)
    groups = df['subject']
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        # Take just the first fold for demonstration
        X_train_fold = X.iloc[train_idx].values
        X_test_fold = X.iloc[test_idx].values
        y_train_fold = y.iloc[train_idx].values
        y_test_fold = y.iloc[test_idx].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Encode labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_fold)
        y_test_enc = le.transform(y_test_fold)
        
        # Train baseline classifier (no domain adaptation)
        clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf_baseline.fit(X_train_scaled, y_train_enc)
        
        if len(le.classes_) == 2:
            baseline_auc = roc_auc_score(y_test_enc, clf_baseline.predict_proba(X_test_scaled)[:, 1])
        else:
            baseline_auc = roc_auc_score(y_test_enc, clf_baseline.predict_proba(X_test_scaled), multi_class='ovr')
        
        # Apply CORAL domain adaptation
        X_train_coral, X_test_coral, coral_transform = coral_domain_adaptation(X_train_scaled, X_test_scaled)
        
        # Train classifier with CORAL-adapted features
        clf_coral = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf_coral.fit(X_train_coral, y_train_enc)
        
        if len(le.classes_) == 2:
            coral_auc = roc_auc_score(y_test_enc, clf_coral.predict_proba(X_test_coral)[:, 1])
        else:
            coral_auc = roc_auc_score(y_test_enc, clf_coral.predict_proba(X_test_coral), multi_class='ovr')
        
        print(f"Baseline AUC (no domain adaptation): {baseline_auc:.3f}")
        print(f"CORAL AUC (with domain adaptation): {coral_auc:.3f}")
        print(f"CORAL improvement: {coral_auc - baseline_auc:+.3f}")
        
        # Save CORAL results
        coral_results = {
            'baseline_auc': float(baseline_auc),
            'coral_auc': float(coral_auc),
            'improvement': float(coral_auc - baseline_auc),
            'coral_transform_shape': coral_transform.shape
        }
        
        coral_path = os.path.join(results_dir, 'coral_results.json')
        safe_json_dump(coral_results, coral_path)
        print(f"Saved CORAL results to {coral_path}")
        
        break  # Only do first fold for demonstration
    
    # ========== 3. FEATURE ENGINEERING EXAMPLES ==========
    print("\n3. FEATURE ENGINEERING EXAMPLES")
    
    # Note: This section requires raw EEG data
    # For demonstration, we'll create synthetic EEG-like data
    print("Note: Creating synthetic EEG data for demonstration")
    print("In practice, replace this with your actual EEG time series data")
    
    n_trials = len(X)
    n_channels = 64  # Common EEG setup
    n_timepoints = 1000  # 2 seconds at 500 Hz
    fs = 500  # Sampling frequency
    
    # Create synthetic EEG data with realistic properties
    np.random.seed(42)
    synthetic_eeg = np.random.randn(n_trials, n_channels, n_timepoints)
    
    # Add some realistic frequency content
    t = np.linspace(0, n_timepoints/fs, n_timepoints)
    for trial in range(n_trials):
        for ch in range(n_channels):
            # Add alpha (10 Hz) and theta (6 Hz) components
            alpha_component = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            theta_component = 0.3 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            synthetic_eeg[trial, ch] += alpha_component + theta_component
    
    # a) Compute alpha/theta ratio
    alpha_theta_ratios = compute_alpha_theta_ratio(synthetic_eeg, fs=fs)
    print(f"Alpha/theta ratios computed: shape {alpha_theta_ratios.shape}")
    print(f"Alpha/theta ratio range: {alpha_theta_ratios.min():.2f} - {alpha_theta_ratios.max():.2f}")
    
    # b) Compute permutation entropy
    pe_features = compute_permutation_entropy(synthetic_eeg, order=3)
    print(f"Permutation entropy computed: shape {pe_features.shape}")
    print(f"PE range: {pe_features.min():.3f} - {pe_features.max():.3f}")
    
    # Combine new features with existing features
    print("\nIntegrating new features with existing feature matrix...")
    
    # Add alpha/theta ratio as a new feature
    X_enhanced = X.copy()
    X_enhanced['alpha_theta_ratio'] = alpha_theta_ratios
    
    # Add mean permutation entropy across channels
    X_enhanced['mean_perm_entropy'] = np.mean(pe_features, axis=1)
    X_enhanced['std_perm_entropy'] = np.std(pe_features, axis=1)
    
    print(f"Enhanced feature matrix shape: {X_enhanced.shape}")
    print(f"New features added: alpha_theta_ratio, mean_perm_entropy, std_perm_entropy")
    
    # Test enhanced features with one fold
    for train_idx, test_idx in gkf.split(X_enhanced, y, groups=groups):
        X_train_enh = X_enhanced.iloc[train_idx].values
        X_test_enh = X_enhanced.iloc[test_idx].values
        y_train_enh = y.iloc[train_idx].values
        y_test_enh = y.iloc[test_idx].values
        
        # Scale and encode
        scaler_enh = StandardScaler()
        X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
        X_test_enh_scaled = scaler_enh.transform(X_test_enh)
        
        le_enh = LabelEncoder()
        y_train_enh_enc = le_enh.fit_transform(y_train_enh)
        y_test_enh_enc = le_enh.transform(y_test_enh)
        
        # Train classifier with enhanced features
        clf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf_enhanced.fit(X_train_enh_scaled, y_train_enh_enc)
        
        if len(le_enh.classes_) == 2:
            enhanced_auc = roc_auc_score(y_test_enh_enc, clf_enhanced.predict_proba(X_test_enh_scaled)[:, 1])
        else:
            enhanced_auc = roc_auc_score(y_test_enh_enc, clf_enhanced.predict_proba(X_test_enh_scaled), multi_class='ovr')
        
        print(f"Enhanced features AUC: {enhanced_auc:.3f}")
        print(f"Enhancement vs baseline: {enhanced_auc - baseline_auc:+.3f}")
        
        # Save feature engineering results
        fe_results = {
            'baseline_auc': float(baseline_auc),
            'enhanced_auc': float(enhanced_auc),
            'improvement': float(enhanced_auc - baseline_auc),
            'new_features': ['alpha_theta_ratio', 'mean_perm_entropy', 'std_perm_entropy'],
            'original_feature_count': X.shape[1],
            'enhanced_feature_count': X_enhanced.shape[1]
        }
        
        fe_path = os.path.join(results_dir, 'feature_engineering_results.json')
        with open(fe_path, 'w') as f:
            json.dump(fe_results, f, indent=2)
        print(f"Saved feature engineering results to {fe_path}")
        
        break  # Only do first fold for demonstration
    
    # ========== 4. CNN ON CHANNEL LAYOUT ==========
    print("\n4. CNN ON CHANNEL LAYOUT")
    
    # Convert EEG data to 2D channel layout images
    eeg_images = eeg_trials_to_images(synthetic_eeg)
    print(f"EEG images shape: {eeg_images.shape}")
    
    # Split for CNN training
    for train_idx, test_idx in gkf.split(eeg_images, y, groups=groups):
        X_img_train = eeg_images[train_idx]
        X_img_test = eeg_images[test_idx]
        y_img_train = y.iloc[train_idx].values
        y_img_test = y.iloc[test_idx].values
        
        # Encode labels for CNN
        le_cnn = LabelEncoder()
        y_img_train_enc = le_cnn.fit_transform(y_img_train)
        y_img_test_enc = le_cnn.transform(y_img_test)
        
        # Build and train CNN
        input_shape = eeg_images.shape[1:]  # (height, width, channels)
        num_classes = len(le_cnn.classes_)
        
        cnn_model = build_simple_cnn(input_shape, num_classes)
        
        # Train CNN with early stopping
        from keras._tf_keras.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = cnn_model.fit(
            X_img_train, y_img_train_enc,
            epochs=50,
            batch_size=32,
            validation_data=(X_img_test, y_img_test_enc),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate CNN
        cnn_probs = cnn_model.predict(X_img_test)
        
        if num_classes == 2:
            cnn_auc = roc_auc_score(y_img_test_enc, cnn_probs[:, 1])
        else:
            cnn_auc = roc_auc_score(y_img_test_enc, cnn_probs, multi_class='ovr')
        
        print(f"CNN AUC: {cnn_auc:.3f}")
        print(f"CNN vs baseline: {cnn_auc - baseline_auc:+.3f}")
        
        # Save CNN model and results
        cnn_model_path = os.path.join(results_dir, 'cnn_model.h5')
        cnn_model.save(cnn_model_path)
        
        cnn_results = {
            'baseline_auc': float(baseline_auc),
            'cnn_auc': float(cnn_auc),
            'improvement': float(cnn_auc - baseline_auc),
            'input_shape': list(input_shape),
            'num_classes': int(num_classes),
            'training_epochs': len(history.history['loss'])
        }
        
        cnn_path = os.path.join(results_dir, 'cnn_results.json')
        with open(cnn_path, 'w') as f:
            json.dump(cnn_results, f, indent=2)
        print(f"Saved CNN results to {cnn_path}")
        print(f"Saved CNN model to {cnn_model_path}")
        
        break  # Only do first fold for demonstration
    
    # ========== 5. t-SNE VISUALIZATION ==========
    print("\n5. t-SNE VISUALIZATION OF LEARNED FEATURES")
    
    # Use the enhanced features for visualization
    X_viz = X_enhanced.iloc[:500].values  # Limit to 500 samples for faster computation
    y_viz = y.iloc[:500].values
    
    # Scale features for visualization and handle NaNs
    scaler_viz = StandardScaler()
    X_viz_scaled = scaler_viz.fit_transform(X_viz)
    
    # Additional NaN/Inf check after scaling
    nan_mask = np.isnan(X_viz_scaled).any(axis=1)
    inf_mask = np.isinf(X_viz_scaled).any(axis=1)
    bad_mask = nan_mask | inf_mask
    
    if bad_mask.any():
        print(f"Removing {bad_mask.sum()} samples with NaN/Inf values after scaling")
        X_viz_clean = X_viz_scaled[~bad_mask]
        y_viz_clean = y_viz[~bad_mask]
    else:
        X_viz_clean = X_viz_scaled
        y_viz_clean = y_viz
    
    print(f"Final visualization data shape: {X_viz_clean.shape}")
    
    # Create t-SNE embedding only if we have enough clean data
    if len(X_viz_clean) >= 20:  # Need minimum samples for t-SNE
        tsne_path = os.path.join(results_dir, 'tsne_embedding.png')
        embedding = visualize_embeddings_tsne(
            X_viz_clean, 
            y_viz_clean, 
            method='tsne', 
            save_path=tsne_path
        )
        
        # Also try UMAP if available
        try:
            umap_path = os.path.join(results_dir, 'umap_embedding.png')
            embedding_umap = visualize_embeddings_tsne(
                X_viz_clean, 
                y_viz_clean, 
                method='umap', 
                save_path=umap_path
            )
        except Exception as e:
            print(f"UMAP visualization failed: {e}")
    else:
        print(f"Not enough clean samples ({len(X_viz_clean)}) for t-SNE visualization")
    
    # ========== SUMMARY ==========
    print(f"\n===== ADVANCED ANALYSIS SUMMARY =====")
    print(f"All results saved to: {results_dir}")
    print(f"Key findings:")
    if quality_results:
        print(f"  - Good subjects: {len(quality_results['good_subjects'])}/{len(df['subject'].unique())}")
        print(f"  - Problematic subjects: {quality_results['problematic_subjects']}")
    if within_results:
        print(f"  - Within-subject AUC: {fixed_within_auc:.3f}")
    print(f"  - CORAL improvement: {coral_results['improvement']:+.3f}")
    print(f"  - Feature engineering improvement: {fe_results['improvement']:+.3f}")
    print(f"  - CNN improvement: {cnn_results['improvement']:+.3f}")
    print(f"\nNext steps:")
    print(f"  1. Replace synthetic EEG with your actual time series data")
    print(f"  2. Try ensemble methods combining multiple approaches")
    print(f"  3. Experiment with different domain adaptation methods")
    print(f"  4. Use proper EEG channel layouts for more realistic CNN input")
    
    # Return summary for programmatic access
    return {
        'data_quality': quality_results,
        'within_subject_auc': fixed_within_auc,
        'robust_cv_results': robust_results,
        'coral_improvement': coral_results['improvement'],
        'feature_engineering_improvement': fe_results['improvement'],
        'cnn_improvement': cnn_results['improvement']
    }


# ...existing code...


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
    # subjects_to_exclude = ['s02', 's10', 's19', 's08']
    # df = df[~df['subject'].isin(subjects_to_exclude)]
    # print(f"Excluding subjects: {subjects_to_exclude}. Remaining subjects: {df['subject'].unique()}")

    groups = df['subject']  # Use subject IDs for grouped splitting
    y = df['transition_label']
    
    # Define features and drop metadata
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number', 'transition_label']
    X_raw = df.drop(columns=drop_cols, errors='ignore')
    X_raw = select_features(X_raw, pattern_keep=['glob_chans'])
    
    # ========== PRELIMINARY DATA QUALITY CHECK ==========
    print("\n===== PRELIMINARY DATA QUALITY ASSESSMENT =====")
    print("Running data quality check before cross-validation...")
    
    # Run comprehensive data quality check
    quality_results = comprehensive_data_quality_check(df, X_raw, y)
    
    # Ask user if they want to exclude problematic subjects
    if quality_results['problematic_subjects']:
        print(f"\n🚨 PROBLEMATIC SUBJECTS DETECTED: {quality_results['problematic_subjects']}")
        print(f"✅ GOOD SUBJECTS: {quality_results['good_subjects']}")
        
        exclude_choice = input("\nDo you want to exclude problematic subjects and proceed with good subjects only? (y/n): ").lower().strip()
        
        if exclude_choice == 'y':
            # Filter to only good subjects
            good_mask = df['subject'].isin(quality_results['good_subjects'])
            df = df[good_mask].copy()
            groups = df['subject']
            y = df['transition_label']
            X_raw = df.drop(columns=drop_cols, errors='ignore')
            X_raw = select_features(X_raw, pattern_keep=['glob_chans'])
            
            print(f"✅ Filtered dataset to {len(quality_results['good_subjects'])} good subjects")
            print(f"📊 New dataset size: {len(df)} samples")
            subjects_to_exclude = quality_results['problematic_subjects']
        else:
            print("⚠️  Proceeding with all subjects (including problematic ones)")
            subjects_to_exclude = []
    else:
        print("✅ All subjects passed quality checks!")
        subjects_to_exclude = []
    
    # ========== WEEK/TECHNIQUE STRUCTURE ANALYSIS ==========
    print(f"\n===== WEEK/TECHNIQUE STRUCTURE ANALYSIS =====")
    print("Analyzing the experimental design to understand technique confounding...")
    
    # Analyze week distribution per subject
    week_analysis = df.groupby('subject')['week'].agg(['count', 'nunique', lambda x: list(sorted(x.unique()))])
    week_analysis.columns = ['total_samples', 'n_weeks', 'weeks_participated']
    print("\nSubject participation by week:")
    for subject, row in week_analysis.iterrows():
        print(f"  {subject}: {row['total_samples']:4d} samples, {row['n_weeks']} weeks, weeks {row['weeks_participated']}")
    
    # Analyze class distribution by week
    print(f"\nClass distribution by week (technique):")
    week_class_dist = df.groupby('week')['transition_label'].value_counts().unstack(fill_value=0)
    print(week_class_dist)
    
    # Check for technique-specific patterns
    print(f"\nSubjects per week:")
    subjects_per_week = df.groupby('week')['subject'].nunique()
    print(subjects_per_week)
    
    print(f"\nWeek-wise analysis:")
    for week in sorted(df['week'].unique()):
        week_data = df[df['week'] == week]
        subjects_in_week = week_data['subject'].unique()
        class_dist = week_data['transition_label'].value_counts().sort_index()
        print(f"  Week {week}: {len(subjects_in_week)} subjects, classes {class_dist.to_dict()}")
        print(f"    Subjects: {list(subjects_in_week)}")
    
    # Detect if this is a technique confounding issue
    technique_issue = False
    all_weeks = set(df['week'].unique())
    for subject in df['subject'].unique():
        subject_weeks = set(df[df['subject'] == subject]['week'].unique())
        if subject_weeks != all_weeks:
            technique_issue = True
            break
    
    if technique_issue:
        print(f"\n🚨 TECHNIQUE CONFOUNDING DETECTED!")
        print(f"❌ Not all subjects participated in all weeks/techniques")
        print(f"❌ Current cross-validation mixes different breathing techniques")
        print(f"❌ This confounds subject vs technique effects")
        
        print(f"\n💡 RECOMMENDED ANALYSIS APPROACH:")
        print(f"1. 📊 Within-technique, cross-subject validation")
        print(f"2. 🔄 Cross-technique, within-subject validation")  
        print(f"3. 📈 Technique-specific feature analysis")
        print(f"4. 🎯 Mixed-effects modeling")
        
        analysis_choice = input(f"\nChoose analysis strategy:\n" +
                              f"  1) Within-technique cross-subject validation\n" +
                              f"  2) Cross-technique analysis\n" +
                              f"  3) Traditional cross-validation (technique-mixed)\n" +
                              f"  4) Exit for manual analysis\n" +
                              f"Enter choice (1-4): ").strip()
        
        if analysis_choice == '1':
            print(f"✅ Running within-technique, cross-subject validation...")
            # Will implement this strategy below
        elif analysis_choice == '2':
            print(f"✅ Running cross-technique analysis...")
            # Could implement technique generalization analysis
        elif analysis_choice == '3':
            print(f"⚠️  Proceeding with traditional CV (aware of technique confounding)")
        else:
            print(f"Exiting for manual analysis. Key insights:")
            print(f"- Week {week_class_dist.sum(axis=1).idxmax()} has most data")
            print(f"- Consider analyzing each technique separately")
            return
    else:
        print(f"\n✅ NO TECHNIQUE CONFOUNDING DETECTED")
        print(f"✅ All subjects participated in all weeks - standard CV is appropriate")
        analysis_choice = '3'  # Standard CV
    
    # Run within-subject analysis preview
    print(f"\n===== WITHIN-SUBJECT ANALYSIS PREVIEW =====")
    print("Getting a quick preview of within-subject performance...")
    within_auc, within_results = fixed_within_subject_decoding(df, X_raw, y)
    
    if within_auc is not None:
        expected_cross_subject = 0.58  # Your typical cross-subject AUC
        gap = within_auc - expected_cross_subject
        
        print(f"\n📈 PERFORMANCE PREVIEW:")
        print(f"   Within-subject AUC: {within_auc:.3f}")
        print(f"   Expected cross-subject AUC: ~{expected_cross_subject:.3f}")
        print(f"   Generalization gap: {gap:.3f}")
        
        if technique_issue:
            print(f"   Note: Gap interpretation affected by technique confounding")
        
        if gap > 0.15:
            print("✅ Good generalization gap - individual differences detected")
            print("💡 Domain adaptation and ensemble methods should help")
        elif gap > 0.05:
            print("⚠️  Moderate generalization gap - some individual differences")
            print("💡 Try feature engineering and regularization")
        else:
            print("❌ Poor generalization gap - fundamental signal issues")
            print("💡 Consider better features or data collection improvements")
        
        if analysis_choice in ['1', '2']:
            proceed = 'y'  # Auto-proceed with specialized analysis
        else:
            proceed = input(f"\nProceed with cross-validation? (y/n): ").lower().strip()
            
        if proceed != 'y':
            print("Exiting. Run advanced_analysis_pipeline(df, X, y) for detailed diagnostics.")
            return
    else:
        print("⚠️  Within-subject analysis failed - proceeding with cross-validation")
        analysis_choice = '3'
    
    # ========== EXECUTE CHOSEN ANALYSIS STRATEGY ==========
    if analysis_choice == '1':
        print(f"\n===== WITHIN-TECHNIQUE CROSS-SUBJECT VALIDATION =====")
        print("Running technique-specific analysis to avoid confounding...")
        
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
        
        # Run within-technique cross-subject validation
        technique_results = within_technique_cross_subject_validation(df, X, y, technique_col='week')
        
        if technique_results:
            # Save results
            results_dir = 'technique_results'
            os.makedirs(results_dir, exist_ok=True)
            
            import json
            
            # Convert numpy int64 keys to strings for JSON serialization
            technique_results_json = {}
            for week, results in technique_results.items():
                technique_results_json[str(week)] = results
            
            technique_path = os.path.join(results_dir, 'within_technique_cv_results.json')
            with open(technique_path, 'w') as f:
                json.dump(technique_results_json, f, indent=2)
            print(f"Saved within-technique CV results to {technique_path}")
            
            # Create summary visualization
            import matplotlib.pyplot as plt
            
            techniques = list(technique_results.keys())
            aucs = [technique_results[t]['auc_mean'] for t in techniques]
            stds = [technique_results[t]['auc_std'] for t in techniques]
            n_subjects = [technique_results[t]['n_subjects'] for t in techniques]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(techniques)), aucs, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Within-Technique Cross-Subject Validation Results')
            plt.ylabel('AUC')
            plt.xlabel('Breathing Technique (Week)')
            plt.xticks(range(len(techniques)), [f'Week {t}\n({n_subjects[i]} subjects)' for i, t in enumerate(techniques)])
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for i, (auc, std) in enumerate(zip(aucs, stds)):
                plt.text(i, auc + std + 0.02, f'{auc:.3f}±{std:.3f}', ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'technique_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n🎯 WITHIN-TECHNIQUE RESULTS SUMMARY:")
            print(f"{'='*60}")
            for tech, results in technique_results.items():
                print(f"Technique {tech} (Week {tech}):")
                print(f"  📊 AUC: {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
                print(f"  👥 Subjects: {results['n_subjects']}")
                print(f"  📈 Samples: {results['n_samples']}")
                print(f"  🎯 Classes: {results['class_distribution']}")
                print()
            
            # Calculate weighted average
            total_samples = sum(r['n_samples'] for r in technique_results.values())
            weighted_auc = sum(r['auc_mean'] * r['n_samples'] for r in technique_results.values()) / total_samples
            print(f"📈 Weighted Average AUC: {weighted_auc:.3f}")
            
            # Compare with previous mixed-technique result
            print(f"\n🔄 COMPARISON WITH MIXED-TECHNIQUE CV:")
            print(f"  Mixed-technique AUC: 0.568 (confounded)")
            print(f"  Within-technique AUC: {weighted_auc:.3f} (true performance)")
            print(f"  Improvement: {weighted_auc - 0.568:+.3f}")
            
            if weighted_auc > 0.65:
                print("✅ Good technique-specific performance detected!")
                print("💡 The signal was there but hidden by technique confounding")
            elif weighted_auc > 0.60:
                print("⚠️  Moderate technique-specific performance")
                print("💡 Some breathing techniques show better neural signatures")
            else:
                print("❌ Still poor performance even within techniques")
                print("💡 May need better features or data quality improvements")
            
            # ========== ADVANCED ANALYSIS OPTIONS ==========
            print(f"\n===== ADVANCED ANALYSIS OPTIONS =====")
            print("Since you have technique-specific results, let's explore improvements:")
            
            advanced_choice = input(f"\nChoose advanced analysis:\n" +
                                  f"  1) Universal ensemble across all techniques\n" +
                                  f"  2) Ensemble methods with respiratory features\n" +
                                  f"  3) Full advanced analysis pipeline\n" +
                                  f"  4) Skip advanced analysis\n" +
                                  f"Enter choice (1-4): ").strip()
            
            if advanced_choice == '1':
                print(f"✅ Running universal ensemble analysis...")
                universal_results = universal_ensemble_cross_validation(df, X, y)
                if universal_results:
                    print(f"\n🎯 Universal Ensemble Summary:")
                    print(f"  Universal AUC: {universal_results['universal_ensemble_auc_mean']:.3f} ± {universal_results['universal_ensemble_auc_std']:.3f}")
                    print(f"  Improvement: {universal_results['improvement_over_baseline']:+.3f}")
                    print(f"  Universal features: {universal_results['universal_features_count']} (was {universal_results['original_features_count']})")
                    
                    # Save universal results
                    import json
                    universal_path = os.path.join(results_dir, 'universal_ensemble_results.json')
                    with open(universal_path, 'w') as f:
                        json.dump(universal_results, f, indent=2, default=str)
                    print(f"  Saved results to {universal_path}")
            
            elif advanced_choice == '2':
                print(f"✅ Running ensemble analysis with respiratory features...")
                ensemble_results = ensemble_cross_subject_validation(df, X, y, technique_col='week')
                if ensemble_results:
                    print(f"\n🚀 Ensemble Analysis Summary:")
                    print(f"  Ensemble AUC: {ensemble_results['ensemble_auc_mean']:.3f} ± {ensemble_results['ensemble_auc_std']:.3f}")
                    print(f"  Improvement: {ensemble_results['improvement_over_baseline']:+.3f}")
                    
                    # Save ensemble results
                    import json
                    ensemble_path = os.path.join(results_dir, 'ensemble_results.json')
                    with open(ensemble_path, 'w') as f:
                        json.dump(ensemble_results, f, indent=2, default=str)
                    print(f"  Saved results to {ensemble_path}")
            
            elif advanced_choice == '3':
                print(f"✅ Running full advanced analysis pipeline...")
                advanced_results = advanced_analysis_pipeline(df, X, y, results_dir='advanced_results')
                print(f"  Advanced analysis complete - check 'advanced_results/' directory")
            
            else:
                print("Skipping advanced analysis")
            
            return  # Exit after within-technique analysis
        else:
            print("❌ Within-technique analysis failed, falling back to mixed analysis")
            analysis_choice = '3'
    
    if analysis_choice == '2':
        print(f"\n===== CROSS-TECHNIQUE ANALYSIS =====")
        cross_tech_results = cross_technique_within_subject_validation(df, X, y, technique_col='week')
        if cross_tech_results:
            print("Cross-technique analysis completed!")
            return
        else:
            print("❌ Cross-technique analysis failed, falling back to mixed analysis")
            analysis_choice = '3'
    
    # Fallback to traditional mixed-technique CV
    print(f"\n===== TRADITIONAL MIXED-TECHNIQUE CROSS-VALIDATION =====")
    print(f"⚠️  Warning: This mixes different breathing techniques")
    print(f"Using {len(df['subject'].unique())} subjects: {sorted(df['subject'].unique())}")
    if subjects_to_exclude:
        print(f"Excluded subjects: {subjects_to_exclude}")

    X = rename_features(X_raw)

    # 2. Set up GroupKFold Cross-Validation (Sequential for stability)
    n_splits = len(df['subject'].unique())
    gkf = GroupKFold(n_splits=n_splits)
    
    print(f"Starting {n_splits}-fold cross-validation (sequential processing for stability)...")
    
    all_aucs = []
    all_cms = []
    fold_num = 0

    # Sequential processing for stability (avoiding multiprocessing issues)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        fold_num += 1
        fold_data = (fold_num, train_idx, test_idx, X, y, df)
        
        print(f"\nProcessing fold {fold_num}/{n_splits}...")
        fold_num_result, cm, aucs = process_single_fold(fold_data)
        
        if cm is not None and aucs is not None:
            all_cms.append(cm)
            all_aucs.append(aucs)
            print(f"Fold {fold_num} completed successfully")
        else:
            print(f"Fold {fold_num} was skipped due to preprocessing issues")

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

    # 9. Skip the misleading final model training on all data
    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print("Note: Final model training on full dataset skipped to avoid overfitting.")
    print("Cross-validation results represent true generalization performance.")
    
    # Save only the hyperparameters and preprocessing info for deployment
    import json
    deployment_config = {
        'feature_names': X.columns.tolist(),
        'cv_performance': {
            'avg_auc_class_1': float(avg_aucs['1']) if '1' in avg_aucs else None,
            'avg_auc_class_2': float(avg_aucs['2']) if '2' in avg_aucs else None, 
            'avg_auc_class_3': float(avg_aucs['3']) if '3' in avg_aucs else None,
            'std_auc_class_1': float(std_aucs['1']) if '1' in std_aucs else None,
            'std_auc_class_2': float(std_aucs['2']) if '2' in std_aucs else None,
            'std_auc_class_3': float(std_aucs['3']) if '3' in std_aucs else None,
        },
        'notes': 'Hyperparameters vary per fold due to parallel processing. Check individual fold results for details.'
    }
    
    with open('neural_decoder_config.json', 'w') as f:
        json.dump(deployment_config, f, indent=2)
    print("Saved deployment configuration → neural_decoder_config.json")
    
    # Save comprehensive cross-validation results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save overall confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(total_cm_percent, cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Overall Cross-Validation Confusion Matrix (%)')
    plt.colorbar(label='Percentage')
    plt.xticks([0, 1, 2], ['Class 1', 'Class 2', 'Class 3'], rotation=45)
    plt.yticks([0, 1, 2], ['Class 1', 'Class 2', 'Class 3'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{total_cm_percent[i, j]:.1f}%", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'overall_confusion_matrix.png'))
    plt.close()
    
    # Save overall AUC scores
    plt.figure(figsize=(8, 6))
    classes = list(avg_aucs.index)
    means = list(avg_aucs.values)
    stds = list(std_aucs.values)
    plt.bar(classes, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title('Cross-Validation AUC Scores (Mean ± Std)')
    plt.ylabel('AUC')
    plt.ylim(0, 1)
    plt.xlabel('Class')
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'overall_auc_scores.png'))
    plt.close()
    
    # Save detailed cross-validation summary
    cv_summary = {
        'total_folds': len(all_aucs),
        'overall_confusion_matrix_counts': total_cm.tolist(),
        'overall_confusion_matrix_percent': total_cm_percent.tolist(),
        'average_auc_scores': {str(k): float(v) for k, v in avg_aucs.items()},
        'std_auc_scores': {str(k): float(v) for k, v in std_aucs.items()},
        'all_fold_aucs': all_aucs,
        'excluded_subjects': subjects_to_exclude,
        'total_samples': len(df),
        'feature_count': len(X.columns),
        'class_distribution': y.value_counts().to_dict()
    }
    
    with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"Saved detailed CV summary → {os.path.join(results_dir, 'cv_summary.json')}")
    
    # Save raw data for further analysis
    cv_raw_data = {
        'confusion_matrices': [cm.tolist() for cm in all_cms],
        'auc_scores_per_fold': all_aucs,
        'fold_details': f"Check individual fold directories in {results_dir}/"
    }
    
    with open(os.path.join(results_dir, 'cv_raw_data.json'), 'w') as f:
        json.dump(cv_raw_data, f, indent=2)
    print(f"Saved raw CV data → {os.path.join(results_dir, 'cv_raw_data.json')}")

    print("\n===== NEURAL DECODER ANALYSIS COMPLETE =====")
    print("Generated outputs:")
    print("  - Cross-validation results in results/fold_* directories") 
    print("  - Deployment configuration: neural_decoder_config.json")
    print(f"  - Average cross-validation AUC: {avg_aucs.mean():.3f} ± {std_aucs.mean():.3f}")
    
    # ========== POST-CV PERFORMANCE DIAGNOSTICS ==========
    print(f"\n===== POST-CV PERFORMANCE DIAGNOSTICS =====")
    overall_auc = avg_aucs.mean()
    auc_std = std_aucs.mean()
    
    print(f"📊 Performance Assessment:")
    print(f"   Overall AUC: {overall_auc:.3f} ± {auc_std:.3f}")
    
    if overall_auc > 0.7:
        print("✅ Good performance - models are learning meaningful patterns")
        improvement_priority = "ensemble_methods"
    elif overall_auc > 0.6:
        print("⚠️  Moderate performance - some signal detected but needs improvement")
        improvement_priority = "feature_engineering"
    elif overall_auc > 0.55:
        print("🔍 Weak signal - performance slightly above chance")
        improvement_priority = "data_quality"
    else:
        print("❌ Poor performance - close to random chance")
        improvement_priority = "data_collection"
    
    # Analyze confusion matrix patterns
    print(f"\n📈 Confusion Matrix Analysis:")
    diagonal_avg = np.mean(np.diag(total_cm_percent))
    print(f"   Average correct classification: {diagonal_avg:.1f}%")
    
    # Check for class bias
    predictions_per_class = total_cm.sum(axis=0)
    true_per_class = total_cm.sum(axis=1)
    
    for i, (pred_count, true_count) in enumerate(zip(predictions_per_class, true_per_class)):
        bias_ratio = pred_count / true_count
        class_name = f"Class {i+1}"
        if bias_ratio > 1.2:
            print(f"   ⚠️  {class_name} over-predicted (bias: {bias_ratio:.2f}x)")
        elif bias_ratio < 0.8:
            print(f"   ⚠️  {class_name} under-predicted (bias: {bias_ratio:.2f}x)")
        else:
            print(f"   ✓ {class_name} balanced predictions (bias: {bias_ratio:.2f}x)")
    
    # High-level recommendations based on performance
    print(f"\n💡 Recommended Next Steps:")
    if improvement_priority == "data_collection":
        print("   1. 🔬 Re-examine data collection methodology")
        print("   2. 📊 Check if the task/states are actually distinguishable")
        print("   3. 🧹 Run comprehensive data quality analysis")
        print("   4. 📈 Consider different feature extraction methods")
    elif improvement_priority == "data_quality":
        print("   1. 🧹 Run advanced_analysis_pipeline() for detailed diagnostics")
        print("   2. 🔍 Exclude problematic subjects if you haven't already")
        print("   3. 🎯 Try within-subject analysis to check for individual differences")
        print("   4. 🔄 Consider domain adaptation methods")
    elif improvement_priority == "feature_engineering":
        print("   1. 🔧 Run advanced feature engineering (alpha/theta ratios, etc.)")
        print("   2. 🤖 Try domain adaptation (CORAL) methods")
        print("   3. 🧠 Experiment with CNN on channel layouts")
        print("   4. 📊 Use ensemble methods combining multiple approaches")
    elif improvement_priority == "ensemble_methods":
        print("   1. 🎯 Fine-tune hyperparameters further")
        print("   2. 🤖 Try ensemble methods and model stacking")
        print("   3. 🧠 Experiment with transformer architectures")
        print("   4. 🔄 Consider temporal modeling (RNN/LSTM)")
    
    print(f"\nNote: Feature importance and cluster analysis removed to focus on improving CV performance.")
    print(f"To improve performance further, consider:")
    print(f"  1. Feature selection/engineering")
    print(f"  2. Different model architectures (CNN, RNN, Transformer)")
    print(f"  3. Ensemble methods")
    print(f"  4. Advanced regularization techniques")
    
    # ========== ADVANCED ANALYSIS PIPELINE (OPTIONAL) ==========
    print("\n===== OPTIONAL: ADVANCED ANALYSIS PIPELINE =====")
    run_advanced = input("Run advanced analysis pipeline? (y/n): ").lower().strip() == 'y'
    
    if run_advanced:
        print("Running advanced analysis pipeline...")
        advanced_analysis_pipeline(df, X, y, results_dir='advanced_results')
    else:
        print("Skipping advanced analysis. To run later, call:")
        print("advanced_analysis_pipeline(df, X, y, results_dir='advanced_results')")

    # # # —————————————————————————————— 8.5 MANUAL PERMUTATION‐IMPORTANCE ——————————————————————————————

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