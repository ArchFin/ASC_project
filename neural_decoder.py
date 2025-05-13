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
  4. Builds and trains a feed-forward neural network
  5. Outputs performance metrics and saves model + label encoder
"""
import tensorflow as tf
print(tf.__version__)

import sys
print(f"Python executable: {sys.executable}")
print(f"Version info: {sys.version_info}")
import argparse
import pandas as pd
import numpy as np
import keras._tf_keras.keras.models
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Input, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.utils import to_categorical
import keras_tuner as kt
from sklearn.utils import resample, shuffle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_data(csv_path, drop_cols=None, target_col='transition_label'):
    df = pd.read_csv(csv_path)
    if drop_cols:
        df = df.drop(drop_cols, axis=1, errors='ignore')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def select_features(X, pattern_keep=None):
    if not pattern_keep:
        return X
    cols = [c for c in X.columns if any(p in c for p in pattern_keep)]
    return X[cols]


def clean_impute(X, threshold=2.5):
    mask_allnan = X.isna().all(axis=1)
    X = X[~mask_allnan]
    for col in X.columns:
        mu, sigma = X[col].mean(), X[col].std()
        X[col] = np.where(
            np.abs(X[col] - mu) <= threshold * sigma,
            X[col],
            np.nan
        )
    X = X.fillna(method='ffill').fillna(method='bfill')
    X = X.dropna(axis=0)
    return X


def preprocess(X, y, test_size=0.2, random_state=42):
    X_clean = clean_impute(X)
    y_clean = y.loc[X_clean.index]
    Xs, ys = shuffle(X_clean, y_clean, random_state=random_state)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    le = LabelEncoder()
    y_enc = le.fit_transform(ys)
    Y = to_categorical(y_enc)
    X_train, X_test, Y_train, Y_test = train_test_split(
        Xs, Y, test_size=test_size, random_state=random_state,
        stratify=y_enc
    )
    return X_train, X_test, Y_train, Y_test, le, le.classes_


def build_model(input_dim, num_classes, hp=None):
    if hp:
        units = hp.Int('units', 32, 256, step=32)
        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        activation = hp.Choice('activation', ['relu', 'tanh'])
    else:
        units, lr, activation = 64, 1e-3, 'relu'

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation=activation, kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def tune_hyperparameters(X_train, Y_train, X_val, Y_val):
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
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(X_train.shape[1], Y_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    model.fit(
        X_train, Y_train,
        epochs=250, batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    probs = model.predict(X_test)
    preds = probs.argmax(axis=1)
    true = Y_test.argmax(axis=1)

    # Confusion matrix (normalized to percentages)
    cm = confusion_matrix(true, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    print("Confusion Matrix (counts):\n", cm)
    print("Confusion Matrix (percentages):\n", np.round(cm_percent, 2))

    # Plot & save confusion matrix (percentages)
    plt.figure(figsize=(6,5))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Confusion Matrix')
    plt.colorbar(label='Percentage')
    ticks = range(len(label_names))
    plt.xticks(ticks, label_names, rotation=45)
    plt.yticks(ticks, label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in ticks:
        for j in ticks:
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha='center', va='center', color='black')
    cm_path = os.path.join(out_dir, 'confusion_matrix_0.6.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Compute and report AUC per class
    aucs = {}
    for i, label in enumerate(label_names):
        aucs[str(label)] = roc_auc_score((true == i).astype(int), probs[:, i])
    print("AUC per class:", aucs)

    # Plot & save AUC bar chart
    plt.figure(figsize=(6,4))
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

def analyze_cluster_averages(X, model, label_names, out_dir=f'/Users/a_fin/Desktop/Year 4/Project/Data/'):
    # Ensure X is a DataFrame with original feature names
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with original feature names.")
    
    preds = model.predict(X.values).argmax(axis=1)
    X_clustered = X.copy()
    # Use label_names for cluster labels
    X_clustered['Cluster'] = [label_names[p] if p < len(label_names) else f'Cluster_{p}' for p in preds]

    cluster_averages = X_clustered.groupby('Cluster').mean()
    cluster_medians = X_clustered.groupby('Cluster').median()
    cluster_std = X_clustered.groupby('Cluster').std()

    print("\nCluster Averages:\n", cluster_averages)
    print("\nCluster Medians:\n", cluster_medians)
    print("\nCluster Standard Deviations:\n", cluster_std)

    cluster_averages.T.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Neural Features per Cluster')
    plt.xlabel('Neural Feature')
    plt.ylabel('Average Value')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'avg_neural_features.png'))
    plt.close()

    cluster_medians.T.plot(kind='bar', figsize=(12, 8), colormap='viridis')
    plt.title('Median Neural Features per Cluster')
    plt.xlabel('Neural Feature')
    plt.ylabel('Median Value')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'median_neural_features.png'))
    plt.close()

    cluster_std.T.plot(kind='bar', figsize=(12, 8), colormap='coolwarm')
    plt.title('Standard Deviation of Neural Features per Cluster')
    plt.xlabel('Neural Feature')
    plt.ylabel('Std')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'std_neural_features.png'))
    plt.close()

    # Heatmap of feature means by cluster (relative values per feature)
    rel_cluster_averages = cluster_averages.T.div(cluster_averages.T.mean(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(rel_cluster_averages, annot=True, cmap='vlag', center=100, fmt='.1f')
    plt.title('Heatmap of Feature Means by Cluster (Relative to Feature Mean)')
    plt.xlabel('Cluster')
    plt.ylabel('Neural Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'heatmap_feature_means_by_cluster.png'))
    plt.close()

    # Top 5 features that differ most between clusters (by max relative percent difference)
    feature_means = cluster_averages.mean(axis=1)
    max_vals = cluster_averages.max(axis=0)
    min_vals = cluster_averages.min(axis=0)
    # Avoid division by zero
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

    plt.figure(figsize=(14, 8))
    X_melt = pd.melt(X_clustered, id_vars=['Cluster'], var_name='Feature', value_name='Value')
    sns.boxplot(x='Cluster', y='Value', data=X_melt, palette='Set3')
    plt.title('Distribution of Neural Features by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Feature Value')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_distribution.png'))
    plt.close()

    for cluster in X_clustered['Cluster'].unique():
        plt.figure(figsize=(10, 8))
        corr = X_clustered[X_clustered['Cluster'] == cluster].drop(columns='Cluster').corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap - Cluster {cluster}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'correlation_cluster_{cluster}.png'))
        plt.close()

    # Pairwise cluster comparisons
    from sklearn.metrics import confusion_matrix
    cluster_labels = list(X_clustered['Cluster'].unique())
    for i in range(len(cluster_labels)):
        for j in range(i+1, len(cluster_labels)):
            clust_a = cluster_labels[i]
            clust_b = cluster_labels[j]
            mask = X_clustered['Cluster'].isin([clust_a, clust_b])
            X_pair = X_clustered[mask]
            preds_pair = X_pair['Cluster']
            y_true = preds_pair.values
            y_pred = preds_pair.values  # Since these are predicted clusters
            cm = confusion_matrix(y_true, y_pred, labels=[clust_a, clust_b])
            cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
            print(f"\nConfusion Matrix for {clust_a} vs {clust_b} (predicted clusters):\n", cm)
            # Plot and save confusion matrix (percentages)
            plt.figure(figsize=(5,4))
            plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
            plt.title(f'Confusion Matrix: {clust_a} vs {clust_b}')
            plt.colorbar(label='Percentage')
            plt.xticks([0,1], [clust_a, clust_b], rotation=45)
            plt.yticks([0,1], [clust_a, clust_b])
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
            # Mean and relative difference
            means = X_pair.groupby('Cluster').mean()
            diff = means.loc[clust_a] - means.loc[clust_b]
            rel_diff = (diff / means.loc[[clust_a, clust_b]].mean()) * 100
            # Plot absolute mean difference
            plt.figure(figsize=(10, 6))
            diff.abs().sort_values(ascending=False).plot(kind='bar', color='purple')
            plt.title(f'Absolute Mean Difference: {clust_a} vs {clust_b}')
            plt.ylabel('Absolute Mean Difference')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'mean_diff_{clust_a}_vs_{clust_b}.png'))
            plt.close()
            # Plot relative mean difference
            plt.figure(figsize=(10, 6))
            rel_diff.abs().sort_values(ascending=False).plot(kind='bar', color='teal')
            plt.title(f'Relative Mean Difference (%): {clust_a} vs {clust_b}')
            plt.ylabel('Relative Mean Difference (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'rel_mean_diff_{clust_a}_vs_{clust_b}.png'))
            plt.close()

    return cluster_averages, cluster_medians, cluster_std

def main():
    csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete_2.csv'
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number']

    X, y = load_data(csv_path=csv_path, drop_cols=drop_cols)
    X = select_features(X, pattern_keep=['glob'])
    columns = X.columns
    #Retitle columns in X
    X['psd global offset'] = X['psd_metrics_combined_avg__Offset_glob_chans']
    X['wSMI 2 global'] = X['wsmi_2_global_2__wSMI_glob_chans']
    X['psd global gamma'] = X['psd_metrics_combined_avg__Gamma_Power_glob_chans']
    X['wSMI 4 global'] = X['wsmi_4_global_2__wSMI_glob_chans']
    X['psd global beta '] = X['psd_metrics_combined_avg__Beta_Power_glob_chans']
    X['psd global exponent '] = X['psd_metrics_combined_avg__Exponent_glob_chans']
    X['wSMI 8 global'] = X['wsmi_8_global_2__wSMI_glob_chans']
    X['psd global delta '] = X['psd_metrics_combined_avg__Delta_Power_glob_chans']
    X['psd global alpha '] = X['psd_metrics_combined_avg__Alpha_Power_glob_chans']
    X['wSMI 1 global'] = X['wsmi_1_global_2__wSMI_glob_chans']
    X['LZC global'] = X['lz_metrics_combined__glob_chans']
    X['SMI 1 global'] = X['wsmi_1_global_2__SMI_glob_chans']
    X['SMI 2 global'] = X['wsmi_2_global_2__SMI_glob_chans']
    X['SMI 8 global'] = X['wsmi_8_global_2__SMI_glob_chans']
    X['SMI 4 global'] = X['wsmi_4_global_2__SMI_glob_chans']
    X['psd global theta '] = X['psd_metrics_combined_avg__Theta_Power_glob_chans']
    X['pe 2 global'] = X['pe_metrics_combined_2_wide__glob_chans']
    X['pe 1 global'] = X['pe_metrics_combined_1_wide__glob_chans']
    X['pe 4 global'] = X['pe_metrics_combined_4_wide__glob_chans']
    X['pe 8 global'] = X['pe_metrics_combined_8_wide__glob_chans']
    X.drop(columns=columns, inplace=True)

    X_train, X_test, Y_train, Y_test, le, label_names = preprocess(X, y)

    model, aucs, cm = train_and_evaluate(
        X_train, Y_train, X_test, Y_test,
        label_names=label_names,
        out_dir='/Users/a_fin/Desktop/Year 4/Project/Data'
    )

    cluster_averages, cluster_medians, cluster_std = analyze_cluster_averages(X, model, label_names)

    model.save('neural_decoder_model.h5')
    joblib.dump(le, 'label_encoder.pkl')
    print('Saved model to neural_decoder_model.h5 and encoder to label_encoder.pkl')


if __name__ == '__main__':
    main()
