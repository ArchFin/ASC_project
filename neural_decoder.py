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
        epochs=50, batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    probs = model.predict(X_test)
    preds = probs.argmax(axis=1)
    true = Y_test.argmax(axis=1)

    # Confusion matrix
    cm = confusion_matrix(true, preds)
    print("Confusion Matrix:\n", cm)

    # Plot & save confusion matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = range(len(label_names))
    plt.xticks(ticks, label_names, rotation=45)
    plt.yticks(ticks, label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in ticks:
        for j in ticks:
            plt.text(j, i, cm[i, j], ha='center', va='center')
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
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
    auc_path = os.path.join(out_dir, 'auc_per_class.png')
    plt.tight_layout()
    plt.savefig(auc_path)
    plt.close()
    print(f"Saved AUC bar chart to {auc_path}")

    return model, aucs, cm


def main():
    csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete.csv'
    drop_cols = ['subject', 'week', 'run', 'epoch', 'number']

    X, y = load_data(csv_path=csv_path, drop_cols=drop_cols)
    X = select_features(X, pattern_keep=['glob'])
    X_train, X_test, Y_train, Y_test, le, label_names = preprocess(X, y)

    model, aucs, cm = train_and_evaluate(
        X_train, Y_train, X_test, Y_test,
        label_names=label_names,
        out_dir='/Users/a_fin/Desktop/Year 4/Project/Data'
    )

    model.save('neural_decoder_model.h5')
    joblib.dump(le, 'label_encoder.pkl')
    print('Saved model to neural_decoder_model.h5 and encoder to label_encoder.pkl')


if __name__ == '__main__':
    main()
