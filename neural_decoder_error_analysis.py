#!/usr/bin/env python3
"""
error_analysis.py

Run a suite of “error analysis” experiments on top of neural_decoder.py:

  1. Train/Test Split Variability (5-fold CV AUC)
  2. Random Weight Initialization (10 different seeds)
  3. Hyperparameter Tuning Uncertainty (Δ between best & 2nd-best CV AUC)
  4. Imputation Threshold Impact (accuracy shift for thresholds 2.0,2.5,3.0)
  5. Class-Weighting Effects (AUC gain/drop with different weighting schemes)

Usage:
    python error_analysis.py

Make sure neural_decoder.py is in the same folder, and adjust CSV path if needed.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Append current directory so we can import neural_decoder.py
sys.path.append(os.path.dirname(__file__))

# Import everything we need from neural_decoder.py
from neural_decoder import (
    load_data,
    select_features,
    clean_impute,
    preprocess,
    build_model,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ================================================================
# 0. Settings & helper functions
# ================================================================
CSV_PATH = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete_2.csv'
DROP_COLS = ['subject', 'week', 'run', 'epoch', 'number']
PATTERN_KEEP = ['glob_chans']  # same as in neural_decoder.py

# If you want to speed things up, you can reduce epochs in build_model.fit calls.
EPOCHS = 100
BATCH_SIZE = 64
VERBOSE = 0  # change to 1 if you want to see training logs

def make_data():
    """
    1. Load data
    2. Select “glob” features
    3. Rename columns just as in main()
    4. Preprocess into X_train, X_test, Y_train, Y_test (one-hot), label encoder, label_names
    """
    X, y = load_data(csv_path=CSV_PATH, drop_cols=DROP_COLS)
    X = select_features(X, pattern_keep=PATTERN_KEEP)

    # Replicate the column-renaming from main()
    columns = X.columns.copy()
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
    X.drop(columns=columns, inplace=True)

    # Now preprocess exactly as in neural_decoder.py
    X_train, X_test, Y_train, Y_test, le, label_names = preprocess(X, y)
    return X, X_train, X_test, Y_train, Y_test, le, label_names

# ================================================================
# 1. Train/Test Split Variability (5-fold CV AUC)
# ================================================================
def experiment_train_test_variability(X, y, n_splits=5, random_state=42):
    """
    Perform a StratifiedKFold (n_splits) CV on (X, y), training a fresh model
    each fold with default hyperparameters. Return mean ± std of AUC.
    """
    # Step 1: Clean & impute X exactly as preprocess does, then scale
    X_clean = clean_impute(X)
    y_clean = y.loc[X_clean.index].copy()

    # Shuffle & scale
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler
    Xs, ys = shuffle(X_clean, y_clean, random_state=random_state)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)

    # Label-encode & produce one-hot
    le_cv = LabelEncoder()
    y_enc = le_cv.fit_transform(ys)
    num_classes = len(le_cv.classes_)
    from keras._tf_keras.keras.utils import to_categorical
    Y_cat = to_categorical(y_enc)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []

    for train_indices, test_indices in skf.split(Xs, y_enc):
        X_tr, X_te = Xs[train_indices], Xs[test_indices]
        y_tr_cat, y_te_cat = Y_cat[train_indices], Y_cat[test_indices]
        y_te_int = y_enc[test_indices]

        # Build & train a fresh model (default hyperparameters)
        model = build_model(input_dim=X_tr.shape[1], num_classes=num_classes)
        model.fit(
            X_tr, y_tr_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_te, y_te_cat),
            verbose=VERBOSE
        )

        # Predict probabilities on fold's test set
        probs = model.predict(X_te)
        # Compute one-vs-rest AUC for each class and average
        auc_per_class = []
        for c in range(num_classes):
            auc_per_class.append(
                roc_auc_score((y_te_int == c).astype(int), probs[:, c])
            )
        fold_auc = np.mean(auc_per_class)
        auc_scores.append(fold_auc)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"[Train/Test Split Variability] {n_splits}-fold CV AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    return mean_auc, std_auc

# ================================================================
# 2. Random Weight Initialization (10 seeds)
# ================================================================
def experiment_random_initialization(X_train, Y_train, X_test, Y_test, n_seeds=10):
    """
    Fix X_train/X_test split. For each seed in range(n_seeds):
      - set np.random + tf random seed
      - build & train model on (X_train,Y_train), evaluate AUC on X_test
    Return mean ± std of test AUC across seeds.
    """
    # Convert one-hot Y to integer labels for computing roc_auc
    y_test_int = Y_test.argmax(axis=1)
    num_classes = Y_train.shape[1]

    def set_all_seeds(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)

    auc_scores = []
    for seed in range(n_seeds):
        set_all_seeds(seed)
        # Build & train fresh model
        model = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
        model.fit(
            X_train, Y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, Y_test),
            verbose=VERBOSE
        )
        # Predict on X_test
        probs = model.predict(X_test)
        # Compute mean one-vs-rest AUC
        auc_per_class = [
            roc_auc_score((y_test_int == c).astype(int), probs[:, c])
            for c in range(num_classes)
        ]
        auc_scores.append(np.mean(auc_per_class))

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"[Random Init Variability] {n_seeds} seeds AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    return mean_auc, std_auc

# ================================================================
# 3. Hyperparameter Tuning Uncertainty
# ================================================================
def experiment_hyperparameter_uncertainty(X, y, param_grid=None, cv_splits=3, random_state=0):
    """
    Run a manual grid search over param_grid. For each config:
      - do cv_splits-stratified folds,
      - train & compute fold AUCs, then take mean.
    Return delta = best_mean - second_best_mean.
    """
    if param_grid is None:
        param_grid = {
            'units': [32, 64, 128],
            'lr': [1e-3, 1e-4],
            'activation': ['relu', 'tanh'],
            'dropout_rate': [0.3, 0.5]
        }

    # 1) Clean & impute X; shuffle; scale, encode labels exactly as in experiment #1
    X_clean = clean_impute(X)
    y_clean = y.loc[X_clean.index].copy()
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler
    Xs, ys = shuffle(X_clean, y_clean, random_state=random_state)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    le_hp = LabelEncoder()
    y_enc = le_hp.fit_transform(ys)
    num_classes = len(le_hp.classes_)
    from keras._tf_keras.keras.utils import to_categorical
    Y_cat = to_categorical(y_enc)

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []  # list of (mean_auc, std_auc, config)

    for cfg in ParameterGrid(param_grid):
        fold_aucs = []
        for train_idx, val_idx in skf.split(Xs, y_enc):
            X_tr, X_val = Xs[train_idx], Xs[val_idx]
            Y_tr, Y_val = Y_cat[train_idx], Y_cat[val_idx]
            y_val_int = y_enc[val_idx]

            # Build model with these hyperparameters
            def build_with_cfg():
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(X_tr.shape[1],)),
                    tf.keras.layers.Dense(cfg['units'], activation=cfg['activation'], kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(cfg['dropout_rate']),
                    tf.keras.layers.Dense(cfg['units']//2, activation=cfg['activation'], kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(cfg['dropout_rate']/2),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                opt = tf.keras.optimizers.Adam(learning_rate=cfg['lr'])
                model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                return model

            model = build_with_cfg()
            model.fit(
                X_tr, Y_tr,
                epochs=EPOCHS//2,    # fewer epochs per CV run to speed up
                batch_size=BATCH_SIZE,
                validation_data=(X_val, Y_val),
                verbose=VERBOSE
            )
            # Evaluate on validation fold
            probs = model.predict(X_val)
            auc_per_class = [
                roc_auc_score((y_val_int == c).astype(int), probs[:, c])
                for c in range(num_classes)
            ]
            fold_aucs.append(np.mean(auc_per_class))

        results.append((np.mean(fold_aucs), np.std(fold_aucs), cfg))

    # Sort by mean AUC descending
    results.sort(key=lambda x: x[0], reverse=True)
    best_mean, best_std, best_cfg = results[0]
    second_mean, second_std, second_cfg = results[1]
    delta = best_mean - second_mean

    print(f"[Hyperparam Uncertainty] Best mean AUC: {best_mean:.3f}, 2nd best: {second_mean:.3f} → Δ = {delta:.3f}")
    print(f"  Best cfg: {best_cfg}")
    print(f"  2nd-best cfg: {second_cfg}")
    return delta, best_cfg, second_cfg

# ================================================================
# 4. Imputation Threshold Impact
# ================================================================
def experiment_imputation_threshold(X, y, thresholds=[2.0, 2.5, 3.0], test_size=0.2, random_state=0):
    """
    For each threshold in `thresholds`, we:
      - clean_impute(X, threshold=thresh)
      - preprocess (shuffle/scale/one-hot) and do a single hold-out train/test split
      - train a default model on train, evaluate accuracy on test
    Return the accuracy list and print max shift between adjacent thresholds.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle

    # Collect (threshold → test accuracy)
    acc_list = []

    for thresh in thresholds:
        # 1) clean/impute using given threshold
        X_clipped = clean_impute(X, threshold=thresh)
        y_clipped = y.loc[X_clipped.index].copy()

        # 2) shuffle + scale
        Xs, ys = shuffle(X_clipped, y_clipped, random_state=random_state)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)

        # 3) encode labels to one-hot
        le_it = LabelEncoder()
        y_enc = le_it.fit_transform(ys)
        num_classes = len(le_it.classes_)
        from keras._tf_keras.keras.utils import to_categorical
        Y_cat = to_categorical(y_enc)

        # 4) hold-out split
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            Xs, Y_cat, test_size=test_size, random_state=random_state, stratify=y_enc
        )
        y_te_int = Y_te.argmax(axis=1)

        # 5) train a default model
        model = build_model(input_dim=X_tr.shape[1], num_classes=num_classes)
        model.fit(
            X_tr, Y_tr,
            epochs=EPOCHS//2,
            batch_size=BATCH_SIZE,
            validation_data=(X_te, Y_te),
            verbose=VERBOSE
        )
        # 6) evaluate accuracy on X_te
        probs = model.predict(X_te)
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_te_int, preds)
        acc_list.append(acc)
        print(f"[Imputation Threshold] σ={thresh:.1f} → test accuracy = {acc*100:.2f}%")

    # Compute shifts between adjacent thresholds
    shifts = [abs(acc_list[i] - acc_list[i-1]) * 100 for i in range(1, len(acc_list))]
    max_shift = max(shifts)
    print(f"  → Max accuracy shift between thresholds = {max_shift:.2f}%")
    return acc_list, shifts

# ================================================================
# 5. Class-Weighting Effects
# ================================================================
def experiment_class_weighting(X_train, Y_train, X_test, Y_test):
    """
    Train with:
      1) no weighting
      2) sklearn 'balanced' weights
      3) a manual weight scheme (e.g. invert-frequency)
    Measure test AUC each time, report gain/drop relative to baseline (no weighting).
    """
    num_classes = Y_train.shape[1]
    y_tr_int = Y_train.argmax(axis=1)
    y_te_int = Y_test.argmax(axis=1)

    # 1) Baseline (no class_weight)
    model_base = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
    model_base.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        verbose=VERBOSE
    )
    probs_base = model_base.predict(X_test)
    aucs_base = [roc_auc_score((y_te_int == c).astype(int), probs_base[:, c]) for c in range(num_classes)]
    mean_auc_base = np.mean(aucs_base)
    print(f"[Class-Weighting] Baseline (no weights) AUC = {mean_auc_base:.3f}")

    # 2) “balanced” from sklearn
    weights_sklearn = compute_class_weight('balanced', classes=np.unique(y_tr_int), y=y_tr_int)
    cw_balanced = {i: float(w) for i, w in enumerate(weights_sklearn)}
    model_bal = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
    model_bal.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        class_weight=cw_balanced,
        verbose=VERBOSE
    )
    probs_bal = model_bal.predict(X_test)
    aucs_bal = [roc_auc_score((y_te_int == c).astype(int), probs_bal[:, c]) for c in range(num_classes)]
    mean_auc_bal = np.mean(aucs_bal)
    diff_bal = mean_auc_bal - mean_auc_base
    print(f"  → “balanced” weights mean AUC = {mean_auc_bal:.3f} (Δ = {diff_bal:+.3f})")

    # 3) Manual: e.g. inverse to class frequency (normalized)
    class_counts = np.bincount(y_tr_int)
    inv_freq = class_counts.max() / class_counts
    cw_manual = {i: float(inv_freq[i]) for i in range(num_classes)}
    model_man = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
    model_man.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        class_weight=cw_manual,
        verbose=VERBOSE
    )
    probs_man = model_man.predict(X_test)
    aucs_man = [roc_auc_score((y_te_int == c).astype(int), probs_man[:, c]) for c in range(num_classes)]
    mean_auc_man = np.mean(aucs_man)
    diff_man = mean_auc_man - mean_auc_base
    print(f"  → manual inv-freq weights mean AUC = {mean_auc_man:.3f} (Δ = {diff_man:+.3f})")

    # Collect the max gain/drop
    all_diffs = [diff_bal, diff_man]
    max_gain = max(all_diffs)
    max_drop = min(all_diffs)
    print(f"  → AUC gain/drop (best vs worst) = +{max_gain:.3f} / {max_drop:.3f}")
    return mean_auc_base, (diff_bal, diff_man)

# ================================================================
# Main: run all experiments in sequence
# ================================================================
if __name__ == '__main__':
    # ——————————————————————————————————————————————————————————————
    # 0. Build & fetch data splits
    # ——————————————————————————————————————————————————————————————
    print("\n→ Loading & preprocessing data …\n")
    X_full, X_train, X_test, Y_train, Y_test, label_encoder, label_names = make_data()

    # ——————————————————————————————————————————————————————————————
    # 1. Train/Test Split Variability
    # ——————————————————————————————————————————————————————————————
    print("\n===== 1. Train/Test Split Variability =====\n")
    # We need `y` aligned with X_full, so we reload y from the CSV:
    _, y_full = load_data(csv_path=CSV_PATH, drop_cols=DROP_COLS)
    auc_mean_cv, auc_std_cv = experiment_train_test_variability(X_full, y_full, n_splits=5, random_state=42)

    # ——————————————————————————————————————————————————————————————
    # 2. Random Weight Initialization
    # ——————————————————————————————————————————————————————————————
    print("\n===== 2. Random Weight Initialization =====\n")
    auc_mean_seed, auc_std_seed = experiment_random_initialization(X_train, Y_train, X_test, Y_test, n_seeds=10)

    # ——————————————————————————————————————————————————————————————
    # 3. Hyperparameter Tuning Uncertainty
    # ——————————————————————————————————————————————————————————————
    print("\n===== 3. Hyperparameter Tuning Uncertainty =====\n")
    # We'll use a smaller grid to keep runtime reasonable:
    hp_param_grid = {
        'units': [32, 64, 128],
        'lr': [1e-3, 1e-4],
        'activation': ['relu', 'tanh'],
        'dropout_rate': [0.3, 0.5]
    }
    delta_auc_hp, best_cfg, second_cfg = experiment_hyperparameter_uncertainty(X_full, y_full, param_grid=hp_param_grid, cv_splits=3, random_state=0)

    # ——————————————————————————————————————————————————————————————
    # 4. Imputation Threshold Impact
    # ——————————————————————————————————————————————————————————————
    print("\n===== 4. Imputation Threshold Impact =====\n")
    thr_list = [2.0, 2.5, 3.0]
    acc_list, acc_shifts = experiment_imputation_threshold(X_full, y_full, thresholds=thr_list, test_size=0.2, random_state=42)

    # ——————————————————————————————————————————————————————————————
    # 5. Class-Weighting Effects
    # ——————————————————————————————————————————————————————————————
    print("\n===== 5. Class-Weighting Effects =====\n")
    base_auc_cw, diffs_cw = experiment_class_weighting(X_train, Y_train, X_test, Y_test)

    # ——————————————————————————————————————————————————————————————
    # Summary: print a LaTeX-friendly row for each experiment
    # ——————————————————————————————————————————————————————————————
    print("\n\n===================================")
    print("SUMMARY (copy these into your LaTeX table)")
    print("===================================\n")

    # 1. Train/Test Split:
    print(f"Train/test split: AUC variation: {auc_mean_cv:.3f} ± {auc_std_cv:.3f}")

    # 2. Random init:
    print(f"Random weight initialization: AUC variation: {auc_mean_seed:.3f} ± {auc_std_seed:.3f} over 10 seeds")

    # 3. Hyperparam:
    print(f"Hyperparameter tuning: AUC change: {delta_auc_hp:.3f} between best & second‐best")

    # 4. Imputation threshold:
    # We'll report the maximum shift we observed when threshold changes by 0.5
    # (acc_shifts already in %)
    max_shift_pct = max(acc_shifts)
    print(f"Imputation threshold: Classification accuracy shifts by {max_shift_pct:.2f}\\% when threshold varies by 0.5 σ")

    # 5. Class-weighting:
    gain = max([d for d in diffs_cw if d > 0], default=0.0)
    drop = min([d for d in diffs_cw if d < 0], default=0.0)
    print(f"Class-weighting: AUC gain/drop: {gain:+.3f}/{abs(drop):.3f} when using alternative schemes")

    print("\n(End of summary)")
