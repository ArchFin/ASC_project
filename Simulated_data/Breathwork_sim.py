#!/usr/bin/env python3
"""
generate_synthetic_ndt.py

Fully self-contained pipeline to generate scientifically rigorous synthetic data
from NDT_all_12thDec_uncleaned.csv using SDV’s CTGAN, with metadata, hyperparameter
tuning, conditional sampling, evaluation, and post‐processing (clamping & missingness).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
import logging
import json
import argparse
import itertools

# -----------------------------
# USER CONFIGURATION
# -----------------------------
# Define new subjects, weeks, and sessions for conditional sampling
NEW_SUBJECTS = ['s30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38'] # Example new subjects
WEEKS        = [1, 2, 3, 4]
SESSIONS     = [1, 2, 3, 4, 5, 6, 7]
ROWS_PER_CONDITION = 136 # Number of synthetic rows to generate for each combination

INPUT_CSV     = '/Users/a_fin/Desktop/Year 4/Project/Data/NDT_all_12thDec_uncleaned.csv'
OUTPUT_CSV    = '/Users/a_fin/Desktop/Year 4/Project/Data/synthetic_ndt.csv'
RANDOM_STATE  = 42
N_SYNTH_ROWS  = 25000
CTGAN_EPOCHS  = 500
HYPERPARAM_TUNE = True   # Set True to run hyperparameter tuning (slower)
TUNE_GRID = {
    'embedding_dim':   [32, 64],
    'generator_dim':   [[256, 256], [512, 512]],
    'discriminator_dim': [[256, 256], [512, 512]],
    'batch_size':      [500, 1000]
}

# -----------------------------
# SETUP
# -----------------------------
np.random.seed(RANDOM_STATE)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------
# 1. LOAD & CLEAN
# -----------------------------
def load_data(path):
    """Loads data and performs initial column drops."""
    logging.info(f"Loading data from '{path}'...")
    df = pd.read_csv(path)
    df = df.drop(columns=['Condition', 'fo_lzsum', 'ff_lzsum', 'oo_lzsum', 'global_lzsum', 'fo_lzc', 'ff_lzc', 'oo_lzc', 'global_lzc', 'epochs_over_50', 'before_retention', 'after_retention', 'time_of_session', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'offset', 'exponent', 'Cluster', 'BH_Num', 'before_or_after'])
    return df

def clean_data_for_training(df):
    """Cleans data by handling missing values for model training."""
    logging.info("Cleaning data for training...")
    df_clean = df.copy()
    # Drop rows with >50% missing values
    thresh = df_clean.shape[1] * 0.5
    df_clean = df_clean.dropna(thresh=thresh).reset_index(drop=True)
    # Impute numeric columns with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    logging.info("Finished cleaning data.")
    return df_clean

# -----------------------------
# 2. BUILD METADATA
# -----------------------------
def build_metadata(df):
    logging.info("Step 2: Building metadata...")
    md = SingleTableMetadata()
    # Numerical
    for col in df.select_dtypes(include=[np.number]).columns:
        md.add_column(column_name=col, sdtype='numerical')
    # Categorical (adjust list as needed)
    for col in ['Subject', 'Week', 'Session']:
        if col in df.columns:
            md.add_column(column_name=col, sdtype='categorical')
    # (optional) mark any ordinal with 'categorical' + ordering
    logging.info("Step 2: Finished building metadata.")
    return md

# -----------------------------
# 3. HYPERPARAMETER TUNING
# -----------------------------
def tune_ctgan(df, metadata):
    logging.info("Step 3: Starting hyperparameter tuning...")
    best_score = -np.inf
    best_cfg   = None
    for i, cfg in enumerate(ParameterGrid(TUNE_GRID)):
        logging.info(f"Tuning run {i+1}/{len(list(ParameterGrid(TUNE_GRID)))} with params: {cfg}")
        model = CTGANSynthesizer(
            metadata=metadata,
            epochs=CTGAN_EPOCHS,
            cuda='mps',
            verbose=1,
            **cfg
        )
        model.fit(df)
        
        # Generate a synthetic sample to evaluate the model
        synth_data = model.sample(num_rows=df.shape[0])
        quality_report = evaluate_quality(df, synth_data, metadata)
        score = quality_report.get_score()

        logging.info(f"Score: {score:.4f}")
        if score > best_score:
            best_score, best_cfg = score, cfg
    logging.info(f"Step 3: Finished hyperparameter tuning. Best config: {best_cfg}, score: {best_score:.4f}")
    return best_cfg

# -----------------------------
# 4. TRAIN FINAL CTGAN
# -----------------------------
def train_final_ctgan(df, metadata, cfg):
    logging.info("Step 4: Training final CTGAN model...")
    model = CTGANSynthesizer(
        metadata=metadata,
        epochs=CTGAN_EPOCHS,
        cuda='mps',
        verbose=1,
        **cfg
    )
    model.fit(df)
    logging.info("Step 4: Finished training final model.")
    return model

# -----------------------------
# 5. SAMPLING
# -----------------------------
def sample_data(model, n_samples):
    """Sample synthetic data from the trained model."""
    logging.info(f"Step 5: Generating {n_samples} synthetic rows...")
    synth = model.sample(num_rows=n_samples)
    logging.info(f"Step 5: Finished generating synthetic data.")
    return synth

def sample_conditionally(model, subjects, weeks, sessions, rows_per_condition):
    """Generate multiple synthetic data rows for each specific condition."""
    logging.info(f"Step 5: Generating conditional synthetic data...")
    
    # Create a DataFrame of all unique combinations of conditions
    condition_tuples = list(itertools.product(subjects, weeks, sessions))
    unique_conditions = pd.DataFrame(condition_tuples, columns=['Subject', 'Week', 'Session'])
    
    # Repeat each condition 'rows_per_condition' times
    conditions = pd.concat([unique_conditions] * rows_per_condition, ignore_index=True)
    
    logging.info(f"Generating {len(conditions)} rows ({rows_per_condition} per condition) for {len(subjects)} new subjects.")

    # Sample from the model using the conditions
    # The model will generate the remaining columns based on the patterns it learned.
    synth = model.sample_from_conditions(conditions=conditions)
    
    logging.info(f"Step 5: Finished generating conditional synthetic data.")
    return synth


# -----------------------------
# 6. EVALUATION METRICS
# -----------------------------
def plot_kde_comparison(real, synth, col, outdir="kde_plots"):
    os.makedirs(outdir, exist_ok=True)
    xr = np.linspace(real.min(), real.max(), 200)
    kr = gaussian_kde(real)(xr)
    ks = gaussian_kde(synth)(xr)
    plt.figure()
    plt.plot(xr, kr, label='real')
    plt.plot(xr, ks, label='synth')
    plt.title(f"KDE: {col}")
    plt.legend()
    plt.savefig(f"{outdir}/kde_{col}.png")
    plt.close()

def evaluate_distributions(df_real, df_synth):
    logging.info("Step 6a: Evaluating distributions with KDE plots...")
    num_cols = df_real.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        plot_kde_comparison(df_real[col], df_synth[col], col)
    logging.info(f"Step 6a: Finished KDE plots. They are saved in 'kde_plots/'.")

def evaluate_correlations(df_real, df_synth):
    logging.info("Step 6b: Evaluating column correlations...")
    num_cols = df_real.select_dtypes(include=[np.number]).columns
    real_corr  = df_real[num_cols].corr()
    synth_corr = df_synth[num_cols].corr()
    diff = np.linalg.norm(real_corr.values - synth_corr.values)
    logging.info(f"Step 6b: Frobenius norm of correlation matrix difference: {diff:.4f}")

# -----------------------------
# 7. POST-PROCESSING
# -----------------------------
def clamp_values(df):
    logging.info("Step 7a: Clamping numerical values to [0, 1] range...")
    # Example: clamp all numeric to [0,1] if that is domain
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].clip(lower=0.0, upper=1.0)
    logging.info("Step 7a: Finished clamping values.")
    return df

def reintroduce_missingness(df_real, df_synth):
    logging.info("Step 7b: Reintroducing missingness into synthetic data...")
    num_cols = df_real.select_dtypes(include=[np.number]).columns
    # Fit logistic model for each col: P(missing|others)
    for col in num_cols:
        y = df_real[col].isna().astype(int)
        # If there are no missing values in the column, skip it.
        if y.sum() == 0:
            logging.info(f"Skipping missingness for column '{col}' (no missing values in original data).")
            continue
        
        # Use only numeric columns for the model features
        X_cols = df_real.select_dtypes(include=[np.number]).columns.drop(col)
        X = df_real[X_cols].fillna(0)

        model = LogisticRegression(max_iter=200, solver='liblinear') # Added solver for robustness
        model.fit(X, y)

        Xs = df_synth[X_cols].fillna(0)
        pmiss = model.predict_proba(Xs)[:,1]
        mask = np.random.binomial(1, pmiss).astype(bool)
        df_synth.loc[mask, col] = np.nan
    logging.info("Step 7b: Finished reintroducing missingness.")
    return df_synth

# -----------------------------
# 8. MAIN PIPELINE
# -----------------------------
def main(args):
    # 1. Load data (with missingness preserved)
    df_original = load_data(args.input_csv)
    
    # 2. Clean a copy for training the model
    df_for_training = clean_data_for_training(df_original)

    # 3. Metadata
    metadata = build_metadata(df_for_training)

    # 4. Hyperparameter tuning (optional)
    if args.tune:
        best_cfg = tune_ctgan(df_for_training, metadata)
    else:
        logging.info("Skipping hyperparameter tuning.")
        # default to reasonable values
        best_cfg = {
            'embedding_dim':           64,
            'generator_dim':           (512, 512),
            'discriminator_dim':       (512, 512),
            'batch_size':              500,
        }
    # 5. Train final
    model = train_final_ctgan(df_for_training, metadata, best_cfg)
    # 6. Sample synthetic
    synth = sample_conditionally(model, NEW_SUBJECTS, WEEKS, SESSIONS)
    # 7. Evaluate
    evaluate_distributions(df_for_training, synth)
    evaluate_correlations(df_for_training, synth)
    # 8. Post-process
    synth = clamp_values(synth)
    # Use the original dataframe to learn the missingness pattern
    synth = reintroduce_missingness(df_original, synth)
    # 9. Save
    logging.info(f"Step 9: Saving synthetic data to '{args.output_csv}'...")
    synth.to_csv(args.output_csv, index=False)
    logging.info(f"Step 9: Synthetic data saved successfully.")
    logging.info("Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using CTGAN.")
    parser.add_argument('--input_csv', type=str, default=INPUT_CSV, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV, help='Path to save the output synthetic CSV.')
    parser.add_argument('--epochs', type=int, default=CTGAN_EPOCHS, help='Number of epochs to train the CTGAN model.')
    parser.add_argument('--num_rows', type=int, default=N_SYNTH_ROWS, help='Number of synthetic rows to generate.')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning.')
    
    args = parser.parse_args()
    main(args)
