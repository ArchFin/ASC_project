# =========================
# TET Data Simulation Utilities
# =========================

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from scipy.ndimage import gaussian_filter1d

# -------------------------
# 1. Parameter Fitting
# -------------------------
def fit_t_params(df, state_column=None, dfs=5):
    """
    Estimate means & covariances for each state (or overall if state_column=None).
    Returns a dict: {state: {'mean':…, 'cov':…, 'df':…}}.
    """
    # This function is useful if you want to fit t-distribution parameters from real data
    params = {}
    if state_column is None:
        sub = df.dropna()
        params[0] = {
            'mean': sub.values.mean(axis=0),
            'cov':  sub.cov().values,
            'df':   dfs
        }
    else:
        for s in df[state_column].unique():
            sub = df[df[state_column] == s].dropna()
            params[s] = {
                'mean': sub.drop(columns=state_column).values.mean(axis=0),
                'cov':  sub.drop(columns=state_column).cov().values,
                'df':   dfs
            }
    return params

# -------------------------
# 2. Markov Chain State Sequence Generator
# -------------------------
def simulate_markov_states(n_samples, transition_matrix, initial_probs=None, state_labels=None):
    """
    Simulate a sequence of states using a Markov chain.
    transition_matrix: 2D numpy array (K x K), rows sum to 1.
    initial_probs: 1D array of initial state probabilities (optional).
    state_labels: list of state labels (optional, default: 0..K-1)
    Returns: array of state labels
    """
    # This function generates a sequence of state labels with Markovian transitions
    K = transition_matrix.shape[0]
    if state_labels is None:
        state_labels = list(range(K))
    if initial_probs is None:
        initial_probs = np.ones(K) / K
    states = np.empty(n_samples, dtype=object)
    states[0] = np.random.choice(state_labels, p=initial_probs)
    for t in range(1, n_samples):
        prev_idx = state_labels.index(states[t-1])
        states[t] = np.random.choice(state_labels, p=transition_matrix[prev_idx])
    return states

# -------------------------
# 3. Main Simulation Function
# -------------------------
def simulate_tet(
    n_samples,
    emit_params,
    weights=None,
    transition_matrix=None,
    initial_probs=None,
    feature_names=None,
    smoothness=0
):
    """
    Simulate TET data from a mixture or Markov sequence of Student-t emissions.
    n_samples: int, number of samples to generate
    emit_params: dict mapping state → {'mean', 'cov', 'df'}
    weights: array-like, mixture weights (if no Markov chain)
    transition_matrix: 2D array, Markov transition matrix (if simulating Markov chain)
    initial_probs: 1D array, initial state probabilities (Markov chain)
    feature_names: list of feature names (optional)
    smoothness: float, standard deviation for Gaussian smoothing (0 = no smoothing)
    Returns:
      data_df: DataFrame of simulated features
      states:  array of chosen state at each time
    """
    # 1. Generate state sequence (Markov or mixture)
    states_list = list(emit_params.keys())
    K = len(states_list)
    if transition_matrix is not None:
        states = simulate_markov_states(n_samples, np.array(transition_matrix), initial_probs, state_labels=states_list)
    else:
        if weights is None:
            weights = np.ones(K) / K
        weights = np.array(weights)
        assert np.isclose(weights.sum(), 1), "Weights must sum to 1"
        states = np.random.choice(states_list, size=n_samples, p=weights)
    # 2. Draw from the corresponding multivariate t for each state
    obs = []
    for s in states:
        p = emit_params[s]
        obs.append(multivariate_t.rvs(loc=p['mean'], shape=p['cov'], df=p['df']))
    obs = np.vstack(obs)
    # 3. Feature names
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(obs.shape[1])]
    # 4. Smooth features if requested (controls temporal smoothness of the curves)
    if smoothness > 0:
        for i in range(obs.shape[1]):
            obs[:, i] = gaussian_filter1d(obs[:, i], sigma=smoothness)
    # 5. Normalize features to [0, 1]
    obs_min = obs.min(axis=0)
    obs_max = obs.max(axis=0)
    obs_norm = (obs - obs_min) / (obs_max - obs_min + 1e-8)
    data_df = pd.DataFrame(obs_norm, columns=feature_names)
    return data_df, states

# -------------------------
# 4. Save Simulated Data
# -------------------------
def save_simulated_tet_data(data_df, states, filename, extra_columns=None, template_columns=None):
    """
    Save simulated TET data to CSV in a format similar to real TET data.
    data_df: DataFrame of simulated features
    states: array of state labels
    filename: output CSV file path
    extra_columns: dict of {colname: value or list} to add to DataFrame
    template_columns: list of column names to match real data (optional)
    """
    # This function adds metadata columns and saves the DataFrame in the correct format
    df = data_df.copy()
    df['Cluster'] = states  # Add state label column (like real data)
    if extra_columns:
        for col, val in extra_columns.items():
            if hasattr(val, '__len__') and not isinstance(val, str):
                df[col] = val
            else:
                df[col] = [val]*len(df)
    # Add any missing columns from template, fill with NaN
    if template_columns is not None:
        for col in template_columns:
            if col not in df.columns:
                df[col] = np.nan
        # Reorder columns to match template
        df = df[template_columns]
    df.to_csv(filename, index=False)
    print(f"Simulated data saved to {filename}")

# -------------------------
# 5. Example Usage & Step-by-Step Guide
# -------------------------
if __name__ == "__main__":
    # STEP 1: (Optional) Load real TET data to fit archetypal state parameters
    # df = pd.read_csv('your_real_data.csv')
    # features = [ ... your feature names ... ]
    # params = fit_t_params(df[features + ['Cluster']], state_column='Cluster', dfs=6)
    #
    # Or, manually specify parameters for full control:
    # Example for 3 states (2 stable, 1 metastable), 14 features:
    params = {
        0: {'mean': [0]*14, 'cov': np.eye(14), 'df': 5},           # Stable state A
        1: {'mean': [2]*14, 'cov': np.eye(14)*1.5, 'df': 5},       # Stable state B
        2: {'mean': [1]*14, 'cov': np.eye(14)*2.0, 'df': 5}        # Metastable transition state
    }
    feature_names = ['MetaAwareness', 'Presence', 'PhysicalEffort','MentalEffort','Boredom','Receptivity','EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience']
    
    # Markov chain with metastable transition state:
    transition_matrix = [
        [0.90, 0.00, 0.10],  # state 0: mostly stays, can go to metastable
        [0.00, 0.90, 0.10],  # state 1: mostly stays, can go to metastable
        [0.15, 0.15, 0.70]   # metastable: likely to go to 0 or 1, rarely stays
    ]
    initial_probs = [1.0, 0.0, 0.0] # Initial state probabilities
    
    # Print and save the original transition matrix as a labeled confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(np.array(transition_matrix), annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=[f'State {i}' for i in range(len(transition_matrix))],
                     yticklabels=[f'State {i}' for i in range(len(transition_matrix))])
    plt.title('Original Transition Matrix (Confusion Matrix)')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Summer_Data/original_transition_matrix_confusion.png')
    plt.close()
    
    # STEP 3: Save simulated data in a real-data-like format
    # To match your real TET data structure, specify all columns as in your CSV:
    template_columns = [
        'Subject','Week','Session','Condition','MetaAwareness','Presence','PhysicalEffort','MentalEffort','Boredom','Receptivity','EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience','Cluster'
    ]
    # Example: Save Markov chain simulation in real-data format for HMM.py
    # Generate multiple subjects, weeks, and runs for realism
    n_subjects = 14
    n_weeks = 4
    n_sessions = 7  # Sessions per week
    timepoints_per_session = 180  # Number of timepoints per session
    subjects = [f"sim{str(i+1).zfill(2)}" for i in range(n_subjects)]
    weeks = [f"week_{i+1}" for i in range(n_weeks)]
    sessions = [f"run_{str(i+1).zfill(2)}" for i in range(n_sessions)]

    all_data = []
    all_states = []
    subject_col = []
    week_col = []
    session_col = []
    for subj in subjects:
        for week in weeks:
            for sess in sessions:
                # For each (Subject, Week, Session), simulate a time series block
                sim_data_block, sim_states_block = simulate_tet(
                    timepoints_per_session, params, transition_matrix=transition_matrix,
                    initial_probs=initial_probs, feature_names=feature_names, smoothness=8)
                all_data.append(sim_data_block)
                all_states.append(sim_states_block)
                subject_col.extend([subj]*timepoints_per_session)
                week_col.extend([week]*timepoints_per_session)
                session_col.extend([sess]*timepoints_per_session)
    sim_data_full = pd.concat(all_data, ignore_index=True)
    sim_states_full = np.concatenate(all_states)
    extra_cols = {
        'Subject': subject_col,
        'Week': week_col,
        'Session': session_col,
        'Condition': 'Simulated',
        # You can add more or override any column here
    }
    total_rows = len(sim_data_full)
    template_columns = [
        'Subject','Week','Session','Condition','MetaAwareness','Presence','PhysicalEffort','MentalEffort','Boredom','Receptivity','EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience','Cluster'
    ]
    save_simulated_tet_data(
        sim_data_full, sim_states_full,
        '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/simulated_TET_for_HMM.csv',
        extra_columns=extra_cols,
        template_columns=template_columns
    )
    
    # -------------
    # For your own use:
    # 1. Edit 'params' to set means/covs/dfs for each state (including metastable)
    # 2. Set feature_names to match your features
    # 3. Choose mixture or Markov mode
    # 4. Call simulate_tet() and use the output
    # 5. Adjust 'smoothness' for more/less realistic curves
    # -------------
