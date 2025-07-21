# =========================
# TET Data Simulation Utilities
# =========================

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from scipy.ndimage import gaussian_filter1d

class TETSimulator:
    def __init__(self, params=None, feature_names=None, transition_matrix=None, initial_probs=None, smoothness=0):
        self.params = params
        self.feature_names = feature_names
        self.transition_matrix = transition_matrix
        self.initial_probs = initial_probs
        self.smoothness = smoothness

    @staticmethod
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

    def simulate_markov_states(self, n_samples, transition_matrix=None, initial_probs=None, state_labels=None):
        """
        Simulate a sequence of states using a Markov chain.
        transition_matrix: 2D numpy array (K x K), rows sum to 1.
        initial_probs: 1D array of initial state probabilities (optional).
        state_labels: list of state labels (optional, default: 0..K-1)
        Returns: array of state labels
        """
        # This function generates a sequence of state labels with Markovian transitions
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        if initial_probs is None:
            initial_probs = self.initial_probs
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

    def simulate_tet(self, n_samples, emit_params=None, weights=None, transition_matrix=None, initial_probs=None, feature_names=None, smoothness=None):
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
        if emit_params is None:
            emit_params = self.params
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        if initial_probs is None:
            initial_probs = self.initial_probs
        if feature_names is None:
            feature_names = self.feature_names
        if smoothness is None:
            smoothness = self.smoothness
        states_list = list(emit_params.keys())
        K = len(states_list)
        if transition_matrix is not None:
            states = self.simulate_markov_states(n_samples, np.array(transition_matrix), initial_probs, state_labels=states_list)
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

    @staticmethod
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
    feature_names = ['MetaAwareness', 'Presence', 'PhysicalEffort','MentalEffort','Boredom','Receptivity','EmotionalIntensity','Clarity','Release','Bliss','Embodiment','Insightfulness','Anxiety','SpiritualExperience']
    n_features = len(feature_names)

    # Define feature groups for structured, anti-correlated experiences
    # Group 1: Internal/Cognitive experiences
    group1_features = ['MetaAwareness', 'Clarity', 'Insightfulness', 'Receptivity', 'Presence', 'Release', 'SpiritualExperience']
    # Group 2: Somatic/Emotional experiences (anti-correlated with Group 1)
    group2_features = ['PhysicalEffort', 'EmotionalIntensity', 'Bliss', 'Boredom', 'Anxiety', 'MentalEffort']
    # Group 3: Features that are moderately active
    group3_features = ['Embodiment' ]


    # Helper to get indices of features
    get_indices = lambda features: [feature_names.index(f) for f in features]
    g1_idx, g2_idx, g3_idx = get_indices(group1_features), get_indices(group2_features), get_indices(group3_features)

    # --- State Definitions ---
    # State A: "Focused Internal" - High on cognitive, low on somatic
    mean_A = np.full(n_features, 0.5) # Start with a baseline
    mean_A[g1_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8, 0.85] # High internal awareness
    mean_A[g2_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2]  # Low somatic/emotional
    mean_A[g3_idx] = 0.4                   # Moderate presence/release


    # State B: "Somatic Release" - Low on cognitive, high on somatic
    mean_B = np.full(n_features, 0.5) # Start with a baseline
    mean_B[g1_idx] = [0.2, 0.15, 0.2, 0.1, 0.15, 0.2, 0.25] # Low internal awareness
    mean_B[g2_idx] = [0.8, 0.85, 0.9, 0.75, 0.75, 0.8]  # High somatic/emotional
    mean_B[g3_idx] = 0.6                  # Higher presence/release


    # State C: "Metastable/Transition" - In-between state, higher variance
    mean_C = (mean_A + mean_B) / 2 # Average of the two primary states
    mean_C[g3_idx] = 0.5           # Slightly higher background noise

    # Define the parameters dictionary for the simulator
    params = {
        'A': { 'mean': mean_A, 'cov': np.eye(n_features)*0.05, 'df': 5 },
        'B': { 'mean': mean_B, 'cov': np.eye(n_features)*0.05, 'df': 5 },
        'C': { 'mean': mean_C, 'cov': np.eye(n_features)*0.1,  'df': 5 } # Higher covariance for transition
    }
    
    # Markov chain with metastable transition state:
    transition_matrix = [
        [0.90, 0.00, 0.10],  # state A: mostly stays, can go to metastable C
        [0.00, 0.90, 0.10],  # state B: mostly stays, can go to metastable C
        [0.05, 0.05, 0.90]   # state C: likely to go to A or B, rarely stays
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
    sim = TETSimulator(
        params=params,
        feature_names=feature_names,
        transition_matrix=np.array(transition_matrix),
        initial_probs=np.array(initial_probs),
        smoothness=10
    )
    for subj in subjects:
        for week in weeks:
            for sess in sessions:
                # For each (Subject, Week, Session), simulate a time series block
                sim_data_block, sim_states_block = sim.simulate_tet(
                    timepoints_per_session
                )
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
    TETSimulator.save_simulated_tet_data(
        sim_data_full, sim_states_full,
        '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/simulated_TET_for_HMM.csv',
        extra_columns=extra_cols,
        template_columns=template_columns
    )
    
    # === STEP 2.5: Add smooth data configurations to YAML config ===
    # Generate both smooth and unsmooth versions for testing
    # Create enhanced TET data - smooth version with high smoothness value
    sim_smooth = TETSimulator(
        params=params,
        feature_names=feature_names,
        transition_matrix=np.array(transition_matrix),
        initial_probs=np.array(initial_probs),
        smoothness=15  # High smoothness for testing smooth data pipeline
    )

    all_data_smooth = []
    all_states_smooth = []
    subject_col_smooth = []
    week_col_smooth = []
    session_col_smooth = []

    print("Generating SMOOTH version for pipeline testing...")
    for subj in subjects:
        for week in weeks:
            for sess in sessions:
                sim_data_block, sim_states_block = sim_smooth.simulate_tet(timepoints_per_session)
                all_data_smooth.append(sim_data_block)
                all_states_smooth.append(sim_states_block)
                subject_col_smooth.extend([subj]*timepoints_per_session)
                week_col_smooth.extend([week]*timepoints_per_session)
                session_col_smooth.extend([sess]*timepoints_per_session)

    sim_data_full_smooth = pd.concat(all_data_smooth, ignore_index=True)
    sim_states_full_smooth = np.concatenate(all_states_smooth)

    extra_cols_smooth = {
        'Subject': subject_col_smooth,
        'Week': week_col_smooth,
        'Session': session_col_smooth,
        'Condition': 'Simulated_Smooth',
    }

    TETSimulator.save_simulated_tet_data(
        sim_data_full_smooth, sim_states_full_smooth,
        '/Users/a_fin/Desktop/Year 4/Project/Summer_Data/simulated_TET_SMOOTH_for_HMM.csv',
        extra_columns=extra_cols_smooth,
        template_columns=template_columns
    )

    print(f"Smooth data smoothness metric: {np.mean(np.abs(np.diff(sim_data_full_smooth[feature_names].values, axis=0))):.6f}")
    print(f"Original data smoothness metric: {np.mean(np.abs(np.diff(sim_data_full[feature_names].values, axis=0))):.6f}")

    # -------------
    # For your own use:
    # 1. Edit 'params' to set means/covs/dfs for each state (including metastable)
    # 2. Set feature_names to match your features
    # 3. Choose mixture or Markov mode
    # 4. Call simulate_tet() and use the output
    # 5. Adjust 'smoothness' for more/less realistic curves
    # -------------
