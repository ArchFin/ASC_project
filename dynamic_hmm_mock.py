import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

# Step 1: Generate Mock EEG Data
np.random.seed(42)

# Define parameters for 3 hidden states
n_samples = 300  # Total time steps
state_means = [0, 5, -3]  # Mock state-specific means
state_variances = [1, 0.5, 2]  # Mock state-specific variances
true_states = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])  # True states

# Generate observations based on the true states
observations = np.array([
    np.random.normal(state_means[state], state_variances[state]) for state in true_states
])

# Derived features: Add rolling average as a feature
window_size = 5
rolling_avg = np.convolve(observations, np.ones(window_size)/window_size, mode='same')

# Combine raw observations with rolling average into a 2D feature array
features = np.column_stack((observations, rolling_avg))

# Step 2: Dynamic Transition Matrix Simulation
def get_dynamic_transmat(timestep):
    """Simulates a time-varying transition matrix."""
    base_transmat = np.array([
        [0.7, 0.2, 0],
        [0.1, 0.8, 0],
        [0.2, 0.3, 0]
    ])
    # Add periodic variation to the transition probabilities
    variation = 0.1 * np.sin(2 * np.pi * timestep / 50)
    dynamic_transmat = np.clip(base_transmat + variation, 0.01, 0.99)  # Ensure valid probabilities
    
    # Normalize rows to sum to 1
    dynamic_transmat /= dynamic_transmat.sum(axis=1, keepdims=True)
    return dynamic_transmat

# Step 3: Create and Train a Dynamic HMM
n_states = 3
hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)

# Train on the feature data
hmm_model.fit(features)

# Simulate dynamic transitions
dynamic_transmats = [get_dynamic_transmat(t) for t in range(n_samples)]
decoded_states = []
for t, obs in enumerate(features):
    if t > 0:
        hmm_model.transmat_ = dynamic_transmats[t]
    state = hmm_model.predict(obs.reshape(1, -1))
    decoded_states.append(state[0])

decoded_states = np.array(decoded_states)

# Step 4: Visualization
plt.figure(figsize=(12, 8))

# Plot observations and states
plt.subplot(2, 1, 1)
plt.plot(observations, label="Observations (Mock EEG)", color="black", alpha=0.7)
plt.plot(decoded_states, label="Decoded States (HMM)", color="blue", alpha=0.6, linestyle="--")
plt.scatter(range(n_samples), true_states, label="True States", color="red", s=10)
plt.xlabel("Time Steps")
plt.ylabel("EEG Value")
plt.title("Dynamic HMM on Mock EEG Data")
plt.legend()

# Plot transition matrix heatmap at a specific time
plt.subplot(2, 1, 2)
sns.heatmap(dynamic_transmats[n_samples // 2], annot=True, cmap="viridis", fmt=".2f")
plt.title("Dynamic Transition Matrix (Middle of Sequence)")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# Step 5: Model Evaluation
ari_score = adjusted_rand_score(true_states, decoded_states)
print(f"Adjusted Rand Index (Decoded vs True States): {ari_score:.3f}")

print("Final Transition Matrix:\n", hmm_model.transmat_)
print("Means:\n", hmm_model.means_)
print("Covariances:\n", hmm_model.covars_)