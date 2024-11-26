# Importing the necessary libraries
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Generating synthetic EEG-like data with state changes
# Define the number of samples per state and features
n_samples_per_state = 250
n_features = 3

# Create three distinct states with different means and variances
state_1 = np.random.normal(loc=1, scale=0.5, size=(n_samples_per_state, n_features))
state_2 = np.random.normal(loc=5, scale=0.5, size=(n_samples_per_state, n_features))
state_3 = np.random.normal(loc=10, scale=0.5, size=(n_samples_per_state, n_features))
state_4 = np.random.normal(loc=3, scale=0.5, size=(n_samples_per_state, n_features))

# Concatenate the states to form a complete dataset
data = np.vstack([state_1, state_2, state_3, state_4])

# Step 2: Preprocessing the data (e.g., normalization)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Step 3: Defining the HMM model
n_states = 4  # Number of hidden states (customize based on the application)
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)

# Step 4: Fitting the HMM to the data
model.fit(data_normalized)

# Step 5: Predicting the hidden states
hidden_states = model.predict(data_normalized)

# Step 6: Extracting the transition matrix, means, and covariances
transition_matrix = model.transmat_
means = model.means_
covariances = model.covars_

# Step 7: Plotting the hidden states over time
plt.figure(figsize=(15, 5))
plt.plot(hidden_states, label='Hidden States', lw=2)
plt.xlabel("Time Points")
plt.ylabel("Hidden State")
plt.title("Inferred Hidden States Over Time")
plt.legend()
plt.show()

import seaborn as sns

# Visualize the transition matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Transition Matrix Heatmap")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# Assume 'means' is a 2D array with shape (n_states, n_features)
n_features = means.shape[1]
plt.figure(figsize=(10, 6))

for i in range(n_features):
    plt.bar(range(n_states), means[:, i], alpha=0.7, label=f"Feature {i+1}")

plt.title("Means of Each Hidden State")
plt.xlabel("Hidden State")
plt.ylabel("Feature Value")
plt.legend()
plt.show()

# Visualize the covariance matrices
for i, cov_matrix in enumerate(covariances):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cov_matrix, annot=True, cmap="Reds", fmt=".2f")
    plt.title(f"Covariance Matrix for Hidden State {i+1}")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.show()


