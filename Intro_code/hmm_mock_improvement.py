import numpy as np
import pyhsmm
from pyhsmm.util.text import progprint_xrange
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
# Simulate data with 3 states
true_states = np.random.choice(3, size=1000, p=[0.3, 0.5, 0.2])
data = np.zeros((1000, 1))
for t, state in enumerate(true_states):
    if state == 0:
        data[t] = np.random.normal(0, 1)  # State 0
    elif state == 1:
        data[t] = np.random.normal(5, 1)  # State 1
    elif state == 2:
        data[t] = np.random.normal(10, 1)  # State 2

# Step 2: Define the HDP-HMM
Nmax = 10  # Maximum number of states
obs_dim = data.shape[1]  # Dimension of the observations

# Define observation model (Gaussian emissions)
obs_hypparams = {
    'mu_0': np.zeros(obs_dim), 
    'sigma_0': np.eye(obs_dim),
    'kappa_0': 0.25,
    'nu_0': obs_dim + 2
}
obs_distns = [
    pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)
]

# Define the HDP-HMM
model = pyhsmm.models.HDPHMM(
    alpha=6.0, gamma=6.0, init_state_concentration=1.0,
    obs_distns=obs_distns
)

# Step 3: Add Data to the Model
model.add_data(data)

# Step 4: Run Inference
for iteration in progprint_xrange(100):
    model.resample_model()

# Step 5: Visualize Inferred States
inferred_states = model.stateseqs[0]
plt.figure(figsize=(12, 6))
plt.plot(data, label="Observations", alpha=0.7)
plt.plot(inferred_states, label="Inferred States", alpha=0.9)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("HDP-HMM: Observations and Inferred States")
plt.legend()
plt.show()

# Step 6: Visualize Transition Matrix
plt.figure(figsize=(8, 6))
plt.imshow(model.trans_distn.trans_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Transition Probability')
plt.title("Inferred Transition Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()