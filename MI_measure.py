import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.stats import entropy

# Helper function to compute entropy of label distribution
def compute_entropy(labels):
    # Since our labels are integers, we can use np.bincount
    counts = np.bincount(labels.astype(int))
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)

# --------------------------
# Data Loading and Preprocessing
# --------------------------

# Load VKM data and adjust the 'clust' column by repeating each value 7 times.
# 3 is 2a (positive stable), 2 is 2b (negative stable), 1 is 3, and 0 is 1.
df_VKM = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Data/VKM_output.csv')

# VKM the clustering labels
df_1 = df_VKM['cluster_label']

# Load HMM data
df_HMM = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv')
df_2 = df_HMM['transition_label']


# At high iterations cluster 1 is the negative stable and the cluster 2 is the positive stable so 1 to 2 is positive vectorial and 2 to 1 is the negative vectorial
# k-means the green (1) is the negative stable cluster, the red is the positive vectorial cluster(0), the blue is negative vectorial cluster(2), yellow is positve cluster (3)
# Re-label the HMM data using the provided mapping dictionary
# mapping = {'2': 3, '1': 1, '2 to 1': 2, '1 to 2': 0}
# Triple cluster map
# mapping = {'2': 3, '1': 1, '2 to 1': 2, '1 to 2': 0}
# Quadruple cluster mapping
# df_2 = df_2.replace(mapping)


# Print lengths of each label set for validation
print("Length of df_1:", len(df_1))
print("Length of df_2:", len(df_2))

# ----------------------------------|
# Compute Mutual Information Metrics|
# ----------------------------------|

mi = mutual_info_score(df_1, df_2)
nmi = normalized_mutual_info_score(df_1, df_2)
ami = adjusted_mutual_info_score(df_1, df_2)

print("Mutual Information:", mi)
print("Normalized Mutual Information:", nmi)
print("Adjusted Mutual Information:", ami)

# -------------------------------------------|
# Compute Entropy for Each Label Distribution|
# -------------------------------------------|

# H1 = compute_entropy(df_1.to_numpy())
# H2 = compute_entropy(df_2.to_numpy())

# print("Entropy of VKM (df_1):", H1)
# print("Entropy of HMM (df_2):", H2)

# -----------------|
# Additional Notes:|
# -----------------|
# - MI gives the amount of shared information between the two clusterings.
# - Normalized MI scales the value between 0 and 1.
# - Adjusted MI corrects for chance and is useful if clusters are imbalanced.
# - The entropy values (H1 and H2) indicate the uncertainty or variability in each labeling.
#   Since MI cannot exceed the minimum entropy of the two systems, comparing these values can provide
#   insight into the relative predictability and shared information.


