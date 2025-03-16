import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

# Load VKM data and adjust the 'clust' column by repeating each value 7 times.
# 3 is 2a (positive stable), 2 is 2b (negative stable), 1 is 3, and 0 is 1.
df_VKM = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Data/VKM_output.csv')

# VKM the clustering labels
df_1 = df_VKM['cluster_label']

# Load HMM data
df_HMM = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted.csv')
df_2 = df_HMM['transition_label']

# mapping = {'2': 3, '1': 1, '2 to 1': 2, '1 to 2': 0}
# df_2 = df_2.replace(mapping)

print(df_1.value_counts())
print(df_2.value_counts())

print(mutual_info_score(df_1, df_2))
print(normalized_mutual_info_score(df_1, df_2))
print(adjusted_mutual_info_score(df_1, df_2))

# 1. Total count of each Transition Type
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=df_VKM, x='cluster_label', order=df_VKM['cluster_label'].value_counts().index)
plt.title("Total Count of Each Transition Type")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(data=df_HMM, x='transition_label', order=df_HMM['transition_label'].value_counts().index)
plt.title("Total Count of Each Transition Type")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/transition_type_count_adjusted.png')

contingency = pd.crosstab(df_VKM['cluster_label'], df_HMM['transition_label'])
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, cmap='viridis', fmt='d')
plt.title('Contingency Matrix of Cluster Labels')
plt.xlabel('HMM Transition Labels')
plt.ylabel('VKM Cluster Labels')
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/contingency_adjusted.png')