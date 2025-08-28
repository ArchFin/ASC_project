import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('/Users/a_fin/Desktop/Year 4/Project/ASC_project/converted_csv/combined_all_subjects_labelled.csv')

feeling_columns = [
    'Aperture','Boredom','Clarity','Conflict','Dereification','Wakefulness','Emotion','Effort','Stability','MetaAwareness','ObjectOrientation','Source'
]
X = data[feeling_columns].values

# Range of k values to try
k_values = range(1, 31)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('K-Means Elbow Method')
plt.xticks(k_values)
plt.grid(True)
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/ASC_project/elbow_plot.png')
plt.close()