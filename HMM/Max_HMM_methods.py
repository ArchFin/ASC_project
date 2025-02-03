from sklearn.cluster import KMeans
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_t
from hmmlearn import hmm


# Load YAML file
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary


class TETProcessor:
    """
    Handles EEG data loading, scaling, and processing.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaled_data = None

    def load_data(self):
        """Loads EEG data from CSV and extracts 'feeling' columns."""
        self.data = pd.read_csv(self.file_path)
        print(f"Data loaded successfully: {self.data.shape}")

    def scale_data(self, feature_columns):
        """Scales feature columns using StandardScaler."""
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data[feature_columns])
        print("Feature scaling complete.")

    def get_scaled_data(self):
        """Returns scaled data for clustering and HMM."""
        return self.scaled_data


class ClusterAnalysis:
    """
    Performs K-Means clustering on EEG data.
    """
    def __init__(self, data, max_clusters=20):
        self.data = data
        self.max_clusters = max_clusters
        self.optimal_k = None
        self.cluster_labels = None

    def find_optimal_clusters(self):
        """
        Uses the Elbow Method to determine the best K.
        """
        inertia_values = []
        for k in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia_values.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.max_clusters + 1), inertia_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.show()

        self.optimal_k = int(input("Optimal number of clusters based on elbow plot: "))
        print(f"Selected clusters: {self.optimal_k}")

    def perform_clustering(self):
        """Runs K-Means and assigns cluster labels."""
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.data)
        print("Clustering complete.")

    def get_cluster_labels(self):
        return self.cluster_labels


class StatisticalFunctions:
    """
    Implements statistical functions like multivariate t-distribution PDF.
    """
    @staticmethod
    def mvtpdf(x, mean, covariance, nu):
        """
        Computes the multivariate Student’s t-distribution probability density.
        """
        d = len(x)
        x_mu = x - mean
        Sigma_inv = np.linalg.inv(covariance)
        mahalanobis_dist = np.dot(x_mu, np.dot(Sigma_inv, x_mu))

        normalization_const = (multivariate_t.pdf(x, mean=mean, shape=covariance, df=nu))
        return normalization_const


class HMMModel:
    """
    Implements an HMM with EM (Expectation-Maximisation).
    """
    def __init__(self, cluster_labels, num_states):
        self.cluster_labels = cluster_labels
        self.num_states = num_states
        self.hmm_model = None
        self.state_sequence = None

    def train_hmm(self):
        """
        Trains a Gaussian HMM using clustered labels.
        """
        reshaped_labels = np.array(self.cluster_labels).reshape(-1, 1)
        self.hmm_model = hmm.GaussianHMM(n_components=self.num_states, covariance_type="full", random_state=42)
        self.hmm_model.fit(reshaped_labels)
        print("HMM training complete.")

    def predict_states(self):
        """
        Predicts state sequences using the Viterbi algorithm.
        """
        reshaped_labels = np.array(self.cluster_labels).reshape(-1, 1)
        self.state_sequence = self.hmm_model.predict(reshaped_labels)
        print("State sequence prediction complete.")

    def e_step(self, data, means, covariances, transProb, nu):
        """
        Performs the Expectation Step of the EM algorithm.
        """
        numData, numStates = len(data), self.num_states
        alpha = np.zeros((numData, numStates))
        beta = np.zeros((numData, numStates))
        gamma = np.zeros((numData, numStates))
        xi = np.zeros((numStates, numStates, numData - 1))

        # Forward Pass (Alpha)
        for t in range(numData):
            for j in range(numStates):
                if t == 0:
                    alpha[t, j] = StatisticalFunctions.mvtpdf(data[t, :], means[j, :], covariances[j, :, :], nu[j]) * (1 / numStates)
                else:
                    alpha[t, j] = StatisticalFunctions.mvtpdf(data[t, :], means[j, :], covariances[j, :, :], nu[j]) * np.sum(alpha[t - 1, :] * transProb[:, j])
            alpha[t, :] /= np.sum(alpha[t, :])

        # Backward Pass (Beta)
        beta[-1, :] = 1
        for t in range(numData - 2, -1, -1):
            for i in range(numStates):
                beta[t, i] = np.sum(beta[t + 1, :] * transProb[i, :] *
                                    np.array([StatisticalFunctions.mvtpdf(data[t + 1, :], means[j, :], covariances[j, :, :], nu[j])
                                              for j in range(numStates)]))
            beta[t, :] /= np.sum(beta[t, :])

        # Compute Gamma
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Compute Xi (Transition Probabilities)
        for t in range(numData - 1):
            for i in range(numStates):
                for j in range(numStates):
                    xi[i, j, t] = alpha[t, i] * transProb[i, j] * StatisticalFunctions.mvtpdf(data[t + 1, :], means[j, :], covariances[j, :, :], nu[j]) * beta[t + 1, j]
            xi[:, :, t] /= np.sum(xi[:, :, t])

        return gamma, xi

    def get_state_sequence(self):
        return self.state_sequence


class TETVisualisation:
    """
    Handles visualisation of clusters and transitions.
    """
    def __init__(self, state_sequence, data):
        self.state_sequence = state_sequence
        self.data = data

    def visualize_clusters_and_transitions(self, clusters):
        """
        Visualises PCA-based clusters and transitions.
        """
        pca = PCA(n_components=2)
        score = pca.fit_transform(self.data)

        pc1, pc2 = score[:, 0], score[:, 1]
        colors = plt.cm.get_cmap('tab10', np.max(clusters) + 1)

        for i in range(1, np.max(clusters) + 1):
            cluster_idx = clusters == i
            plt.scatter(pc1[cluster_idx], pc2[cluster_idx], s=10, color=colors(i), label=f'Cluster {i}', alpha=0.6)

        for i in range(1, len(self.state_sequence)):
            plt.plot([pc1[i - 1], pc1[i]], [pc2[i - 1], pc2[i]], 'k--', linewidth=1.5)

        plt.legend()
        plt.title("PCA-Based Cluster and Transition Visualisation")
        plt.show()


# --- Running the Process ---
file_path = "tet_data.csv"
