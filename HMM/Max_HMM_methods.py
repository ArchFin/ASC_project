# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import savemat
from scipy.stats import zscore
from scipy.linalg import inv, det
from scipy.special import gamma
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Define the HMMModel class for Hidden Markov Model
class HMMModel:
    def __init__(self, num_states, num_emissions, random_seed=12345):
        # Initialize the number of states and emissions
        self.num_states = num_states
        self.num_emissions = num_emissions
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize transition probabilities, emission means, covariances, and degrees of freedom (nu)
        self.trans_prob = np.random.rand(num_states, num_states)
        # Normalize transition probabilities so they sum to 1
        self.trans_prob /= self.trans_prob.sum(axis=1)[:, np.newaxis]
        # Initialize emission means randomly
        self.emission_means = np.random.rand(num_states, num_emissions)
        # Initialize emission covariances as identity matrices
        self.emission_covs = np.stack([np.eye(num_emissions)] * num_states)
        # Initialize degrees of freedom (nu) for the t-distribution
        self.nu = np.full(num_states, 5)
    
    @staticmethod
    def mvtpdf(x, mu, Sigma, nu):
        """
        Compute the multivariate Student's t-distribution PDF.
        """
        d = len(mu)
        x_mu = x - mu
        try:
            # Compute the inverse and determinant of the covariance matrix
            Sigma_inv = inv(Sigma)
            det_Sigma = det(Sigma)
            # Compute the Mahalanobis distance
            mahalanobis_dist = np.dot(x_mu, np.dot(Sigma_inv, x_mu))
            
            # Compute the normalization constant for the t-distribution
            normalization_const = (gamma((d + nu) / 2) /
                                   (gamma(nu / 2) * (np.pi ** (d / 2)) * (det_Sigma ** 0.5) *
                                    (1 + mahalanobis_dist / nu) ** ((d + nu) / 2)))
            return normalization_const
        except np.linalg.LinAlgError:
            # Return 0 if there's a linear algebra error (e.g., singular matrix)
            return 0.0
    
    @staticmethod
    def make_positive_definite(cov_matrix):
        # Ensure the covariance matrix is positive definite
        min_eigenvalue = 1e-6
        cov_matrix = np.array(cov_matrix)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Set small eigenvalues to a minimum value
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        # Reconstruct the covariance matrix
        cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return cov_matrix

    @staticmethod
    def estimate_nu(gamma_vals, data, mean, covariance):
        # Initialize degrees of freedom (nu) with a guess
        nu = 5
        tol = 1e-3
        max_iter = 100
        for _ in range(max_iter):
            old_nu = nu
            # Placeholder for iterative update (e.g., Newton–Raphson)
            if np.abs(nu - old_nu) < tol:
                break
        return nu

    @staticmethod
    def update_transition_probabilities(xi):
        # Update transition probabilities based on the expected counts
        trans_prob = np.sum(xi, axis=2)
        trans_prob = trans_prob / np.sum(trans_prob, axis=1)[:, np.newaxis]
        return trans_prob

    @staticmethod
    def update_emission_parameters(data, gamma, num_states, make_positive_definite, estimate_nu):
        num_data, num_emissions = data.shape
        means = np.zeros((num_states, num_emissions))
        covariances = np.zeros((num_states, num_emissions, num_emissions))
        nu = np.zeros(num_states)
        for j in range(num_states):
            gamma_j = gamma[:, j]
            sum_gamma_j = np.sum(gamma_j)
            # Update emission means
            means[j, :] = np.sum(data * gamma_j[:, np.newaxis], axis=0) / sum_gamma_j
            S = np.zeros((num_emissions, num_emissions))
            for t in range(num_data):
                diff = data[t, :] - means[j, :]
                # Update emission covariances
                S += gamma_j[t] * np.outer(diff, diff)
            covariances[j, :, :] = S / sum_gamma_j
            # Ensure the covariance matrix is positive definite
            covariances[j, :, :] = make_positive_definite(covariances[j, :, :])
            # Update degrees of freedom (nu)
            nu[j] = estimate_nu(gamma_j, data, means[j, :], covariances[j, :, :])
        return means, covariances, nu

    def e_step(self, data):
        num_data, _ = data.shape
        num_states = self.num_states

        # Print covariance matrix shape for debugging
        for j in range(num_states):
            print(f"Covariance matrix shape for state {j}: {self.emission_covs[j, :, :].shape}")

        alpha = np.zeros((num_data, num_states))
        beta = np.zeros((num_data, num_states))
        gamma_vals = np.zeros((num_data, num_states))
        xi = np.zeros((num_states, num_states, num_data - 1))
        
        # Compute alpha (forward probabilities)
        for t in range(num_data):
            for j in range(num_states):
                try:
                    if t == 0:
                        alpha[t, j] = (self.mvtpdf(data[t, :], self.emission_means[j, :],
                                                     self.emission_covs[j, :, :], self.nu[j]) *
                                         (1 / num_states))
                    else:
                        alpha[t, j] = (self.mvtpdf(data[t, :], self.emission_means[j, :],
                                                     self.emission_covs[j, :, :], self.nu[j]) *
                                         np.sum(alpha[t - 1, :] * self.trans_prob[:, j]))
                except Exception as e:
                    print(f"Error computing alpha at t={t}, state={j}: {e}")
            if np.sum(alpha[t, :]) > 0:
                alpha[t, :] /= np.sum(alpha[t, :])
        
        # Compute beta (backward probabilities)
        beta[-1, :] = 1
        for t in range(num_data - 2, -1, -1):
            for i in range(num_states):
                try:
                    beta[t, i] = np.sum(beta[t + 1, :] * self.trans_prob[i, :] *
                                          np.array([self.mvtpdf(data[t + 1, :], self.emission_means[j, :],
                                                                 self.emission_covs[j, :, :], self.nu[j])
                                                    for j in range(num_states)]))
                except Exception as e:
                    print(f"Error computing beta at t={t}, state={i}: {e}")
            if np.sum(beta[t, :]) > 0:
                beta[t, :] /= np.sum(beta[t, :])
        
        # Compute gamma (smoothed state probabilities)
        for t in range(num_data):
            gamma_vals[t, :] = alpha[t, :] * beta[t, :]
            if np.sum(gamma_vals[t, :]) > 0:
                gamma_vals[t, :] /= np.sum(gamma_vals[t, :])
        
        # Compute xi (pairwise state transition probabilities)
        for t in range(num_data - 1):
            for i in range(num_states):
                for j in range(num_states):
                    try:
                        xi[i, j, t] = (alpha[t, i] * self.trans_prob[i, j] *
                                       self.mvtpdf(data[t + 1, :], self.emission_means[j, :],
                                                   self.emission_covs[j, :, :], self.nu[j]) * beta[t + 1, j])
                    except Exception as e:
                        print(f"Error computing xi at t={t}, state_i={i}, state_j={j}: {e}")
            if np.sum(xi[:, :, t]) > 0:
                xi[:, :, t] /= np.sum(xi[:, :, t])
        
        return gamma_vals, xi

    def forward_backward(self, data):
        num_data = data.shape[0]
        num_states = self.num_states
        alpha = np.zeros((num_data, num_states))
        beta = np.zeros((num_data, num_states))
        gamma_vals = np.zeros((num_data, num_states))
        
        # Forward pass
        for t in range(num_data):
            for j in range(num_states):
                covMat = self.emission_covs[j, :, :]
                if t == 0:
                    alpha[t, j] = (self.mvtpdf(data[t, :], self.emission_means[j, :], covMat, self.nu[j])
                                   * (1 / num_states))
                else:
                    alpha[t, j] = (self.mvtpdf(data[t, :], self.emission_means[j, :], covMat, self.nu[j]) *
                                   np.sum(alpha[t-1, :] * self.trans_prob[:, j]))
            if np.sum(alpha[t, :]) > 0:
                alpha[t, :] /= np.sum(alpha[t, :])
        
        beta[-1, :] = 1
        # Backward pass
        for t in range(num_data - 2, -1, -1):
            for i in range(num_states):
                beta[t, i] = np.sum(beta[t+1, :] * self.trans_prob[i, :] *
                                    np.array([self.mvtpdf(data[t+1, :], self.emission_means[j, :],
                                                           self.emission_covs[j, :, :], self.nu[j])
                                              for j in range(num_states)]))
            if np.sum(beta[t, :]) > 0:
                beta[t, :] /= np.sum(beta[t, :])
        
        # Compute gamma
        for t in range(num_data):
            gamma_vals[t, :] = alpha[t, :] * beta[t, :]
            if np.sum(gamma_vals[t, :]) > 0:
                gamma_vals[t, :] /= np.sum(gamma_vals[t, :])
        
        log_lik = np.sum(np.log(np.sum(alpha, axis=1)))
        return alpha, beta, gamma_vals, log_lik

    def decode(self, data):
        num_data = data.shape[0]
        num_states = self.num_states
        delta = np.zeros((num_data, num_states))
        psi = np.zeros((num_data, num_states), dtype=int)
        
        # Initialisation step
        for j in range(num_states):
            covMat = self.emission_covs[j, :, :]
            delta[0, j] = (self.mvtpdf(data[0, :], self.emission_means[j, :], covMat, self.nu[j])
                           * (1 / num_states))
        if np.sum(delta[0, :]) > 0:
            delta[0, :] /= np.sum(delta[0, :])
        
        # Recursion step
        for t in range(1, num_data):
            for j in range(num_states):
                max_val, max_idx = max((delta[t-1, i] * self.trans_prob[i, j], i)
                                         for i in range(num_states))
                covMat = self.emission_covs[j, :, :]
                delta[t, j] = max_val * self.mvtpdf(data[t, :], self.emission_means[j, :], covMat, self.nu[j])
                psi[t, j] = max_idx
            if np.sum(delta[t, :]) > 0:
                delta[t, :] /= np.sum(delta[t, :])
        
        # Backtracking
        state_seq = np.zeros(num_data, dtype=int)
        state_seq[-1] = np.argmax(delta[-1, :])
        for t in range(num_data-2, -1, -1):
            state_seq[t] = psi[t+1, state_seq[t+1]]
        
        log_prob = np.sum(np.log(np.max(delta, axis=1)))
        return state_seq, log_prob

    def train(self, data, num_iterations=10):
        """
        A simple Baum–Welch training loop.
        """
        for iteration in range(num_iterations):
            print(f"Training iteration {iteration + 1} of {num_iterations}")
            gamma_vals, xi = self.e_step(data)
            self.trans_prob = self.update_transition_probabilities(xi)
            self.emission_means, self.emission_covs, self.nu = self.update_emission_parameters(
                data, gamma_vals, self.num_states,
                make_positive_definite=self.make_positive_definite,
                estimate_nu=self.estimate_nu
            )
        return self.trans_prob, self.emission_means, self.emission_covs, self.nu


class DataLoader:
    @staticmethod
    def load_tet_data(csv_file_path, feelings):
        # Load data from a CSV file
        data = pd.read_csv(csv_file_path)
        required_columns = {'Session', 'Week', 'Subject'}
        if not required_columns.issubset(data.columns):
            raise ValueError('CSV file is missing one or more of the required columns: Session, Week, Subject.')
        if not set(feelings).issubset(data.columns):
            raise ValueError('CSV file is missing one or more of the required "feelings" columns.')
        all_tet_data = data[feelings]
        session_ids = data['Session'].tolist()
        weeks = data['Week'].tolist()
        subjects = data['Subject'].tolist()
        unique_session_ids = [f'{s}_Week{w}_Subject{sbj}' for s, w, sbj in zip(session_ids, weeks, subjects)]
        return all_tet_data, session_ids, weeks, subjects, unique_session_ids


class Visualiser:
    @staticmethod
    
    def visualise_clusters_and_transitions(data, clusters, labels, savelocation):
        # Perform PCA to reduce data to 2 dimensions for visualization
        pca = PCA(n_components=2)
        score = pca.fit_transform(data)
        pc1 = score[:, 0]
        pc2 = score[:, 1]
        plt.figure()
        plt.title('Clusters and Transitions (PCA)')
        colors = plt.cm.get_cmap('tab10', np.max(clusters) + 1)
        for i in range(1, np.max(clusters) + 1):
            cluster_idx = clusters == i
            plt.scatter(pc1[cluster_idx], pc2[cluster_idx], s=10, color=colors(i),
                        label=f'Cluster {i}', alpha=0.6)
        for i in range(1, len(labels)):
            if isinstance(labels[i], str) and 'Transition' in labels[i]:
                plt.plot([pc1[i-1], pc1[i]], [pc2[i-1], pc2[i]], 'k--', linewidth=1.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors(i), markersize=6) for i in range(1, np.max(clusters) + 1)]
        plt.legend(handles=handles, labels=[f'Cluster {i}' for i in range(1, np.max(clusters) + 1)])
        plt.savefig(savelocation + f'HMM_clusters_and_transitions')
        plt.show()

    @staticmethod
    def visualise_session(results_table, start_row, end_row, savelocation):
        # Visualize EEG data and state transitions for a specific session
        segment = results_table.iloc[start_row:end_row, :]
        state_colors = plt.cm.get_cmap('tab10', int(segment['State'].max() + 1))
        plt.figure()
        plt.title(f'EEG Data Visualisation from Row {start_row} to {end_row}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude (offset for each dimension)')
        num_dimensions = segment['TETData'].iloc[0].size
        offset = 5
        for i in range(num_dimensions):
            plt.plot(segment['TETData'].apply(lambda x: x[i]) + offset * i)
        max_value = max(segment['TETData'].apply(np.max)) + 2 * num_dimensions
        for i in range(1, int(segment['State'].max()) + 1):
            idx = segment['State'] == i
            if idx.any():
                plt.fill_between(np.where(idx)[0],
                                 0,
                                 max_value,
                                 color=state_colors(i / (segment['State'].max() + 1)),
                                 alpha=0.3)
        for i, trans_window in enumerate(segment['TransitionalWindow']):
            if trans_window:
                trans_window = list(map(int, trans_window.replace('Start: ', '').replace('End: ', '').split(',')))
                x = np.arange(trans_window[0], trans_window[1] + 1)
                if len(x) > 0 and x[-1] <= end_row:
                    plt.plot(x, [max_value] * len(x), 'k-', linewidth=2)
                    

        plt.savefig(savelocation + f'HMM_visualise_sessions')
        plt.show()