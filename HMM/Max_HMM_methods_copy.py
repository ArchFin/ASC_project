import numpy as np
import random
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.linalg import inv, det
from scipy.special import gamma
from matplotlib.lines import Line2D
from scipy.io import savemat
import os

class HMMModel:
    def __init__(self, num_states, num_emissions, random_seed=12345):
        self.num_states = num_states
        self.num_emissions = num_emissions
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Initialise transition probabilities and normalise each row.
        self.trans_prob = np.random.rand(num_states, num_states)
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)

        # Initialise emission parameters.
        self.emission_means = np.random.rand(num_states, num_emissions)
        self.emission_covs = np.stack([np.eye(num_emissions) for _ in range(num_states)])
        self.nu = np.full(num_states, 5)

    @staticmethod
    def mvtpdf(x, mu, Sigma, nu):
        """
        Compute the multivariate Student's t-distribution PDF for a given observation.
        """
        d = len(mu)
        x_mu = x - mu
        Sigma_inv = inv(Sigma)
        det_Sigma = det(Sigma)
        mahalanobis_dist = np.dot(x_mu, np.dot(Sigma_inv, x_mu))
        # Compute the normalisation constant.
        norm_const = gamma((d + nu) / 2) / (gamma(nu / 2) * (np.pi * nu) ** (d / 2) * (det_Sigma ** 0.5))
        pdf = norm_const * (1 + mahalanobis_dist / nu) ** (-(d + nu) / 2)
        return pdf

    @staticmethod
    def make_positive_definite(cov_matrix):
        """
        Ensure that the covariance matrix is positive definite by adjusting small eigenvalues.
        """
        cov_matrix = np.array(cov_matrix)
        min_eigenvalue = 1e-6
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    @staticmethod
    def estimate_nu(gamma_vals, data, mean, covariance):
        """
        Estimate degrees of freedom (nu) for the t-distribution. This is a placeholder 
        for an iterative update (e.g. Newton–Raphson) and returns 5 for now.
        """
        nu = 5
        # Iterative update could be inserted here.
        return nu

    @staticmethod
    def update_transition_probabilities(xi):
        """
        Update transition probabilities using expected counts.
        """
        # Sum over the time axis and normalise.
        trans_prob = np.sum(xi, axis=2)
        trans_prob /= trans_prob.sum(axis=1, keepdims=True)
        return trans_prob

    @staticmethod
    def update_emission_parameters(data, gamma, num_states, make_positive_definite, estimate_nu):
        """
        Update emission means, covariances and nu using weighted statistics.
        """
        num_data, num_emissions = data.shape
        means = np.zeros((num_states, num_emissions))
        covariances = np.zeros((num_states, num_emissions, num_emissions))
        nu = np.zeros(num_states)

        for j in range(num_states):
            gamma_j = gamma[:, j]
            sum_gamma = np.sum(gamma_j)
            # Update means using a weighted average.
            means[j, :] = np.sum(data * gamma_j[:, np.newaxis], axis=0) / sum_gamma
            # Vectorised covariance update:
            diff = data - means[j, :]
            covariances[j, :, :] = (diff.T * gamma_j) @ diff / sum_gamma
            covariances[j, :, :] = make_positive_definite(covariances[j, :, :])
            nu[j] = estimate_nu(gamma_j, data, means[j, :], covariances[j, :, :])
        return means, covariances, nu

    def _compute_emission_probs(self, data):
        """
        Precompute the emission probability for each observation and state.
        Returns an array of shape (num_data, num_states).
        """
        num_data = data.shape[0]
        emission_probs = np.zeros((num_data, self.num_states))
        for j in range(self.num_states):
            # Compute the probability of each observation under state j.
            pdf_func = lambda x: self.mvtpdf(x, self.emission_means[j], self.emission_covs[j], self.nu[j])
            emission_probs[:, j] = np.apply_along_axis(pdf_func, 1, data)
        return emission_probs

    def e_step(self, data):
        """
        The expectation (E‐step) computes the forward and backward probabilities and then the 
        smoothed state probabilities (gamma) and the pairwise state transition probabilities (xi).
        """
        num_data = data.shape[0]
        num_states = self.num_states

        emission_probs = self._compute_emission_probs(data)
        alpha = np.zeros((num_data, num_states))
        beta = np.zeros((num_data, num_states))
        gamma_vals = np.zeros((num_data, num_states))
        xi = np.zeros((num_states, num_states, num_data - 1))

        # Forward pass (vectorised)
        alpha[0, :] = emission_probs[0, :] * (1 / num_states)
        alpha[0, :] /= np.sum(alpha[0, :])
        for t in range(1, num_data):
            # Compute alpha[t] = emission * (previous alpha * transition)
            alpha[t, :] = emission_probs[t, :] * (alpha[t - 1, :] @ self.trans_prob)
            if np.sum(alpha[t, :]) > 0:
                alpha[t, :] /= np.sum(alpha[t, :])

        # Backward pass (vectorised)
        beta[-1, :] = 1
        for t in range(num_data - 2, -1, -1):
            beta[t, :] = self.trans_prob @ (beta[t + 1, :] * emission_probs[t + 1, :])
            if np.sum(beta[t, :]) > 0:
                beta[t, :] /= np.sum(beta[t, :])

        # Compute gamma: element-wise product of alpha and beta, normalised.
        gamma_vals = alpha * beta
        gamma_vals /= gamma_vals.sum(axis=1, keepdims=True)

        # Compute xi: pairwise state probabilities.
        for t in range(num_data - 1):
            denominator = np.dot(alpha[t, :], self.trans_prob * (emission_probs[t + 1, :] * beta[t + 1, :]).reshape(1, -1)).sum()
            if denominator == 0:
                continue
            xi[:, :, t] = (alpha[t, :].reshape(-1, 1) * self.trans_prob *
                           (emission_probs[t + 1, :] * beta[t + 1, :])) / denominator

        return gamma_vals, xi

    def forward_backward(self, data):
        """
        Perform the forward–backward algorithm, returning alpha, beta, the smoothed probabilities (gamma)
        and the log likelihood.
        """
        num_data = data.shape[0]
        num_states = self.num_states
        emission_probs = self._compute_emission_probs(data)

        alpha = np.zeros((num_data, num_states))
        beta = np.zeros((num_data, num_states))
        gamma_vals = np.zeros((num_data, num_states))

        # Forward pass.
        alpha[0, :] = emission_probs[0, :] * (1 / num_states)
        alpha[0, :] /= np.sum(alpha[0, :])
        for t in range(1, num_data):
            alpha[t, :] = emission_probs[t, :] * (alpha[t - 1, :] @ self.trans_prob)
            if np.sum(alpha[t, :]) > 0:
                alpha[t, :] /= np.sum(alpha[t, :])

        # Backward pass.
        beta[-1, :] = 1
        for t in range(num_data - 2, -1, -1):
            beta[t, :] = self.trans_prob @ (beta[t + 1, :] * emission_probs[t + 1, :])
            if np.sum(beta[t, :]) > 0:
                beta[t, :] /= np.sum(beta[t, :])

        # Compute gamma and log likelihood.
        gamma_vals = alpha * beta
        gamma_vals /= gamma_vals.sum(axis=1, keepdims=True)
        log_lik = np.sum(np.log(alpha.sum(axis=1) + 1e-12))
        return alpha, beta, gamma_vals, log_lik

    def decode(self, data):
        """
        Viterbi decoding to obtain the most likely state sequence and log probability.
        """
        num_data = data.shape[0]
        num_states = self.num_states
        emission_probs = self._compute_emission_probs(data)
        delta = np.zeros((num_data, num_states))
        psi = np.zeros((num_data, num_states), dtype=int)

        # Initialisation.
        delta[0, :] = emission_probs[0, :] * (1 / num_states)
        delta[0, :] /= np.sum(delta[0, :])

        # Recursion.
        for t in range(1, num_data):
            for j in range(num_states):
                # For each state j, compute the maximum probability and record the best previous state.
                prev_vals = delta[t - 1, :] * self.trans_prob[:, j]
                max_idx = np.argmax(prev_vals)
                delta[t, j] = prev_vals[max_idx] * emission_probs[t, j]
                psi[t, j] = max_idx
            if np.sum(delta[t, :]) > 0:
                delta[t, :] /= np.sum(delta[t, :])

        # Backtracking.
        state_seq = np.zeros(num_data, dtype=int)
        state_seq[-1] = np.argmax(delta[-1, :])
        for t in range(num_data - 2, -1, -1):
            state_seq[t] = psi[t + 1, state_seq[t + 1]]
        log_prob = np.sum(np.log(np.max(delta, axis=1) + 1e-12))
        return state_seq, log_prob

    def train(self, data, num_iterations=10):
        """
        Baum–Welch training loop. At each iteration the E‐step is computed and then the model parameters are updated.
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

# DataLoader and Visualiser classes remain largely the same.
# They are kept for organisational purposes and visualisation of the results.

class DataLoader:
    @staticmethod
    def load_tet_data(csv_file_path, feelings):
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
        pca = PCA(n_components=2)
        score = pca.fit_transform(data)
        pc1, pc2 = score[:, 0], score[:, 1]
        plt.figure()
        plt.title('Clusters and Transitions (PCA)')
        colors = plt.cm.get_cmap('tab10', np.max(clusters) + 1)
        for i in range(1, np.max(clusters) + 1):
            cluster_idx = clusters == i
            plt.scatter(pc1[cluster_idx], pc2[cluster_idx], s=10, color=colors(i),
                        label=f'Cluster {i}', alpha=0.6)
        for i in range(1, len(labels)):
            if isinstance(labels[i], str) and 'Transition' in labels[i]:
                plt.plot([pc1[i - 1], pc1[i]], [pc2[i - 1], pc2[i]], 'k--', linewidth=1.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors(i), markersize=6) for i in range(1, np.max(clusters) + 1)]
        plt.legend(handles=handles, labels=[f'Cluster {i}' for i in range(1, np.max(clusters) + 1)])
        plt.savefig(os.path.join(savelocation, 'HMM_clusters_and_transitions'))
        plt.show()

    @staticmethod
    def visualise_session(results_table, start_row, end_row, savelocation):
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
                tw = list(map(int, trans_window.replace('Start: ', '').replace('End: ', '').split(',')))
                x = np.arange(tw[0], tw[1] + 1)
                if len(x) > 0 and x[-1] <= end_row:
                    plt.plot(x, [max_value] * len(x), 'k-', linewidth=2)
        plt.savefig(os.path.join(savelocation, 'HMM_visualise_sessions'))
        plt.show()