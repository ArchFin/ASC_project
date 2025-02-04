import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import savemat
from scipy.stats import zscore, multivariate_t
from scipy.linalg import inv, det
from scipy.special import gamma
from sklearn.decomposition import PCA
import seaborn as sns
import networkx as nx

class HMMModel:
    def __init__(self, num_states, num_emissions, random_seed=12345):
        self.num_states = num_states
        self.num_emissions = num_emissions
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize parameters with proper shapes
        self.trans_prob = np.random.rand(num_states, num_states)
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)
        self.emission_means = np.random.rand(num_states, num_emissions)
        self.emission_covs = np.stack([np.eye(num_emissions) for _ in range(num_states)])
        self.nu = np.full(num_states, 5.0)

    def make_positive_definite(self, cov_matrix):
        min_eigenvalue = 1e-6
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def e_step(self, data):
        num_data = data.shape[0]
        B = np.zeros((self.num_states, num_data))
        
        # Precompute emission probabilities
        for j in range(self.num_states):
            try:
                mvt = multivariate_t(
                    loc=self.emission_means[j],
                    shape=self.emission_covs[j],
                    df=self.nu[j]
                )
                B[j] = mvt.pdf(data)
            except np.linalg.LinAlgError:
                B[j] = 1e-6  # Handle singular matrices

        # Forward pass
        alpha = np.zeros((num_data, self.num_states))
        alpha[0] = B[:, 0] * (1/self.num_states)
        alpha[0] /= alpha[0].sum()
        
        for t in range(1, num_data):
            alpha[t] = B[:, t] * (alpha[t-1] @ self.trans_prob.T)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Backward pass
        beta = np.zeros_like(alpha)
        beta[-1] = 1.0
        for t in range(num_data-2, -1, -1):
            beta[t] = self.trans_prob @ (B[:, t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-10

        # Compute posterior probabilities
        gamma_vals = alpha * beta
        gamma_vals /= gamma_vals.sum(axis=1, keepdims=True) + 1e-10

        # Compute xi
        xi = np.zeros((self.num_states, self.num_states, num_data-1))
        for t in range(num_data-1):
            xi[:, :, t] = alpha[t, :, None] * self.trans_prob * B[None, :, t+1] * beta[t+1, None, :]
            xi[:, :, t] /= xi[:, :, t].sum() + 1e-10

        log_lik = np.log(alpha[-1].sum())
        return gamma_vals, xi, log_lik

    def m_step(self, data, gamma_vals, xi):
        # Update transition probabilities
        self.trans_prob = np.sum(xi, axis=2) / np.sum(gamma_vals[:-1], axis=0)[:, None]

        # Update emission parameters
        for j in range(self.num_states):
            gamma_j = gamma_vals[:, j]
            total = gamma_j.sum()
            
            # Update means
            self.emission_means[j] = (data.T @ gamma_j) / total
            
            # Update covariances
            diff = data - self.emission_means[j]
            self.emission_covs[j] = (diff.T * gamma_j) @ diff / total
            self.emission_covs[j] = self.make_positive_definite(self.emission_covs[j])

        return self.trans_prob, self.emission_means, self.emission_covs

    def decode(self, data):
        num_data = data.shape[0]
        delta = np.zeros((num_data, self.num_states))
        psi = np.zeros((num_data, self.num_states), dtype=int)
        
        # Initialize B matrix
        B = np.zeros((self.num_states, num_data))
        for j in range(self.num_states):
            mvt = multivariate_t(
                loc=self.emission_means[j],
                shape=self.emission_covs[j],
                df=self.nu[j]
            )
            B[j] = mvt.pdf(data)

        # Initialization
        delta[0] = B[:, 0] * (1/self.num_states)
        delta[0] /= delta[0].sum()
        
        # Recursion
        for t in range(1, num_data):
            for j in range(self.num_states):
                trans_probs = delta[t-1] * self.trans_prob[:, j]
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] * B[j, t]
            delta[t] /= delta[t].sum() + 1e-10

        # Backtracking
        state_seq = np.zeros(num_data, dtype=int)
        state_seq[-1] = np.argmax(delta[-1])
        for t in range(num_data-2, -1, -1):
            state_seq[t] = psi[t+1, state_seq[t+1]]

        return state_seq, np.sum(np.log(np.max(delta, axis=1)))

    def train(self, data, max_iter=10, tol=1e-4):
        prev_log_lik = -np.inf
        for _ in range(max_iter):
            gamma_vals, xi, log_lik = self.e_step(data)
            self.m_step(data, gamma_vals, xi)
            
            if np.abs(log_lik - prev_log_lik) < tol:
                break
            prev_log_lik = log_lik
        return self.trans_prob, self.emission_means, self.emission_covs

class DataLoader:
    @staticmethod
    def load_tet_data(csv_file, feelings):
        df = pd.read_csv(csv_file)
        required = ['Session', 'Week', 'Subject']
        if not all(col in df.columns for col in required):
            raise ValueError("Missing required columns")
        
        tet_data = df[feelings].values
        session_ids = df['Session'].values
        weeks = df['Week'].values
        subjects = df['Subject'].values
        unique_ids = [f"{s}_Week{w}_Subject{sbj}" 
                     for s, w, sbj in zip(session_ids, weeks, subjects)]
        
        return tet_data, session_ids, weeks, subjects, unique_ids

class Visualiser:
    def __init__(self, data, state_seq, trans_prob, save_path, feelings):
        self.data = data
        self.state_seq = state_seq
        self.trans_prob = trans_prob
        self.save_path = save_path
        self.feelings = feelings
        self.color_map = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'yellow',
        }

    def plot_trajectories(self):
        plt.figure(figsize=(12, 6))
        for i in range(self.data.shape[1]):
            plt.plot(self.data[:, i] + 3 * i, alpha=0.7, label=f'Feature {i}')
        
        transition_points = np.where(np.diff(self.state_seq) != 0)[0]
        for t in transition_points:
            plt.axvline(t, color='k', linestyle='--', alpha=0.5)
            plt.text(t, plt.ylim()[1] * 0.9, f'State {self.state_seq[t]} -> {self.state_seq[t + 1]}',
                     rotation=90, verticalalignment='top')
        
        plt.xlabel('Time')
        plt.ylabel('Feature Value (Offset for Clarity)')
        plt.title('HMM State Transitions Over Time')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.savefig(os.path.join(self.save_path, 'hmm_state_transitions.png'))
        plt.close()
        
    @staticmethod
    def visualise_clusters(data, states, save_path):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=states, 
                            cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='State')
        plt.xlabel('Principal Component 1')  # Add x-axis label
        plt.ylabel('Principal Component 2')  # Add y-axis label
        plt.title('State Clusters (PCA)')  # Add title
        plt.savefig(os.path.join(save_path, 'state_clusters.png'))
        plt.close()

    @staticmethod
    def visualise_transitions(data, states, save_path):
        plt.figure(figsize=(12, 6))
        for i in range(data.shape[1]):
            plt.plot(data[:, i] + 3*i, alpha=0.7)
        
        for t in np.where(np.diff(states) != 0)[0]:
            plt.axvline(t, color='k', linestyle='--', alpha=0.5)
            
        plt.xlabel('Time')  # Add x-axis label
        plt.ylabel('Feature Value (Offset for Clarity)')  # Add y-axis label
        plt.title('State Transitions Over Time')  # Add title
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Add legend
        plt.savefig(os.path.join(save_path, 'state_transitions.png'))
        plt.close()

    @staticmethod
    def visualise_transition_matrix(trans_prob, save_path):
        plt.figure(figsize=(8, 6))
        sns.heatmap(trans_prob, annot=True, cmap='Blues', fmt='.2f', 
                    xticklabels=[f'State {i}' for i in range(trans_prob.shape[0])],
                    yticklabels=[f'State {i}' for i in range(trans_prob.shape[1])])
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.title('Transition Probability Matrix')
        plt.savefig(os.path.join(save_path, 'transition_matrix.png'))
        plt.close()

    @staticmethod
    def visualise_state_transition_diagram(trans_prob, save_path):
        G = nx.DiGraph()
        for i in range(trans_prob.shape[0]):
            for j in range(trans_prob.shape[1]):
                if trans_prob[i, j] > 0:  # Only draw edges with non-zero probability
                    G.add_edge(f'State {i}', f'State {j}', weight=trans_prob[i, j])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
                font_size=15, font_weight='bold', edge_color='gray', width=2, 
                arrows=True, arrowstyle='->', arrowsize=20)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
        plt.title('State Transition Diagram')
        plt.savefig(os.path.join(save_path, 'state_transition_diagram.png'))
        plt.close()


    @staticmethod
    def visualise_transitions(data, states, save_path):
        plt.figure(figsize=(12, 6))
        for i in range(data.shape[1]):
            plt.plot(data[:, i] + 3*i, alpha=0.7, label=f'Feature {i}')
        
        transition_points = np.where(np.diff(states) != 0)[0]
        for t in transition_points:
            plt.axvline(t, color='k', linestyle='--', alpha=0.5)
            plt.text(t, plt.ylim()[1] * 0.9, f'State {states[t]} -> {states[t+1]}', 
                     rotation=90, verticalalignment='top')
        
        plt.xlabel('Time')
        plt.ylabel('Feature Value (Offset for Clarity)')
        plt.title('State Transitions Over Time')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.savefig(os.path.join(save_path, 'state_transitions.png'))
        plt.close()

