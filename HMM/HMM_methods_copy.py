import os
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from scipy.spatial import distance
from scipy.spatial.distance import cdist, euclidean
from scipy.linalg import inv, det
from scipy.special import psi, polygamma, gamma 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import networkx as nx
from statsmodels.tsa.stattools import acf
from hmmlearn import hmm  # Only if you still need it somewhere
from tqdm import tqdm  # For progress bars

# =============================================================================
# Configuration via YAML
# =============================================================================
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

# =============================================================================
# CSV File Handling
# =============================================================================
class csv_splitter:
    def __init__(self, file_path):
        """
        Constructor to initialise the CSV file location.
        """
        self.file_path = file_path

    def read_CSV(self):
        """
        Reads the CSV file and returns it as a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.file_path)
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def split_by_header(self, df, heading):
        """
        Splits the DataFrame into multiple DataFrames based on unique values in the specified column.
        Returns a dictionary and a list of key-value pairs.
        """
        if heading not in df.columns:
            print(f"Error: '{heading}' not found in DataFrame columns.")
            return None, None

        split_df = {value: df[df[heading] == value] for value in df[heading].unique()}
        split_df_array = [[key, value] for key, value in split_df.items()]
        return split_df, split_df_array

# =============================================================================
# Principal Component Finder
# =============================================================================
class principal_component_finder:
    def __init__(self, csv_file, feelings, no_dimensions, savelocation_TET):
        """
        Extracts the required features (feelings) and performs PCA.
        """
        self.csv_file_TET = csv_file[feelings]
        self.feelings = feelings
        self.savelocation = savelocation_TET
        self.no_dimensions = no_dimensions
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.csv_file_TET)

        # Compute PCA
        pca = PCA(n_components=no_dimensions)
        self.principal_components = pca.fit(scaled_data).components_
        self.explained_variance_ratio = pca.explained_variance_ratio_

    def PCA_TOT(self):
        """
        Projects the data onto the principal components and plots bar charts for each component.
        Returns the principal components, explained variance ratio, and the transformed data.
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.csv_file_TET)  # <-- Standardized data

        # Compute PCA
        pca = PCA(n_components=self.no_dimensions)  # Added self. for variable reference
        self.principal_components = pca.fit(scaled_data).components_
        self.explained_variance_ratio = pca.explained_variance_ratio_

        # Project STANDARDIZED DATA 
        df_TET_feelings_prin = pd.DataFrame(scaled_data @ self.principal_components.T,  # <-- KEY FIX
                                            columns=[f"PC{i+1}" for i in range(self.no_dimensions)])

        # Plot bar charts for each principal component.
        for i in range(self.principal_components.shape[0]):
            plt.figure()
            plt.bar(self.feelings, self.principal_components[i])
            plt.title(f'Principal Component {i+1}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.savelocation, f'principal_component{i+1}.png'))
            plt.close()

        # Scatter plot of the first two principal components.
        plt.figure()
        plt.scatter(df_TET_feelings_prin.iloc[:, 0], df_TET_feelings_prin.iloc[:, 1], s=0.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Projection')
        # plt.xlim(-6, 6)
        # plt.ylim(-1, 2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation, 'all_data_in_PCA.png'))
        plt.close()

        # Bar chart for explained variance ratio.
        labels = [f'Principal Component {i+1}' for i in range(len(self.explained_variance_ratio))]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, self.explained_variance_ratio, color='skyblue')
        plt.title('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Explained')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation, 'explained_variance_ratio.png'))
        plt.close()

        return self.principal_components, self.explained_variance_ratio, df_TET_feelings_prin


    def PCA_split(self, split_df_array):
        """
        Projects each split dataset onto the principal components and plots scatter plots.
        Returns a dictionary mapping split names to their PCA-transformed data.
        """
        split_csv_TET = {item[0]: item[1][self.feelings] for item in split_df_array}
        df_TET_feelings_prin_dict = {name: df @ self.principal_components.T for name, df in split_csv_TET.items()}

        for key, value in df_TET_feelings_prin_dict.items():
            plt.figure()
            plt.scatter(value.iloc[:, 0], value.iloc[:, 1], s=0.5)
            plt.title(f'PCA Projection: {key}')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            # plt.xlim(-6, 6)
            # plt.ylim(-1, 2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.savelocation, f'PCA_{key}.png'))
            plt.close()

        return df_TET_feelings_prin_dict
# =============================================================================
# Custom HMM Model using multivariate t–distribution 
# =============================================================================
class HMMModel:
    def __init__(self, num_states, num_emissions, data=None, random_seed=12345):
        self.num_states = num_states
        self.num_emissions = num_emissions
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Improved initialization using K-means
        if data is not None:
            kmeans = KMeans(n_clusters=num_states, random_state=random_seed).fit(data)
            self.emission_means = kmeans.cluster_centers_
            # Diagonal-dominated initialization (favor state persistence):
            self.trans_prob = np.eye(num_states) * 0.8 + np.ones((num_states, num_states)) * 0.2/(num_states-1)
            np.fill_diagonal(self.trans_prob, 0.8)  # 80% self-transition probability
            self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)  # Ensure row normalization
        else:
            self.emission_means = np.random.randn(num_states, num_emissions)
            self.trans_prob = np.random.dirichlet(np.ones(num_states), size=num_states)

        # Regularization for covariance matrices
        self.emission_covs = np.stack([np.eye(num_emissions)+1e-4*np.random.randn(num_emissions,num_emissions) 
                                      for _ in range(num_states)])
        self.nu = np.clip(np.random.gamma(5, 1, num_states), 2, 10)

    @staticmethod
    def mvtpdf(x, mu, Sigma, nu):
        d = len(mu)
        x_mu = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        mahalanobis = x_mu @ Sigma_inv @ x_mu

        # Compute log-pdf to avoid underflow
        log_norm = (np.log(gamma((nu + d)/2)) 
                    - np.log(gamma(nu/2)) 
                    - (d/2) * np.log(nu * np.pi) 
                    - 0.5 * np.log(det_Sigma) 
                    - ((nu + d)/2) * np.log(1 + mahalanobis/nu))
        return np.exp(log_norm)

    @staticmethod
    def make_positive_definite(cov_matrix):
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # Force symmetry
        d = cov_matrix.shape[0]
        cov_matrix += 1e-6 * np.eye(d)  # Add regularization
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.clip(eigenvalues, 1e-6, None)  # Clip eigenvalues
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    @staticmethod
    def estimate_nu(gamma_vals, data, mean, covariance, max_iter=100):
        N, d = data.shape
        diff = data - mean
        inv_cov = np.linalg.pinv(covariance)  # More stable pseudo-inverse
        
        # Compute squared Mahalanobis distances
        r = np.einsum('ij,ij->i', diff @ inv_cov, diff)
        
        # Initialize nu using moment matching
        nu = np.clip(2/(np.mean(r/(d + 2)) - 1), 2, 10)
        
        for _ in range(max_iter):
            w = (nu + d)/(nu + r)
            numerator = np.sum(gamma_vals * (np.log(w) - w))
            denominator = np.sum(gamma_vals * (psi((nu + d)/2) - np.log((nu + d)/2)))
            
            if denominator == 0:
                break
                
            nu_new = nu - numerator/denominator
            if abs(nu_new - nu) < 1e-4:
                break
            nu = np.clip(nu_new, 2, 10)
            
        return nu
    
    @staticmethod
    def update_transition_probabilities(xi, num_states):
        # regularized version:
        state_persistence_prior = 5.0  # Higher values = stronger self-transition preference
        trans_prob = np.sum(xi, axis=2) + np.eye(num_states) * state_persistence_prior
        trans_prob += 1e-12  # Numerical stability
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
            means[j, :] = np.sum(data * gamma_j[:, np.newaxis], axis=0) / sum_gamma
            diff = data - means[j, :]
            covariances[j, :, :] = (diff.T * gamma_j) @ diff / sum_gamma
            covariances[j, :, :] = make_positive_definite(covariances[j, :, :])
            # covariances[j, :, :] = (diff.T * gamma_j) @ diff / sum_gamma
            # Add regularization and mean separation:
            covariances[j, :, :] += np.eye(num_emissions) * 1e-3  # Covariance regularization
            if j > 0:  # Push cluster means apart
                means[j, :] += 0 * (means[j, :] - np.mean(means[:j, :], axis=0))
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
            pdf_func = lambda x: self.mvtpdf(x, self.emission_means[j], self.emission_covs[j], self.nu[j])
            emission_probs[:, j] = np.apply_along_axis(pdf_func, 1, data)
        return emission_probs

    def e_step(self, data):
        """
        Perform the expectation (E‐step) by computing forward (alpha), backward (beta),
        smoothed probabilities (gamma) and pairwise state probabilities (xi).
        """
        num_data = data.shape[0]
        num_states = self.num_states

        emission_probs = self._compute_emission_probs(data)
        alpha = np.zeros((num_data, num_states))
        beta = np.zeros((num_data, num_states))
        gamma_vals = np.zeros((num_data, num_states))
        xi = np.zeros((num_states, num_states, num_data - 1))

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

        gamma_vals = alpha * beta
        gamma_vals /= gamma_vals.sum(axis=1, keepdims=True)

        for t in range(num_data - 1):
            denominator = np.sum(alpha[t, :] @ self.trans_prob * emission_probs[t + 1, :] @ beta[t + 1, :])
            if denominator == 0:
                continue
            xi[:, :, t] = (alpha[t, :].reshape(-1, 1) * self.trans_prob *
                           (emission_probs[t + 1, :] * beta[t + 1, :])) / denominator

        return gamma_vals, xi

    def forward_backward(self, data):
        """
        Execute the forward–backward algorithm and return alpha, beta, gamma and the log likelihood.
        """
        num_data = data.shape[0]
        num_states = self.num_states
        emission_probs = self._compute_emission_probs(data)

        alpha = np.zeros((num_data, num_states))
        beta = np.zeros((num_data, num_states))
        gamma_vals = np.zeros((num_data, num_states))

        alpha[0, :] = emission_probs[0, :] * (1 / num_states)
        alpha[0, :] /= np.sum(alpha[0, :])
        for t in range(1, num_data):
            alpha[t, :] = emission_probs[t, :] * (alpha[t - 1, :] @ self.trans_prob)
            if np.sum(alpha[t, :]) > 0:
                alpha[t, :] /= np.sum(alpha[t, :])

        beta[-1, :] = 1
        for t in range(num_data - 2, -1, -1):
            beta[t, :] = self.trans_prob @ (beta[t + 1, :] * emission_probs[t + 1, :])
            if np.sum(beta[t, :]) > 0:
                beta[t, :] /= np.sum(beta[t, :])

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

        delta[0, :] = emission_probs[0, :] * (1 / num_states)
        delta[0, :] /= np.sum(delta[0, :])

        for t in range(1, num_data):
            for j in range(num_states):
                prev_vals = delta[t - 1, :] * self.trans_prob[:, j]
                max_idx = np.argmax(prev_vals)
                delta[t, j] = prev_vals[max_idx] * emission_probs[t, j]
                psi[t, j] = max_idx
            if np.sum(delta[t, :]) > 0:
                delta[t, :] /= np.sum(delta[t, :])

        state_seq = np.zeros(num_data, dtype=int)
        state_seq[-1] = np.argmax(delta[-1, :])
        for t in range(num_data - 2, -1, -1):
            state_seq[t] = psi[t + 1, state_seq[t + 1]]
        log_prob = np.sum(np.log(np.max(delta, axis=1) + 1e-12))
        return state_seq, log_prob

    def train(self, data, num_iterations, num_states):
        """
        Baum–Welch training loop. At each iteration, compute the E‐step and update model parameters.
        """
        for iteration in range(num_iterations):
            print(f"Training iteration {iteration + 1} of {num_iterations}")
            # Gradually increase transition flexibility
            transition_constraint = max(0.5, 1.0 - iteration/num_iterations)
            
            # Modified E-step with constraints
            gamma_vals, xi = self.e_step(data)
            
            # Apply curriculum learning to transitions
            constrained_xi = xi * transition_constraint
            self.trans_prob = self.update_transition_probabilities(constrained_xi, num_states)

            self.emission_means, self.emission_covs, self.nu = self.update_emission_parameters(
                data, gamma_vals, self.num_states,
                make_positive_definite=self.make_positive_definite,
                estimate_nu=self.estimate_nu
            )
        return self.trans_prob, self.emission_means, self.emission_covs, self.nu

# =============================================================================
# Custom HMM Clustering using the Custom HMM Model
# =============================================================================
class CustomHMMClustering:
    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original, feelings, principal_components, no_of_jumps):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.df_csv_file_original = df_csv_file_original
        self.feelings = feelings
        self.principal_components = principal_components
        self.no_of_jumps = no_of_jumps

    def preprocess_data(self):
        # Group the data by Subject, Week and Session then skip rows according to no_of_jumps.
        split_dict_skip = {}
        for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            group = group.iloc[::self.no_of_jumps].copy()
            split_dict_skip[(subject, week, session)] = group

        self.df_csv_file = pd.concat(split_dict_skip.values())

        # Create a dictionary grouping by Subject, Week and Session again.
        split_dict = {}
        for (subject, week, session), group in self.df_csv_file.groupby(['Subject', 'Week', 'Session']):
            split_dict[(subject, week, session)] = group.copy()

        # Concatenate the groups (dropping the last row from each group if needed).
        self.array = pd.concat([df[:-1] for df in split_dict.values()])
        self.array['number'] = range(self.array.shape[0])

    def perform_clustering(self, num_states, num_iterations, num_repetitions):
        """
        run multiple repetitions (with different random seeds)
        and then average the resulting outputs to obtain final parameters and state sequences.
        """
        num_emissions = len(self.feelings)
        data = self.array.iloc[:, 4:4+len(self.feelings)].values
        
        # Z-score normalization
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        data_normalized = (data - self.mean) / self.std

        # Rest of the method uses normalized data for training
        N = data_normalized.shape[0]

        # Preallocate arrays to store results for each repetition.
        all_trans_probs = np.zeros((num_states, num_states, num_repetitions))
        all_emission_means = np.zeros((num_states, num_emissions, num_repetitions))
        all_emission_covs = np.zeros((num_states, num_emissions, num_emissions, num_repetitions))
        all_fs = np.zeros((N, num_states, num_repetitions))
        all_state_seqs = np.zeros((N, num_repetitions))
        all_log_probs = np.zeros(num_repetitions)
        all_log_liks = np.zeros(num_repetitions)

        for rep in range(num_repetitions):
            print(f"Repetition {rep + 1} of {num_repetitions}")
            # Set random seeds for reproducibility.
            np.random.seed(12345 + rep)
            random.seed(12345 + rep)
            
            # Create and train the model.
            model = HMMModel(num_states, num_emissions=num_emissions, data=data_normalized,  random_seed=12345 + rep)
            model.train(data_normalized, num_iterations=num_iterations, num_states = num_states)
            
            # Decode state sequence and get log probability.
            state_seq, log_prob = model.decode(data_normalized)
            # Obtain forward-backward results.
            _, _, fs, log_lik = model.forward_backward(data_normalized)
            
            # Optionally print shapes to verify.
            print(f"Transition probabilities shape: {model.trans_prob.shape}")
            print(f"Emission means shape: {model.emission_means.shape}")
            print(f"Emission covariances shape: {model.emission_covs.shape}")
            print(f"Degrees of freedom shape: {model.nu.shape}")
            
            # Save outputs for this repetition.
            all_trans_probs[:, :, rep] = model.trans_prob
            all_emission_means[:, :, rep] = model.emission_means
            all_emission_covs[:, :, :, rep] = model.emission_covs
            all_fs[:, :, rep] = fs
            all_state_seqs[:, rep] = state_seq
            all_log_probs[rep] = log_prob
            all_log_liks[rep] = log_lik

        
        # Average results across repetitions.
        avg_trans_prob = np.mean(all_trans_probs, axis=2)
        avg_emission_means = np.mean(all_emission_means, axis=2)
        avg_emission_covs = np.mean(all_emission_covs, axis=3)
        self.avg_fs = np.mean(all_fs, axis=2)
        # For state sequence, take the mode at each time point.
        avg_state_seq = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_state_seqs)
        avg_log_prob = np.mean(all_log_probs)
        avg_log_lik = np.mean(all_log_liks)

        print(f"Selected clusters with average log probability: {avg_log_prob}")

        # Assign the averaged state sequence and emission means as cluster centres.
        self.array['labels'] = avg_state_seq
        self.labels_fin = self.array['labels']
        self.cluster_centres_fin = avg_emission_means * self.std + self.mean
        
    def calculate_dictionary_clust_labels(self):
        """
        Calculate a dictionary mapping each unique cluster label to a human-readable string.
        For example, if the labels are 0, 1, 2 then this creates:
          {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}
        """
        unique_labels = sorted(self.array['labels'].unique())
        self.dictionary_clust_labels = {label: f"Cluster {label+1}" for label in unique_labels}

    def plot_results(self):
        """
        Project the data onto the principal components and plot the state assignments.
        The cluster centres (emission means) are also projected and plotted as arrows.
        """
        # Compute the principal component projection for the data.
        data_features = self.array[self.feelings]  # Ensure correct feature extraction
        projected = data_features.dot(self.principal_components.T)
        self.array["principal component 1"] = projected.iloc[:, 0]
        self.array["principal component 2"] = projected.iloc[:, 1]

        # Create a scatter plot of the state assignments.
        num_states = np.max(self.labels_fin) + 1
        cmap = plt.get_cmap('tab10')
        colours = [cmap(i) for i in range(num_states)]
        plt.figure(figsize=(8, 6))
        plt.scatter(self.array["principal component 1"],
                    self.array["principal component 2"],
                    c=[colours[label] for label in self.labels_fin],
                    s=1)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("HMM State Assignments (Custom t–Distribution HMM)")
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + 'HMM_state_scatter_plot.png')
        plt.close()

        # Correctly standardize cluster centers before projection
        centres_normalized = (self.cluster_centres_fin - self.mean) / self.std
        centres_projected = centres_normalized.dot(self.principal_components.T)
        
        plt.figure(figsize=(8, 6))
        for i in range(centres_projected.shape[0]):
            plt.arrow(0, 0, centres_projected[i, 0], centres_projected[i, 1],
                    head_width=0.01, head_length=0.05, linewidth=0.5, color=colours[i], length_includes_head=True)
            plt.text(centres_projected[i, 0], centres_projected[i, 1], f'State {i+1}', color=colours[i])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("State Centres for Custom HMM (Standardized)")
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + 'state_centres_custom_HMM.png')
        plt.close()

    def _plot_cluster_features(self):
        """Plot bar charts of emission means for each cluster in original feature space"""
        for cluster_idx in range(self.cluster_centres_fin.shape[0]):
            plt.figure(figsize=(10, 6))
            means = self.cluster_centres_fin[cluster_idx]
            sorted_indices = np.argsort(means)[::-1]
            
            plt.bar(range(len(self.feelings)), means[sorted_indices], 
                    color=plt.cm.tab10(cluster_idx))
            plt.xticks(range(len(self.feelings)), 
                       np.array(self.feelings)[sorted_indices], 
                       rotation=45, ha='right')
            plt.title(f'State {cluster_idx+1} - Feature Means')
            plt.ylabel('Mean Value')
            plt.tight_layout()
            plt.savefig(self.savelocation_TET +
                        f'state_{cluster_idx+1}_feature_means.png')
            plt.close()

    def _create_cluster_summary(self):
        """Create text file summarizing key characteristics of each cluster"""
        summary = []
        for cluster_idx in range(self.cluster_centres_fin.shape[0]):
            means = self.cluster_centres_fin[cluster_idx]
            sorted_features = sorted(zip(self.feelings, means), 
                                    key=lambda x: x[1], reverse=True)
            
            # Get top 3 features
            top_features = [f"{feat} ({val:.2f})" for feat, val in sorted_features[:4]]
            
            # Get bottom 3 features
            bottom_features = [f"{feat} ({val:.2f})" for feat, val in sorted_features[-4:]]
            
            summary.append(f"""
            State {cluster_idx+1}:
            - Dominant features: {', '.join(top_features)}
            - Lowest features: {', '.join(bottom_features)}
            - Mean vector norm: {np.linalg.norm(means):.2f}
            """)
        
        # Save to file
        with open( self.savelocation_TET+'cluster_summary.txt', 'w') as f:
            f.write("\n".join(summary))

    def analyze_transitions(self, num_states):
        """
        Analyze transitions per group (Subject, Week, Session) using dynamic thresholding.
        Each transition tuple: (transition_start, transition_end, from_state, to_state)
        """
        self.group_transitions = {}  # dictionary to store transitions per group

        # Loop over each group (Subject, Week, Session)
        for heading, group in self.array.groupby(['Subject', 'Week', 'Session']):
            group_labels = group['labels'].values
            
            # Determine which columns to use for features.
            # If self.avg_fs is a DataFrame, use its columns; otherwise, assume self.avg_fs is a NumPy array
            # and use self.feelings as the feature names.
            if hasattr(self.avg_fs, 'columns'):
                fs_columns = self.avg_fs.columns
            else:
                fs_columns = self.feelings  # adjust this if your features are stored differently

            # Extract the feature matrix for this group.
            group_fs = group[fs_columns].values
            
            # Calculate the dynamic threshold for this group.
            threshold = self.calculate_dynamic_threshold(group[fs_columns])
            min_state_duration = 5  # adjust as needed

            transitions = []
            # Use segment_start to mark the beginning of the current state segment.
            segment_start = 0

            # Iterate over the state sequence for this group, starting at the second element.
            for i in range(1, len(group_labels)):
                # Check if the state changed between the previous index and the current one.
                if group_labels[i] != group_labels[i - 1]:
                    # Only consider the transition if the current segment is long enough.
                    if i - segment_start >= min_state_duration:
                        # new_state is the state we transitioned into.
                        new_state = group_labels[i]
                        # Set initial boundaries: the change occurs between i-1 and i.
                        transition_start = i - 1  # Last index of the old state.
                        transition_end = i        # First index of the new state.

                        # Adjust the start boundary: look backward as long as the difference is significant.
                        while (transition_start > segment_start and
                            np.abs(group_fs[transition_start, new_state] - group_fs[transition_start - 1, new_state]) > threshold / 2):
                            transition_start -= 1

                        # Adjust the end boundary: look forward as long as the difference remains significant.
                        while (transition_end < len(group_labels) - 1 and
                            np.abs(group_fs[transition_end, new_state] - group_fs[transition_end + 1, new_state]) > threshold / 2):
                            transition_end += 1

                        # Append the transition tuple: (start index, end index, from_state, to_state)
                        transitions.append((transition_start, transition_end, group_labels[i - 1], new_state))

                    # Update the segment_start to the current index for the next segment.
                    segment_start = i

            # Save the transitions for this group.
            self.group_transitions[heading] = transitions

    def calculate_dynamic_threshold(self, avg_fs):
        # Compute absolute changes in state probabilities
        changes = np.abs(np.diff(avg_fs, axis=0))
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        threshold = mean_change + 0.05 * std_change  # From Maria's paper (Page 19)
        return threshold

    def run(self, num_states, num_iterations, num_repetitions):
        """
        Execute the complete pipeline: preprocessing, clustering, and plotting.
        Also calculates the dictionary of cluster labels.
        """
        self.preprocess_data()
        self.perform_clustering(num_states, num_iterations, num_repetitions)
        self.calculate_dictionary_clust_labels()
        self.plot_results()
        self._plot_cluster_features()
        self._create_cluster_summary()
        self.analyze_transitions(num_states)
        return self.array, self.dictionary_clust_labels , self.group_transitions

# =============================================================================
# Visualiser Class for Trajectory Plotting
# =============================================================================
class Visualiser:
    def __init__(self, filelocation_TET, savelocation_TET, array, df_csv_file_original,
                 dictionary_clust_labels, principal_components, feelings, no_of_jumps, colours, transitions):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.array = array
        self.df_csv_file_original = df_csv_file_original
        self.dictionary_clust_labels = dictionary_clust_labels
        self.principal_components = principal_components
        self.feelings = feelings
        self.no_of_jumps = no_of_jumps
        self.color_map = colours
        self.group_transitions = transitions  

    def preprocess_data(self):
        """
        Project feelings data onto the principal components and group the data by Subject, Week, and Session.
        """
        self.array[["principal component 1 non-diff", "principal component 2 non-diff"]] = \
            self.array[self.feelings].dot(self.principal_components.T)
            
        self.traj_transitions_dict = {}
        self.traj_transitions_dict_original = {}
        
        for heading, group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            self.traj_transitions_dict_original[heading] = group
        for heading, group in self.array.groupby(['Subject', 'Week', 'Session']):
            self.traj_transitions_dict[heading] = group

    def plot_trajectories(self):
        """
        For each subject/group trajectory, plot the time series for each feeling and overlay:
        - Cluster-based shading (if available)
        - Vertical dashed lines indicating the start and end of transitions
        - Text annotations showing the state change (from_state → to_state)
        """
        time_jump = 28  # as in the original implementation

        for heading, value in self.traj_transitions_dict_original.items():
            plt.figure()
            
            # Build the time axis for this group.
            starting_time = 0
            time_array = np.arange(starting_time, starting_time + time_jump * value.shape[0], time_jump)
            
            # Plot each feeling’s time series (scaled as needed).
            for feeling in self.feelings:
                plt.plot(time_array, value[feeling] * 10, label=feeling)
            
            # Clean up the title from the heading tuple.
            combined = ''.join(map(str, heading))
            cleaned = combined.replace("\\", "").replace("'", "").replace(" ", "").replace("(", "").replace(")", "")
            plt.title(cleaned)
            plt.xlabel('Time/s')
            plt.ylabel('Rating')
            plt.tight_layout()
            
            # --- Cluster-based shading (if applicable) ---
            if heading in self.traj_transitions_dict:
                traj_group = self.traj_transitions_dict[heading]
                prev_color_val = traj_group['labels'].iloc[0]
                start_index = 0
                for index, color_val in enumerate(traj_group['labels']):
                    if color_val != prev_color_val or index == traj_group.shape[0] - 1:
                        if index != traj_group.shape[0] - 1:
                            end_index = index * (time_jump * self.no_of_jumps)
                        else:
                            end_index = time_array[-1]
                        plt.axvspan(start_index * (time_jump * self.no_of_jumps), end_index,
                                    facecolor=self.color_map.get(prev_color_val, 'grey'), alpha=0.3)
                        start_index = index
                        prev_color_val = color_val
            
            # --- Transition markers with labels (per group) ---
            if heading in self.group_transitions:
                for (start_idx, end_idx, from_state, to_state) in self.group_transitions[heading]:
                    start_time = start_idx * (time_jump * self.no_of_jumps)
                    end_time = end_idx * (time_jump * self.no_of_jumps)
                    plt.axvline(x=start_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
                    plt.axvline(x=end_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
                    # Calculate mid-point for the annotation.
                    mid_time = (start_time + end_time) / 2
                    # Position the text near the top of the plot.
                    y_pos = plt.ylim()[1] * 0.95
                    plt.text(mid_time, y_pos, f'State {from_state+1} → {to_state+1}', 
                            horizontalalignment='center', verticalalignment='top', fontsize=8, color='black')
                    # Optionally, uncomment the next line to add a shaded region between markers.
                    # plt.axvspan(start_time, end_time, facecolor='gray', alpha=0.2)
            
            # --- Legend ---
            cluster_patches = [mpatches.Patch(color=color, label=f'Cluster {cluster}')
                            for cluster, color in self.color_map.items()]
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.extend(cluster_patches)
            labels.extend([f'Cluster {label}' for label in self.dictionary_clust_labels.values()])
            plt.legend(handles=handles, labels=labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # --- Save the figure ---
            save_path = os.path.join(self.savelocation_TET, f'HMM_stable_cluster_centroids{cleaned}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    def run(self):
        self.preprocess_data()
        self.plot_trajectories()

# ==================================================================================|
# Jump Analysis Class (using raw feelings rather than differences like in k-means)  |
# ==================================================================================|

class HMMJumpAnalysis:
    """
    This class performs jump analysis for the custom HMM clustering method.
    It iterates over different time‐step (jump) values, running the clustering
    each time and computing stability and consistency measures.
    """
    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original,
                 feelings, principal_components, num_states, num_iterations, num_repetitions):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.df_csv_file_original = df_csv_file_original
        self.feelings = feelings
        self.principal_components = principal_components
        self.num_states = num_states
        self.num_iterations = num_iterations
        self.num_repetitions = num_repetitions

    def determine_optimal_jumps(self):
        stability_scores = []
        consistency_scores = []
        
        for jump in tqdm(range(1, 15), desc="Evaluating Jumps"):
            # Train with current jump
            cluster = CustomHMMClustering(..., no_of_jumps=jump)
            cluster.run(num_states=3, num_iterations=30)
            
            # Calculate stability (ratio of dominant cluster)
            counts = np.bincount(cluster.array['labels'])
            stability = np.max(counts)/np.sum(counts)
            
            # Calculate consistency (agreement with full data)
            full_cluster = CustomHMMClustering(..., no_of_jumps=1)
            full_cluster.run(num_states=3)
            consistency = np.mean(cluster.array['labels'] == full_cluster.array['labels'][::jump])
            
            stability_scores.append(stability)
            consistency_scores.append(consistency)
            
        # Find optimal jump (max stability while maintaining >80% consistency)
        optimal_jump = np.argmax([s*(c>0.8) for s,c in zip(stability_scores, consistency_scores)]) + 1
        return optimal_jump

    def determine_no_jumps_stability(self):
        """
        For each jump value, run HMM clustering and calculate the ratio between the count
        of the dominant (most frequent) state and the sum of the counts of the other states.
        Additionally, check whether the dominant state corresponds to the state with the smallest
        magnitude (i.e. the smallest L2 norm of the emission mean). A plot is generated to display
        the relationship between the number of time jumps and the stability measure.
        """
        y_labels = []
        x_labels = []
        
        for jump in range(1, 30):
            print(f"Processing jump value: {jump}")
            # Run the HMM clustering with the current jump value.
            hmm_cluster = CustomHMMClustering(
                filelocation_TET=self.filelocation_TET,
                savelocation_TET=self.savelocation_TET + str(jump),
                df_csv_file_original=self.df_csv_file_original,
                feelings=self.feelings,
                principal_components=self.principal_components,
                no_of_jumps=jump
            )
            # Run the complete clustering pipeline.
            cluster_array, _ = hmm_cluster.run(self.num_states, self.num_iterations, self.num_repetitions)
            
            # Retrieve the state labels from the clustering output.
            labels = cluster_array['labels'].values
            unique, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique, counts))
            print(f"For jump {jump}, cluster distribution: {label_counts}")
            
            # Determine the dominant cluster (with the highest count).
            dominant_cluster = unique[np.argmax(counts)]
            
            # Calculate the magnitude (L2 norm) for each state's emission mean.
            emission_means = hmm_cluster.cluster_centres_fin  # Averaged emission means from HMM clustering
            magnitudes = [np.linalg.norm(emission_means[i]) for i in range(emission_means.shape[0])]
            smallest_magnitude_state = np.argmin(magnitudes)
            
            if dominant_cluster == smallest_magnitude_state:
                print("Dominant state matches the state with the smallest magnitude: True")
            else:
                print("Dominant state matches the state with the smallest magnitude: False")
            
            # Calculate the ratio of the dominant state's count to the sum of the other counts.
            if sum(counts) - np.max(counts) > 0:
                ratio = np.max(counts) / (sum(counts) - np.max(counts))
            else:
                ratio = np.nan
            y_labels.append(ratio)
            x_labels.append(jump)
        
        # Plot the stable cluster dominance ratio as a function of jump value.
        plt.figure()
        plt.plot(x_labels, y_labels, marker='o')
        plt.title('Stable Cluster Dominance with Number of Time Jumps')
        plt.xlabel('Number of Time Jumps')
        plt.ylabel('Dominant Cluster Count : Other Clusters Count')
        save_path = os.path.join(self.savelocation_TET, f'HMM_stable_cluster_dominance_jump.png')
        plt.savefig(save_path)
        

    def determine_no_jumps_consistency(self):
        """
        For each jump value, this method measures the consistency of the HMM clustering.
        The approach is to downsample the original data according to the jump value,
        propagate the state labels to the original dataset, and then compute the proportion
        of correct assignments based on the nearest emission mean. The ratio (Hughes' measure)
        is plotted against the number of time jumps.
        """
        hughes_ratios = []
        x_jumps = []
        
        for jump in range(1, 30):
            print(f"Processing jump value for consistency: {jump}")
            # Run HMM clustering with the current jump value.
            hmm_cluster = CustomHMMClustering(
                filelocation_TET=self.filelocation_TET,
                savelocation_TET=self.savelocation_TET,
                df_csv_file_original=self.df_csv_file_original,
                feelings=self.feelings,
                principal_components=self.principal_components,
                no_of_jumps=jump
            )
            cluster_array, _ = hmm_cluster.run(self.num_states, self.num_iterations, self.num_repetitions)
            labels_downsampled = cluster_array['labels'].values
            emission_means = hmm_cluster.cluster_centres_fin
            
            # Downsample the original data and record original indices.
            downsampled_groups = []
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                group_ds = group.iloc[::jump].copy()
                group_ds = group_ds[:-1]  # Ensure consistent length
                group_ds['Original_Index'] = group_ds.index
                downsampled_groups.append(group_ds)
            df_downsampled = pd.concat(downsampled_groups)
            # Assign the downsampled cluster labels.
            df_downsampled['Cluster_Label'] = labels_downsampled
            
            # Initialise the Cluster_Label column in the original dataframe.
            df_original = self.df_csv_file_original.copy()
            df_original['Cluster_Label'] = np.nan
            
            # Propagate the downsampled labels back to the original data.
            for _, row in df_downsampled.iterrows():
                original_index = row['Original_Index']
                label = row['Cluster_Label']
                group_info = df_original.loc[original_index, ['Subject', 'Week', 'Session']]
                group_mask = (
                    (df_original['Subject'] == group_info['Subject']) &
                    (df_original['Week'] == group_info['Week']) &
                    (df_original['Session'] == group_info['Session'])
                )
                group_indices = df_original[group_mask].index.tolist()
                pos_in_group = group_indices.index(original_index)
                start_idx = pos_in_group - (pos_in_group % jump)
                end_idx = min(start_idx + jump, len(group_indices))
                for i in range(start_idx, end_idx):
                    df_original.at[group_indices[i], 'Cluster_Label'] = label
            
            # Calculate Hughes' measure
            correct = 0
            total = 0
            for idx, row in df_original.dropna(subset=['Cluster_Label']).iterrows():
                true_label = row['Cluster_Label']
                features = row[self.feelings].values
                features_normalized = (features - hmm_cluster.mean) / hmm_cluster.std
                distances = [euclidean(features_normalized, centre) 
                        for centre in hmm_cluster.cluster_centres_fin]
                predicted_label = np.argmin(distances)
                if predicted_label == true_label:
                    correct += 1
                total += 1
            
            hughes_ratio = correct / total if total > 0 else np.nan
            hughes_ratios.append(hughes_ratio)
            x_jumps.append(jump)
        
        # Plot the consistency measure as a function of jump value.
        plt.figure()
        plt.plot(x_jumps, hughes_ratios, marker='o')
        plt.title('HMM Cluster Consistency (Hughes Measure)')
        plt.xlabel('Number of Time Jumps')
        plt.ylabel('Correct Assignment Ratio')
        save_path = os.path.join(self.savelocation_TET, f'HMM_consistency_plot.png')
        plt.savefig(save_path)
        

    def determine_no_of_jumps_autocorrelation(self):
        """
        Compute the average autocorrelation function (ACF) for each feeling across all subjects,
        weeks and sessions, using a fixed number of lags. A plot is generated to display the
        average ACF for each feeling.
        """
        split_dict = {}
        # Group by Subject, Week and Session.
        for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            split_dict[(subject, week, session)] = group[self.feelings].copy()
        
        acf_results = {feeling: [] for feeling in self.feelings}
        n_lags = 30  # Number of lags
        
        for key, df in split_dict.items():
            for feeling in self.feelings:
                acf_vals = acf(df[feeling], nlags=n_lags, fft=True)
                acf_results[feeling].append(acf_vals)
        
        # Average the autocorrelation values for each feeling.
        acf_averages = {feeling: np.mean(np.vstack(acf_results[feeling]), axis=0) for feeling in self.feelings}
        
        plt.figure(figsize=(12, 8))
        for feeling, acf_vals in acf_averages.items():
            plt.plot(acf_vals, label=feeling)
        
        plt.title('Average Autocorrelation Function for Each Feeling')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend(title='Feeling', loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.savelocation_TET, f'HMM_autocorrelation.png')
        plt.savefig(save_path)