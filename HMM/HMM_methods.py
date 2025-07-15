import os
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations, product
from scipy.spatial import distance
from scipy.spatial.distance import cdist, euclidean
from scipy.linalg import inv, det
from scipy.special import psi, polygamma, gamma 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
import seaborn as sns
import networkx as nx
from statsmodels.tsa.stattools import acf
from hmmlearn import hmm  # Only if you still need it somewhere
from tqdm import tqdm  # For progress bars
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import chi2_contingency
from sklearn.metrics.cluster import normalized_mutual_info_score


# =============================================================================|
# Configuration via YAML                                                       |
# =============================================================================|
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

# =============================================================================|
# CSV File Handling                                                            |
# =============================================================================|
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

# =============================================================================|
# Principal Component Finder                                                   |
# =============================================================================|
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
        self.scaled_data = scaler.fit_transform(self.csv_file_TET)

        # Compute PCA
        pca = PCA(n_components=no_dimensions)
        self.principal_components = pca.fit(self.scaled_data).components_
        self.explained_variance_ratio = pca.explained_variance_ratio_

    def PCA_TOT(self):
        """
        Projects the data onto the principal components and plots bar charts for each component.
        Returns the principal components, explained variance ratio, and the transformed data.
        """

        # Project STANDARDIZED DATA 
        df_TET_feelings_prin = pd.DataFrame(self.scaled_data @ self.principal_components.T,  
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
        if split_df_array == None:
            df_TET_feelings_prin_dict = None
        else:
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

# =============================================================================|
# Custom HMM Model using multivariate t–distribution                           |
# =============================================================================|
class HMMModel:
    def __init__(self, num_base_states, num_emissions, data=None, random_seed=12345, base_prior=0, extra_prior=0, transition_temp=0):
        """
        The model will have an extra state in addition to the base states.
        num_base_states: number of "pure" states.
        Total number of states = num_base_states + 1.
        base_prior: Dirichlet prior for base state transitions (default 1.0)
        extra_prior: Dirichlet prior for transitions involving the extra state (default 1.0)
        transition_temp: temperature for softening transition matrix update (default 1.0)
        """
        self.num_base_states = num_base_states
        self.num_states = num_base_states + 1  # Extra state for transitions.
        self.base_prior = base_prior
        self.extra_prior = extra_prior
        self.transition_temp = transition_temp
        np.random.seed(random_seed)
        random.seed(random_seed)

        if data is not None:
            # Initialize emission means using K-means on base states only.
            kmeans = KMeans(n_clusters=num_base_states, init='k-means++', random_state=random_seed).fit(data)
            base_centers = kmeans.cluster_centers_
            self.emission_means = np.zeros((self.num_states, num_emissions))
            # Base states get the k-means centers.
            self.emission_means[:num_base_states] = base_centers
            # Compute an offset based on spread of base centers.
            offset = 0.2 * (np.max(base_centers, axis=0) - np.min(base_centers, axis=0))
            # Transitional state gets the average plus offset.
            self.emission_means[-1] = np.mean(base_centers, axis=0) - offset 

            # Diagonal-dominated initialization for transition probabilities.
            base_prob = 0.2
            self.trans_prob = np.eye(self.num_states) * base_prob + \
                              np.ones((self.num_states, self.num_states)) * (1 - base_prob) / (self.num_states - 1)
            self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)
        else:
            self.emission_means = np.random.randn(self.num_states, num_emissions)
            self.trans_prob = np.random.dirichlet(np.ones(self.num_states), size=self.num_states)

        # Initialize emission covariances and degrees of freedom for t–distribution.
        self.emission_covs = np.stack([
            np.eye(num_emissions) + 1e-4 * np.random.randn(num_emissions, num_emissions)
            for _ in range(self.num_states)
        ])
        self.nu = np.clip(np.random.gamma(5, 1, self.num_states), 2, 10)

    @staticmethod
    def mvtpdf(x, mu, Sigma, nu):
        d = len(mu)
        x_mu = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        mahalanobis = x_mu @ Sigma_inv @ x_mu

        log_norm = (np.log(gamma((nu + d)/2)) 
                    - np.log(gamma(nu/2)) 
                    - (d/2) * np.log(nu * np.pi) 
                    - 0.5 * np.log(det_Sigma) 
                    - ((nu + d)/2) * np.log(1 + mahalanobis/nu))
        return np.exp(log_norm)

    @staticmethod
    def make_positive_definite(cov_matrix):
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # Ensure symmetry.
        d = cov_matrix.shape[0]
        cov_matrix += 1e-4 * np.eye(d)  # Regularization.
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.clip(eigenvalues, 1e-4, None)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    @staticmethod
    def estimate_nu(gamma_vals, data, mean, covariance, max_iter=100):
        N, d = data.shape
        diff = data - mean
        inv_cov = np.linalg.pinv(covariance)
        r = np.einsum('ij,ij->i', diff @ inv_cov, diff)
        nu = np.clip(2/(np.mean(r/(d + 2)) - 1), 2, 10)
        for _ in range(max_iter):
            w = (nu + d) / (nu + r)
            numerator = np.sum(gamma_vals * (np.log(w) - w))
            denominator = np.sum(gamma_vals * (psi((nu + d) / 2) - np.log((nu + d) / 2)))
            if abs(denominator) < 1e-12:
                break
            nu_new = nu - numerator / denominator
            if abs(nu_new - nu) < 1e-4:
                break
            nu = np.clip(nu_new, 2, 20)
        return nu

    @staticmethod
    def update_transition_probabilities(xi, num_states, base_prior=0, extra_prior=0, extra_self_prior=0, transition_temp=1.0):
        """
        Use a Dirichlet prior with differential weights to encourage state persistence 
        and reduce transitions into/out of the extra state.
        base_prior, extra_prior: user-settable priors for base and extra state transitions
        transition_temp: temperature for softening transition matrix update
        """
        # 1) start with base_prior everywhere
        prior = np.full((num_states, num_states), base_prior)

        # 2) down‑weight any row or column involving the extra state
        prior[-1, :] = extra_prior
        prior[:, -1] = extra_prior

        # 3) but make the extra→extra prior *very* small
        prior[-1, -1] = extra_self_prior

        # 4) add counts and apply temperature
        trans_prob = np.sum(xi, axis=2) + prior
        trans_prob = trans_prob ** (1.0 / transition_temp)
        trans_prob /= trans_prob.sum(axis=1, keepdims=True)


        return trans_prob

    @staticmethod
    def update_emission_parameters(data, gamma, num_states, make_positive_definite, estimate_nu):
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
            covariances[j, :, :] += np.eye(num_emissions) * 1e-4
            if j > 0:  # Optionally push means apart (currently no shift applied)
                means[j, :] += 0 * (means[j, :] - np.mean(means[:j, :], axis=0))
            nu[j] = estimate_nu(gamma_j, data, means[j, :], covariances[j, :, :])
        return means, covariances, nu

    def _compute_emission_probs(self, data):
        num_data = data.shape[0]
        emission_probs = np.zeros((num_data, self.num_states))
        for j in range(self.num_states):
            pdf_func = lambda x: self.mvtpdf(x, self.emission_means[j], self.emission_covs[j], self.nu[j])
            emission_probs[:, j] = np.apply_along_axis(pdf_func, 1, data)
        return emission_probs

    def e_step(self, data):
        num_data = data.shape[0]
        emission_probs = self._compute_emission_probs(data)
        alpha = np.zeros((num_data, self.num_states))
        beta = np.zeros((num_data, self.num_states))
        gamma_vals = np.zeros((num_data, self.num_states))
        xi = np.zeros((self.num_states, self.num_states, num_data - 1))

        # Forward pass.
        alpha[0, :] = emission_probs[0, :] * (1 / self.num_states)
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
            # build the unnormalized ξ_{i,j}(t)
            # shape: (num_states, num_states)
            unnorm = (
                alpha[t, :][:, np.newaxis]
                * self.trans_prob
                * (emission_probs[t + 1, :] * beta[t + 1, :])[np.newaxis, :]
            )
            # full normalization over all i,j
            denom = unnorm.sum()
            if denom > 0:
                xi[:, :, t] = unnorm / denom
            else:
                xi[:, :, t] = 0  # avoid division by zero

        return gamma_vals, xi

    def forward_backward(self, data):
        num_data = data.shape[0]
        log_emission_probs = np.log(self._compute_emission_probs(data) + 1e-12)
        log_trans_prob = np.log(self.trans_prob + 1e-12)
        log_alpha = np.full((num_data, self.num_states), -np.inf)
        log_beta = np.full((num_data, self.num_states), -np.inf)

        log_alpha[0, :] = log_emission_probs[0, :] - np.log(self.num_states)
        for t in range(1, num_data):
            for j in range(self.num_states):
                log_alpha[t, j] = log_emission_probs[t, j] + logsumexp(log_alpha[t - 1, :] + log_trans_prob[:, j])
        
        log_beta[-1, :] = 0
        for t in range(num_data - 2, -1, -1):
            for j in range(self.num_states):
                log_beta[t, j] = logsumexp(log_trans_prob[j, :] + log_emission_probs[t + 1, :] + log_beta[t + 1, :])
        
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        log_lik = logsumexp(log_alpha[-1, :])
        
        return log_alpha, log_beta, np.exp(log_gamma), log_lik

    def decode(self, data):
        num_data = data.shape[0]
        log_emission_probs = np.log(self._compute_emission_probs(data) + 1e-12)
        log_trans_prob = np.log(self.trans_prob + 1e-12)
        log_delta = np.full((num_data, self.num_states), -np.inf)
        psi = np.zeros((num_data, self.num_states), dtype=int)

        log_delta[0, :] = log_emission_probs[0, :] - np.log(self.num_states)
        for t in range(1, num_data):
            for j in range(self.num_states):
                temp = log_delta[t - 1, :] + log_trans_prob[:, j]
                psi[t, j] = np.argmax(temp)
                log_delta[t, j] = log_emission_probs[t, j] + np.max(temp)
        
        log_prob = np.max(log_delta[-1, :])
        last_state = np.argmax(log_delta[-1, :])
        state_seq = np.zeros(num_data, dtype=int)
        state_seq[-1] = last_state
        for t in range(num_data - 2, -1, -1):
            state_seq[t] = psi[t + 1, state_seq[t + 1]]
        
        return state_seq, log_prob 

    def train(self, data, num_iterations, transition_contributions, transition_constraint_lim=0.5):
        for iteration in range(num_iterations):
            print(f"Training iteration {iteration + 1} of {num_iterations}")
            # 1) compute how hard to penalize inter‑state jumps this iteration
            transition_constraint = max(
                0.25,
                1.0 - (1.0 - transition_constraint_lim) * (iteration / num_iterations)
            )

            # 2) E‑step: get soft counts xi[i,j,t]
            gamma_vals, xi = self.e_step(data)

            # 3) build a scaling mask that only down‑weights base⟷extra transitions
            scaling = np.ones_like(xi)               # default = no scaling
            # scale all (extra→base) and (base→extra) by the penalty factor:
            scaling[-1, :] = transition_constraint * transition_contributions
            scaling[:, -1] = transition_constraint * transition_contributions
            # but then exempt the (extra→extra) self‑transition:
            #scaling[-1, -1] = 0.01

            # 4) apply scaling to your xi counts
            constrained_xi = xi * scaling

            # 5) re‑estimate the transition matrix with your priors + temperature
            self.trans_prob = self.update_transition_probabilities(
                constrained_xi, self.num_states,
                base_prior=self.base_prior,
                extra_prior=self.extra_prior,
                extra_self_prior=0,
                transition_temp=self.transition_temp
            )

            # 6) clip & renormalize gammas, then M‑step for the emissions
            gamma_vals = np.clip(gamma_vals, 1e-3, 1.0)
            gamma_vals /= gamma_vals.sum(axis=1, keepdims=True)
            self.emission_means, \
            self.emission_covs, \
            self.nu = self.update_emission_parameters(
                data, gamma_vals, self.num_states,
                make_positive_definite=self.make_positive_definite,
                estimate_nu=self.estimate_nu
            )

        return self.trans_prob, self.emission_means, self.emission_covs, self.nu


# =============================================================================|
# Custom Clustering Pipeline Using the Transition-State HMM Model              |
# =============================================================================|
class CustomHMMClustering:
    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original, feelings, principal_components, no_of_jumps, transition_contributions, base_prior=0, extra_prior=0, transition_temp=1.0, transition_constraint_lim=0.5):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.df_csv_file_original = df_csv_file_original
        self.feelings = feelings
        self.principal_components = principal_components
        self.no_of_jumps = no_of_jumps
        self.has_week = 'Week' in df_csv_file_original.columns
        self.transition_contributions = transition_contributions
        self.base_prior = base_prior
        self.extra_prior = extra_prior
        self.transition_temp = transition_temp
        self.transition_constraint_lim = transition_constraint_lim

    def preprocess_data(self):
        group_keys = ['Subject', 'Week', 'Session'] if self.has_week else ['Subject', 'Session']
        split_dict_skip = {}
        for keys, group in self.df_csv_file_original.groupby(group_keys):
            group = group.iloc[::self.no_of_jumps].copy()
            split_dict_skip[keys] = group
        self.df_csv_file = pd.concat(split_dict_skip.values())
        split_dict = {}
        for keys, group in self.df_csv_file.groupby(group_keys):
            split_dict[keys] = group.copy()
        self.array = pd.concat(split_dict.values())
        self.array['number'] = range(self.array.shape[0])

    def _align_states(self, ref_model, target_model, num_states):
        cost_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(num_states):
                cost_matrix[i, j] = np.linalg.norm(
                    ref_model.emission_means[i] - target_model.emission_means[j]
                )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind

    def _permute_states(self, model, permutation):
        model.emission_means = model.emission_means[permutation]
        model.emission_covs = model.emission_covs[permutation]
        model.trans_prob = model.trans_prob[permutation][:, permutation]
        model.nu = model.nu[permutation]
        return model

    def _balance_clusters(self, num_states):
        label_counts = self.array['labels'].value_counts()
        small_clusters = label_counts[label_counts < 0.1 * len(self.array)].index
        for sc in small_clusters:
            mask = self.array['labels'] == sc
            distances = cdist(self.array.loc[mask, self.feelings], 
                              self.cluster_centres_fin, 'mahalanobis')
            new_labels = np.argmin(distances, axis=1)
            self.array.loc[mask, 'labels'] = new_labels

    def perform_clustering(self, num_base_states, num_iterations, num_repetitions, base_seed=12345):
        num_emissions = len(self.feelings)
        data = self.array[self.feelings].values
        
        # Z-score normalization.
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1.0
        data_normalized = (data - self.mean) / self.std

        N = data_normalized.shape[0]
        all_trans_probs = np.zeros((num_base_states + 1, num_base_states + 1, num_repetitions))
        all_emission_means = np.zeros((num_base_states + 1, num_emissions, num_repetitions))
        all_emission_covs = np.zeros((num_base_states + 1, num_emissions, num_emissions, num_repetitions))
        all_fs = np.zeros((N, num_base_states + 1, num_repetitions))
        all_state_seqs = np.zeros((N, num_repetitions))
        all_log_probs = np.zeros(num_repetitions)
        all_log_liks = np.zeros(num_repetitions)

        for rep in range(num_repetitions):
            print(f"Repetition {rep + 1} of {num_repetitions}")
            seed = base_seed + rep
            np.random.seed(seed)
            random.seed(seed)
            model = HMMModel(num_base_states, num_emissions=num_emissions, data=data_normalized, random_seed=seed,
                             base_prior=self.base_prior, extra_prior=self.extra_prior, transition_temp=self.transition_temp)
            model.train(data_normalized, num_iterations=num_iterations, transition_contributions = self.transition_contributions, transition_constraint_lim = self.transition_constraint_lim)
            
            state_seq, log_prob = model.decode(data_normalized)
            _, _, fs, log_lik = model.forward_backward(data_normalized)
            
            if rep == 0:
                reference_model = model
            else:
                alignment = self._align_states(reference_model, model, num_base_states + 1)
                model = self._permute_states(model, alignment)
                
            print(f"Transition probabilities shape: {model.trans_prob.shape}")
            print(f"Emission means shape: {model.emission_means.shape}")
            print(f"Emission covariances shape: {model.emission_covs.shape}")
            print(f"Degrees of freedom shape: {model.nu.shape}")
            
            all_trans_probs[:, :, rep] = model.trans_prob
            all_emission_means[:, :, rep] = model.emission_means
            all_emission_covs[:, :, :, rep] = model.emission_covs
            all_fs[:, :, rep] = fs
            all_state_seqs[:, rep] = state_seq
            all_log_probs[rep] = log_prob
            all_log_liks[rep] = log_lik

        avg_trans_prob = np.mean(all_trans_probs, axis=2)
        avg_emission_means = np.mean(all_emission_means, axis=2)
        avg_emission_covs = np.mean(all_emission_covs, axis=3)
        self.avg_fs = np.mean(all_fs, axis=2)
        avg_state_seq = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_state_seqs)
        avg_log_prob = np.mean(all_log_probs)
        avg_log_lik = np.mean(all_log_liks)
        gamma_columns = [f'gamma_{i}' for i in range(num_base_states + 1)]
        self.array[gamma_columns] = self.avg_fs

        print(f"Selected clusters with average log probability: {avg_log_prob}")

        self.array['labels'] = avg_state_seq
        self.labels_fin = self.array['labels']
        self.cluster_centres_fin = avg_emission_means[:num_base_states+1] * self.std + self.mean
        
    def calculate_dictionary_clust_labels(self):
        unique_labels = sorted(self.array['labels'].unique())
        self.dictionary_clust_labels = {label: f"Cluster {label+1}" for label in unique_labels}

    def recompute_cluster_centres(self):
        unique_labels = sorted(self.array['labels'].unique())
        cluster_centres = []
        for label in unique_labels:
            cluster_data = self.array[self.array['labels'] == label][self.feelings].values
            cluster_mean = cluster_data.mean(axis=0)
            cluster_centres.append(cluster_mean)
        self.cluster_centres_fin = np.array(cluster_centres)

    def plot_results(self):
        data_features = self.array[self.feelings]
        projected = data_features.dot(self.principal_components.T)
        self.array["principal component 1"] = projected.iloc[:, 0]
        self.array["principal component 2"] = projected.iloc[:, 1]

        num_clusters = np.max(self.labels_fin) + 1
        cmap = plt.get_cmap('tab10')
        colours = [cmap(i) for i in range(num_clusters)]
        fixed_colours = ['green', 'red', 'blue']
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.array["principal component 1"],
            self.array["principal component 2"],
            c=[fixed_colours[label] for label in self.labels_fin],
            s=1
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Transition–State HMM Assignments")
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + 'HMM_state_scatter_plot.png')
        plt.close()

        centres_normalized = (self.cluster_centres_fin - self.mean) / self.std
        centres_projected = centres_normalized.dot(self.principal_components.T)
        
        plt.figure(figsize=(8, 6))
        for i in range(centres_projected.shape[0]):
            plt.arrow(0, 0, centres_projected[i, 0], centres_projected[i, 1],
                      head_width=0.01, head_length=0.05, linewidth=0.5, color=colours[i], length_includes_head=True)
            plt.text(centres_projected[i, 0], centres_projected[i, 1], f'State {i+1}', color=colours[i])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Base State Centres")
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + 'state_centres_transition_HMM.png')
        plt.close()

    def _plot_cluster_features(self):
        for cluster_idx in range(self.cluster_centres_fin.shape[0]):
            plt.figure(figsize=(10, 6))
            means = self.cluster_centres_fin[cluster_idx]
            sorted_indices = np.argsort(means)[::-1]
            plt.bar(range(len(self.feelings)), means[sorted_indices], color=plt.cm.tab10(cluster_idx))
            plt.xticks(range(len(self.feelings)), np.array(self.feelings)[sorted_indices], rotation=45, ha='right')
            plt.title(f'Cluster {cluster_idx+1} - Feature Means')
            plt.ylabel('Mean Value')
            plt.tight_layout()
            plt.savefig(self.savelocation_TET + f'cluster_{cluster_idx+1}_feature_means.png')
            plt.close()

    def _create_cluster_summary(self):
        summary = []
        for cluster_idx in range(self.cluster_centres_fin.shape[0]):
            means = self.cluster_centres_fin[cluster_idx]
            sorted_features = sorted(zip(self.feelings, means), key=lambda x: x[1], reverse=True)
            top_features = [f"{feat} ({val:.2f})" for feat, val in sorted_features[:4]]
            bottom_features = [f"{feat} ({val:.2f})" for feat, val in sorted_features[-4:]]
            summary.append(f"""
            Cluster {cluster_idx+1}:
            - Dominant features: {', '.join(top_features)}
            - Lowest features: {', '.join(bottom_features)}
            - Mean vector norm: {np.linalg.norm(means):.2f}
            """)
        with open(self.savelocation_TET + 'cluster_summary.txt', 'w') as f:
            f.write("\n".join(summary))

    def post_process_cluster_three(self, cluster_three_label=2, gamma_threshold=0.55):
        """
        Reassign points in cluster 3 (the transition cluster) that have weak membership.
        If the gamma value for cluster 3 is below the gamma_threshold, reassign these points 
        to the base cluster with the highest gamma among the base clusters.
        """
        mask = (self.array['labels'] == cluster_three_label) & (self.array[f'gamma_{cluster_three_label}'] < gamma_threshold)
        for idx in self.array[mask].index:
            base_gammas = [self.array.at[idx, f'gamma_{i}'] for i in range(cluster_three_label)]
            new_label = np.argmax(base_gammas)
            self.array.at[idx, 'labels'] = new_label
        self.calculate_dictionary_clust_labels()

    def analyze_transitions(self, num_base_states, abrupt_gamma_threshold=0.6):
        self.array['transition_label'] = self.array['labels'].apply(lambda x: str(x + 1))
        self.copy = self.array.copy()
        group_keys = ['Subject', 'Week', 'Session'] if self.has_week else ['Subject', 'Session']
        self.group_transitions = {}
        gamma_columns = [f'gamma_{i}' for i in range(num_base_states + 1)]
        if not all(col in self.array.columns for col in gamma_columns):
            self.array[gamma_columns] = self.avg_fs

        base_slope_threshold = 0.06
        base_fraction_threshold = 0.25
        small_cluster_threshold = 0.1

        for heading, group in self.array.groupby(group_keys):
            group_labels = group['labels'].values
            group_gammas = group[gamma_columns].values
            group_indices = group.index
            total_in_group = len(group_labels)
            threshold = self.calculate_dynamic_threshold(self.avg_fs)
            min_state_duration = 5
            transitions = []
            segment_start = 0

            for i in range(1, len(group_labels)):
                if group_labels[i] != group_labels[i - 1]:
                    if (i - segment_start) >= min_state_duration:
                        from_state = group_labels[i - 1]
                        to_state = group_labels[i]
                        transition_start = i - 1
                        transition_end = i
                        count_from = np.sum(group_labels == from_state)
                        count_to = np.sum(group_labels == to_state)
                        is_from_small = count_from < (small_cluster_threshold * total_in_group)
                        is_to_small = count_to < (small_cluster_threshold * total_in_group)
                        use_adjusted = is_from_small or is_to_small
                        slope_threshold = base_slope_threshold * (0.8 if use_adjusted else 1.0)
                        fraction_threshold = base_fraction_threshold * (1.2 if use_adjusted else 1.0)
                        expansion_thresh = threshold * (0.3 if use_adjusted else 0.5)

                        while (transition_start > segment_start and
                               np.abs(group_gammas[transition_start, to_state] - 
                                      group_gammas[transition_start - 1, to_state]) > expansion_thresh):
                            transition_start -= 1
                        while (transition_end < len(group_labels) - 1 and
                               np.abs(group_gammas[transition_end, to_state] - 
                                      group_gammas[transition_end + 1, to_state]) > expansion_thresh):
                            transition_end += 1

                        window_gammas = group_gammas[transition_start:transition_end + 1, to_state]
                        gamma_diff = np.diff(window_gammas)
                        max_slope = np.max(np.abs(gamma_diff)) if len(gamma_diff) > 0 else 0
                        fraction_large_slopes = np.mean(np.abs(gamma_diff) >= 0.05) if len(gamma_diff) > 0 else 0

                        if max_slope >= slope_threshold and fraction_large_slopes <= fraction_threshold:
                            transition_type = "abrupt"
                        else:
                            transition_type = "gradual"
                        transitions.append((transition_start, transition_end, from_state, to_state, transition_type))
                        transition_indices = group_indices[transition_start:transition_end + 1]
                        transition_str = f"{from_state + 1} to {to_state + 1}"
                        self.array.loc[transition_indices, 'transition_label'] = transition_str

                    segment_start = i
            self.group_transitions[heading] = transitions
        self.gamma_values = self.array[gamma_columns].values
        self.abrupt_transition_mask = (self.gamma_values.max(axis=1) >= abrupt_gamma_threshold)

    def calculate_dynamic_threshold(self, avg_fs):
        changes = np.abs(np.diff(avg_fs, axis=0))
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        threshold = mean_change + 0.05 * std_change
        return threshold

    def plot_state_sequence(self):
        """
        Plots the state (cluster) assignment over time for the full dataset and saves the plot.
        """
        plt.figure(figsize=(12, 2))
        plt.plot(self.array['number'], self.array['labels'], drawstyle='steps-post', lw=1)
        plt.xlabel('Time Index')
        plt.ylabel('State')
        plt.title('State Sequence Over Time')
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation_TET, 'state_sequence_over_time.png'))
        plt.close()

    def plot_gamma_heatmap(self):
        """
        Plots a heatmap of gamma (state membership probabilities) over time.
        """
        gamma_cols = [col for col in self.array.columns if col.startswith('gamma_')]
        if not gamma_cols:
            return
        plt.figure(figsize=(12, 4))
        sns.heatmap(self.array[gamma_cols].T, cmap='viridis', cbar=True)
        plt.xlabel('Time Index')
        plt.ylabel('State')
        plt.title('State Membership Probabilities (Gamma)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation_TET, 'gamma_heatmap.png'))
        plt.close()

    def plot_transition_matrix(self):
        """
        Plots the HMM’s *learned* transition probability matrix as a heatmap.
        Falls back to empirical counts only if avg_trans_prob is missing.
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns

        if hasattr(self, 'avg_trans_prob'):
            M = self.avg_trans_prob
            plt.figure(figsize=(6, 5))
            sns.heatmap(M, annot=True, fmt=".2f", cmap='Blues')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.title('Learned Transition Probability Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.savelocation_TET, 'learned_transition_matrix.png'))
            plt.close()
        else:
            # fallback: empirical Viterbi counts
            num_states = self.cluster_centres_fin.shape[0]
            trans_mat = np.zeros((num_states, num_states))
            labels = self.labels_fin.values if hasattr(self.labels_fin, 'values') else self.labels_fin
            for i in range(len(labels) - 1):
                trans_mat[labels[i], labels[i + 1]] += 1
            row_sums = trans_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans_mat = trans_mat / row_sums

            plt.figure(figsize=(6, 5))
            sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap='Blues')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.title('Empirical Transition Probability Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.savelocation_TET, 'empirical_transition_matrix.png'))
            plt.close()

    def plot_state_duration_distribution(self):
        """
        Plots a histogram of state durations (how long each state persists).
        """
        labels = self.array['labels'].values
        durations = []
        current = labels[0]
        count = 1
        for l in labels[1:]:
            if l == current:
                count += 1
            else:
                durations.append((current, count))
                current = l
                count = 1
        durations.append((current, count))
        durations = pd.DataFrame(durations, columns=['state', 'duration'])
        plt.figure(figsize=(8, 4))
        for state in durations['state'].unique():
            plt.hist(durations[durations['state']==state]['duration'], bins=20, alpha=0.6, label=f'State {state+1}')
        plt.xlabel('Duration (consecutive time points)')
        plt.ylabel('Count')
        plt.title('State Duration Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation_TET, 'state_duration_distribution.png'))
        plt.close()

    def validate_middle_state(self):
        """
        Checks if the dominant cluster is the one with the smallest L2 norm (middle state validation).
        Saves the result to a text file for traceability.
        """
        labels = self.array['labels'].values
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique[np.argmax(counts)]
        emission_means = self.cluster_centres_fin
        magnitudes = [np.linalg.norm(emission_means[i]) for i in range(emission_means.shape[0])]
        smallest_magnitude_state = np.argmin(magnitudes)
        result = (dominant_cluster == smallest_magnitude_state)
        msg = f"Dominant cluster: {dominant_cluster}, Smallest L2 norm cluster: {smallest_magnitude_state}, Match: {result}\n"
        print(msg)
        # Save to file
        with open(os.path.join(self.savelocation_TET, 'middle_state_validation.txt'), 'w') as f:
            f.write(msg)

    def run(self, num_base_states, num_iterations, num_repetitions, base_seed = 12345):
        self.preprocess_data()
        self.perform_clustering(num_base_states, num_iterations, num_repetitions, base_seed =12345)
        self.calculate_dictionary_clust_labels()
        self.plot_results()
        self._plot_cluster_features()
        self._create_cluster_summary()
        self.post_process_cluster_three(cluster_three_label=2, gamma_threshold=0.55)
        if num_base_states == 3:
            num_base_states = num_base_states - 1
        self.analyze_transitions(num_base_states)
        # --- Added best-practice visualizations ---
        self.plot_state_sequence()
        self.plot_gamma_heatmap()
        self.plot_transition_matrix()
        self.plot_state_duration_distribution()
        # -----------------------------------------
        self.validate_middle_state()

        # === NEW: Plot & save “combined PC1/PC2 loading” for ALL features, per cluster ===
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # 1.  Grab the final centroids in original‐space:
        centres = self.cluster_centres_fin  # shape = (n_clusters, n_features)

        # 2.  Re‐apply the same z‐scaling, then project into PCA-space:
        centres_normalized = (centres - self.mean) / self.std
        centres_pca = centres_normalized.dot(self.principal_components.T)
        #    centres_pca[i] is now a length‑no_dimensions vector [PC1, PC2, PC3, …].

        # 3.  For each cluster i, compute “combined loading” =  pc1*loading₁ + pc2*loading₂:
        for i, (pc1, pc2, *_) in enumerate(centres_pca):
            combined_loading = (
                pc1 * self.principal_components[0]
                + pc2 * self.principal_components[1]
            )  # length = n_features

            # 4.  Plot ALL features on the x-axis, y = combined_loading
            plt.figure(figsize=(12, 6))
            inds = np.arange(len(self.feelings))
            plt.bar(inds, combined_loading, color='steelblue')
            plt.xticks(inds, self.feelings, rotation=45, ha='right')
            plt.ylabel('Combined loading (PC1·λ₁ + PC2·λ₂)')
            plt.title(f'Cluster {i+1} – Combined PCA‐loading for all features')
            plt.tight_layout()

            # 5.  Save to disk alongside your other figures
            filename = os.path.join(
                self.savelocation_TET,
                f'cluster_{i+1}_combined_loading_all_features.png'
            )
            plt.savefig(filename)
            plt.close()
        # === END NEW ===
        return self.array, self.dictionary_clust_labels, self.group_transitions, self.copy

# =============================================================================|
# Visualiser Class for Trajectory Plotting                                     |
# =============================================================================|
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
        self.feeling_colors = {feeling: self.color_map.get(i, 'black') 
                  for i, feeling in enumerate(self.feelings)}
        self.has_week = 'Week' in df_csv_file_original.columns

    def preprocess_data(self):
        group_keys = ['Subject', 'Week', 'Session'] if self.has_week else ['Subject', 'Session']
        self.traj_transitions_dict = {}
        self.traj_transitions_dict_original = {}
                
        for heading, group in self.df_csv_file_original.groupby(group_keys):
            self.traj_transitions_dict_original[heading] = group
        for heading, group in self.array.groupby(group_keys):
            self.traj_transitions_dict[heading] = group

    def annotate_state_durations(self, ax, time_array, transitions):
        """Add duration labels and transition markers to plot"""
        for (start_idx, end_idx, from_state, to_state, trans_type) in transitions:
            start_time = time_array[start_idx]
            end_time = time_array[end_idx]
            duration = end_time - start_time

            # Annotation
            ax.text((start_time + end_time)/2, ax.get_ylim()[1]*0.9,
                    f"{duration:.1f}s", ha='center', va='top', 
                    fontsize=8, color='black')
            
            # Transition markers
            ax.axvline(x=start_time, color='black', linestyle='--', 
                      alpha=0.5, linewidth=1)
            ax.axvline(x=end_time, color='black', linestyle='--', 
                      alpha=0.5, linewidth=1)
            
            # State change label
            ax.text(end_time, ax.get_ylim()[1]*0.90, 
                   f"{from_state+1} → {to_state+1}",
                   ha='left', va='top', fontsize=8, color='black')

    def annotate_conditions(self, ax, time_array, df):
        """Add breathwork condition labels"""
        prev_condition = df['Condition'].iloc[0]
        start_time = time_array[0]

        for idx, (t, condition) in enumerate(zip(time_array, df['Condition'])):
            if condition != prev_condition or idx == len(time_array)-1:
                end_time = t
                ax.text((start_time + end_time)/2, ax.get_ylim()[1]*0.95,
                        prev_condition, ha='center', va='top', 
                        fontsize=6, color='black')
                start_time = t
                prev_condition = condition

    def plot_trajectories(self):
        """Main plotting function with transition visualization"""
        time_jump = 28  # Original data sampling interval (seconds)

        def format_heading(heading):
            if isinstance(heading, tuple):
                # Try to detect common heading tuple patterns
                if len(heading) == 3:
                    subj, week, session = heading
                    return f"{subj} week {week} run {str(session).zfill(2)}"
                elif len(heading) == 2:
                    subj, session = heading
                    return f"{subj} run {str(session).zfill(2)}"
                else:
                    return ' '.join(str(h) for h in heading)
            else:
                return str(heading)

        for heading, value in self.traj_transitions_dict_original.items():
            fig, ax = plt.subplots()
            time_array = np.arange(0, time_jump*value.shape[0], time_jump)

            # Plot feeling trajectories
            for feeling in self.feelings:
                ax.plot(time_array, value[feeling]*10, 
                       label=feeling, color=self.feeling_colors[feeling])

            # Add cluster shading
            if heading in self.traj_transitions_dict:
                traj_group = self.traj_transitions_dict[heading]
                prev_color_val = traj_group['labels'].iloc[0]
                start_index = 0
                for index, color_val in enumerate(traj_group['labels']):
                    if color_val != prev_color_val or index == traj_group.shape[0]-1:
                        end_idx = index if index != traj_group.shape[0]-1 else len(time_array)-1
                        ax.axvspan(time_array[start_index], time_array[end_idx],
                                  facecolor=self.color_map.get(prev_color_val, 'grey'), alpha=0.3)
                        start_index = index
                        prev_color_val = color_val

            # Add annotations
            if heading in self.group_transitions:
                self.annotate_state_durations(ax, time_array, 
                                             self.group_transitions[heading])
            if 'Condition' in value.columns:
                self.annotate_conditions(ax, time_array, value)

            # Finalize plot
            formatted_title = format_heading(heading)
            ax.set_title(formatted_title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Rating')
            
            # Create legend
            cluster_patches = [mpatches.Patch(color=color, label=f'Cluster {cluster}')
                             for cluster, color in self.color_map.items()]
            handles, labels = ax.get_legend_handles_labels()
            handles.extend(cluster_patches)
            labels.extend([f'Cluster {label}' 
                         for label in self.dictionary_clust_labels.values()])
            ax.legend(handles=handles, labels=labels, title='Legend',
                     bbox_to_anchor=(1.05, 1), loc='upper left')

            # Save plot
            combined = ''.join(map(str, heading)).translate(
                {ord(c): None for c in "\\'() "})
            save_path = os.path.join(self.savelocation_TET, 
                                    f'HMM_stable_cluster_centroids{combined}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

    def save_transitions_to_file(self):
        """Save transition metadata to CSV file with per-transition condition frequencies"""
        transitions_list = []
        time_step = 28 * self.no_of_jumps  # Seconds between data points
        window_size = 2
        possible_conditions = self.df_csv_file_original['Condition'].unique()

        for group_key, transitions in self.group_transitions.items():
            if self.has_week:
                subject, week, session = group_key
            else:
                subject, session = group_key
                week = None

            original_group = self.traj_transitions_dict_original.get(group_key, pd.DataFrame())
            if original_group.empty or 'Condition' not in original_group.columns:
                continue

            for transition in transitions:
                start_idx, end_idx, from_state, to_state, trans_type = transition
                start_time = start_idx * time_step
                end_time = (end_idx + 1) * time_step
                duration = end_time - start_time

                # Convert to original data indices
                original_start_idx = max(0, int(start_time // 28))
                original_end_idx = min(len(original_group)-1, int(end_time // 28))

                # Extract conditions in each region
                before_start = max(0, original_start_idx - window_size)
                before_end = original_start_idx - 1
                before_conditions = original_group.iloc[before_start:before_end+1]['Condition'] if before_end >= before_start else pd.Series()

                during_conditions = original_group.iloc[original_start_idx:original_end_idx+1]['Condition']

                after_start = original_end_idx + 1
                after_end = min(len(original_group)-1, original_end_idx + window_size)
                after_conditions = original_group.iloc[after_start:after_end+1]['Condition'] if after_end >= after_start else pd.Series()

                # Calculate frequencies for each condition in each period
                def get_freq(conditions):
                    counts = conditions.value_counts(normalize=True).to_dict()
                    return {cond: counts.get(cond, 0.0) for cond in possible_conditions}

                before_freq = get_freq(before_conditions)
                during_freq = get_freq(during_conditions)
                after_freq = get_freq(after_conditions)

                # Build transition entry with condition frequencies
                transition_entry = {
                    'Subject': subject,
                    'Week': week,
                    'Session': session,
                    'Start Time (s)': start_time,
                    'End Time (s)': end_time,
                    'Duration (s)': duration,
                    'From State': from_state + 1,
                    'To State': to_state + 1,
                    'Transition Type': trans_type,
                }

                # Add condition frequency columns
                for cond in possible_conditions:
                    transition_entry[f'Before {cond} Freq'] = before_freq.get(cond, 0.0)
                    transition_entry[f'During {cond} Freq'] = during_freq.get(cond, 0.0)
                    transition_entry[f'After {cond} Freq'] = after_freq.get(cond, 0.0)

                transitions_list.append(transition_entry)

        # Save to CSV
        if transitions_list:
            df_transitions = pd.DataFrame(transitions_list)
            save_path = os.path.join(self.savelocation_TET, 'transitions_summary.csv')
            df_transitions.to_csv(save_path, index=False)
        else:
            print("No transitions detected for saving")


    def run(self):
        """Execute full visualization pipeline"""
        self.preprocess_data()
        self.plot_trajectories()
        if type(self.group_transitions) != list:
            self.save_transitions_to_file()

# ==================================================================================|
# Analysis Class                                                                    |
# ==================================================================================|

class HMMAnalysis:
    """
    This class performs analysis for the custom HMM clustering method.
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

    def hmm_diagnostics(simulated_labels, hmm_labels, hmm_trans_matrix, true_trans_matrix, gamma_vals=None, emission_means=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.spatial.distance import cdist

        # 1. Empirical transition matrix from simulated labels
        n_states = true_trans_matrix.shape[0]
        sim_trans = np.zeros((n_states, n_states))
        for i in range(len(simulated_labels)-1):
            sim_trans[simulated_labels[i], simulated_labels[i+1]] += 1
        sim_trans = sim_trans / sim_trans.sum(axis=1, keepdims=True)
        print("Empirical transition matrix from simulated data:")
        print(np.round(sim_trans, 2))
        print("True transition matrix:")
        print(np.round(true_trans_matrix, 2))
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.heatmap(sim_trans, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Empirical Simulated Transition Matrix')
        plt.subplot(1,2,2)
        sns.heatmap(true_trans_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('True Transition Matrix')
        plt.tight_layout()
        plt.savefig('transition_matrices_comparison.png')
        plt.close()

        # 2. HMM learned transition matrix
        print("HMM learned transition matrix:")
        print(np.round(hmm_trans_matrix, 2))
        plt.figure(figsize=(5,4))
        sns.heatmap(hmm_trans_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('HMM Learned Transition Matrix')
        plt.tight_layout()
        plt.savefig('hmm_transition_matrix.png')
        plt.close()

        # 3. State occupancy
        print("State occupancy (simulated):", np.bincount(simulated_labels) / len(simulated_labels))
        print("State occupancy (HMM):", np.bincount(hmm_labels) / len(hmm_labels))

        # 4. Average gamma for transition state
        if gamma_vals is not None:
            print("Average gamma (soft assignment) for each state:", np.mean(gamma_vals, axis=0))

        # 5. Emission mean separation
        if emission_means is not None:
            dists = cdist(emission_means, emission_means)
            print("Pairwise distances between emission means:")
            print(np.round(dists, 2))
            min_dist = np.min(dists + np.eye(len(emission_means))*1e6)
            if min_dist < 0.5:
                print("WARNING: Emission means are not well separated! Min distance:", min_dist)

    # Usage example (call after HMM fitting):
    # hmm_diagnostics(sim_states_full, state_seq, model.trans_prob, np.array(transition_matrix), gamma_vals=fs, emission_means=model.emission_means)

