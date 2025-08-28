import os
import yaml
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from HMM_methods import csv_splitter, principal_component_finder, CustomHMMClustering

class ClusteringComparison:
    def __init__(self, config_file="Breathwork.yaml"):
        # Load configuration
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Read data
        csv_splitter_instance = csv_splitter(self.config['filelocation_TET'])
        self.df_csv_file_original = csv_splitter_instance.read_CSV()
        if self.df_csv_file_original is None:
            raise ValueError("CSV file could not be read. Check the file path and try again.")
        
        self.feelings = self.config['feelings']
        self.no_of_jumps = self.config['no_of_jumps']
        
        # Extract principal components
        principal_component_finder_instance = principal_component_finder(
            self.df_csv_file_original, self.feelings,
            self.config['no_dimensions_PCA'],
            self.config['savelocation_TET']
        )
        self.principal_components, _, _ = principal_component_finder_instance.PCA_TOT()
        
        # Prepare results storage
        self.results = {}
        
    def preprocess_data(self):
        """Preprocess data similar to CustomHMMClustering"""
        if 'Week' in self.df_csv_file_original.columns and 'Med_type' in self.df_csv_file_original.columns:
            group_keys = ['Subject', 'Week', 'Session', 'Med_type']
        elif 'Week' in self.df_csv_file_original.columns:
            group_keys = ['Subject', 'Week', 'Session']
        elif 'Med_type' in self.df_csv_file_original.columns:
            group_keys = ['Subject', 'Session', 'Med_type']
        else:
            group_keys = ['Subject', 'Session']
            
        split_dict_skip = {}
        for keys, group in self.df_csv_file_original.groupby(group_keys):
            group = group.iloc[::self.no_of_jumps].copy()
            split_dict_skip[keys] = group
        
        self.df_processed = pd.concat(split_dict_skip.values())
        self.df_processed['number'] = range(self.df_processed.shape[0])
        return self.df_processed
    
    def run_kmeans_clustering(self, n_clusters=3):
        """Run K-means clustering"""
        print("Running K-means clustering...")
        
        # Prepare data
        data = self.df_processed[self.feelings].values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        
        # Create result dataframe
        result_df = self.df_processed.copy()
        result_df['labels'] = labels
        
        # Calculate cluster centers in original space
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Store results
        self.results['kmeans'] = {
            'data': result_df,
            'cluster_centers': cluster_centers,
            'labels': labels,
            'method': 'K-means'
        }
        
        print(f"K-means completed. Found {n_clusters} clusters.")
        return result_df
    
    def run_standard_hmm(self, n_components=3):
        """Run standard HMM clustering using hmmlearn"""
        print("Running standard HMM clustering...")
        
        # Prepare data
        data = self.df_processed[self.feelings].values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Group data by sessions for proper HMM training
        if 'Week' in self.df_processed.columns and 'Med_type' in self.df_processed.columns:
            group_keys = ['Subject', 'Week', 'Session', 'Med_type']
        elif 'Week' in self.df_processed.columns:
            group_keys = ['Subject', 'Week', 'Session']
        elif 'Med_type' in self.df_processed.columns:
            group_keys = ['Subject', 'Session', 'Med_type']
        else:
            group_keys = ['Subject', 'Session']
        
        # Prepare sequences and lengths
        sequences = []
        lengths = []
        
        for keys, group in self.df_processed.groupby(group_keys):
            group_data = group[self.feelings].values
            group_scaled = scaler.transform(group_data)
            sequences.append(group_scaled)
            lengths.append(len(group_scaled))
        
        # Concatenate all sequences
        X = np.vstack(sequences)
        
        # Create and train HMM
        try:
            model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=42)
            model.fit(X, lengths)
            
            # Predict states for each sequence
            all_labels = []
            start_idx = 0
            
            for length in lengths:
                seq = X[start_idx:start_idx + length]
                states = model.predict(seq)
                all_labels.extend(states)
                start_idx += length
            
            # Create result dataframe
            result_df = self.df_processed.copy()
            result_df['labels'] = all_labels
            
            # Calculate cluster centers from means
            cluster_centers = scaler.inverse_transform(model.means_)
            
            # Store results
            self.results['standard_hmm'] = {
                'data': result_df,
                'cluster_centers': cluster_centers,
                'labels': all_labels,
                'method': 'Standard HMM',
                'model': model
            }
            
            print(f"Standard HMM completed. Found {n_components} states.")
            return result_df
            
        except Exception as e:
            print(f"Standard HMM failed: {e}")
            # Create dummy results with K-means as fallback
            kmeans = KMeans(n_clusters=n_components, random_state=42)
            labels = kmeans.fit_predict(data_scaled)
            
            result_df = self.df_processed.copy()
            result_df['labels'] = labels
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            self.results['standard_hmm'] = {
                'data': result_df,
                'cluster_centers': cluster_centers,
                'labels': labels,
                'method': 'Standard HMM (fallback)',
                'model': None
            }
            
            print("Standard HMM failed, using K-means as fallback.")
            return result_df
    
    def run_custom_hmm(self, num_base_states=2):
        """Run custom HMM clustering"""
        print("Running custom HMM clustering...")
        
        # Calculate smoothness for parameter selection
        smoothness = CustomHMMClustering.calculate_smoothness(self.df_csv_file_original, self.feelings)
        
        # Try to load optimal parameters
        realistic_params_path = self.config.get('hyperparameters', {}).get('filelocation_smoothness', None)
        
        if realistic_params_path and os.path.exists(realistic_params_path):
            optimal_params = CustomHMMClustering.get_optimal_params_for_smoothness(
                smoothness, realistic_params_path
            )
        else:
            optimal_params = {
                'gamma_threshold': 0.8,
                'min_nu': 29,
                'transition_contribution': 5,
                'selection_method': 'default'
            }
        
        # Create and run custom HMM clustering
        clustering = CustomHMMClustering(
            self.config['filelocation_TET'], 
            self.config['savelocation_TET'],
            self.df_csv_file_original, 
            self.feelings, 
            self.principal_components, 
            self.no_of_jumps, 
            optimal_params['transition_contribution']
        )
        
        result_df, _, _, _ = clustering.run(
            num_base_states=num_base_states,
            num_iterations=30,
            num_repetitions=1,
            gamma_threshold=optimal_params['gamma_threshold'],
            min_nu=optimal_params['min_nu']
        )
        
        # Store results
        self.results['custom_hmm'] = {
            'data': result_df,
            'cluster_centers': clustering.cluster_centres_fin,
            'labels': result_df['labels'].values,
            'method': 'Custom HMM',
            'clustering_object': clustering
        }
        
        print(f"Custom HMM completed. Found {num_base_states + 1} states.")
        return result_df
    
    def run_all_methods(self, n_clusters=2):
        """Run all clustering methods"""
        # Preprocess data
        self.preprocess_data()
        
        # Run all methods
        self.run_kmeans_clustering(n_clusters)
        self.run_standard_hmm(n_clusters)
        self.run_custom_hmm(n_clusters)  # Custom HMM adds one extra state
        
        return self.results
    
    def save_results(self, save_path='/Users/a_fin/Desktop/Year 4/Project/Data/clustering_comparison_results.pkl'):
        """Save all results to file"""
        # Remove non-serializable objects
        save_results = {}
        for method, result in self.results.items():
            save_results[method] = {
                'data': result['data'],
                'cluster_centers': result['cluster_centers'],
                'labels': result['labels'],
                'method': result['method']
            }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_results, f)
        
        print(f"Results saved to {save_path}")
        return save_path

if __name__ == "__main__":
    # Run comparison
    comparison = ClusteringComparison()
    results = comparison.run_all_methods(n_clusters=3)
    
    # Save results
    comparison.save_results()
    
    print("\nClustering comparison completed!")
    for method, result in results.items():
        print(f"{result['method']}: {len(np.unique(result['labels']))} clusters")