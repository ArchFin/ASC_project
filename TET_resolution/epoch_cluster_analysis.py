import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import os
import yaml
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class EpochClusterAnalysis:
    """
    Analyze cluster distinction under different epoch lengths for EEG data.
    Compares breathwork and meditation conditions to find optimal temporal resolution.
    """
    
    def __init__(self, config_path="Breathwork.yaml"):
        """Initialize with configuration file."""
        self.config_path = config_path
        self.load_config()
        self.results = {}
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default settings.")
            self.config = self._get_default_config()
            
    def _get_default_config(self):
        """Default configuration if YAML file not found."""
        return {
            'filelocation_TET': '/Users/a_fin/Desktop/Year 4/Project/ASC_project/converted_csv/combined_all_subjects_labelled.csv',
            'feelings': ['valence', 'arousal', 'familiarity', 'transcendence', 'positive_mood', 'negative_mood'],
            'no_dimensions_PCA': 3,
            'no_clust': 4,
            'savelocation_TET': '/Users/a_fin/Desktop/Year 4/Project/ASC_project/TET_resolution/'
        }
    
    def load_data(self):
        """Load EEG data from CSV file."""
        try:
            df = pd.read_csv(self.config['filelocation_TET'])
            print(f"Loaded data with shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            print(f"Data file not found: {self.config['filelocation_TET']}")
            return None
            
    def filter_condition(self, df, condition):
        """Filter data for specific condition (breathwork or meditation)."""
        if 'condition' in df.columns:
            return df[df['condition'].str.lower() == condition.lower()].copy()
        elif 'Condition' in df.columns:
            return df[df['Condition'].str.lower() == condition.lower()].copy()
        else:
            print("Warning: No condition column found. Using all data.")
            return df.copy()
    
    def create_epochs(self, df, epoch_length_s, sampling_rate=250):
        """
        Create epochs of specified length from continuous data.
        
        Args:
            df: DataFrame with EEG data
            epoch_length_s: Epoch length in seconds
            sampling_rate: Sampling rate in Hz
        """
        epoch_samples = int(epoch_length_s * sampling_rate)
        epochs = []
        
        # Assuming time column exists or create index-based time
        if 'time' not in df.columns:
            df['time'] = df.index / sampling_rate
            
        # Group by subject if subject column exists
        if 'subject' in df.columns or 'Subject' in df.columns:
            subject_col = 'subject' if 'subject' in df.columns else 'Subject'
            subjects = df[subject_col].unique()
        else:
            subjects = ['all']
            df['subject'] = 'all'
            subject_col = 'subject'
            
        for subject in subjects:
            subject_data = df[df[subject_col] == subject].copy()
            
            # Create epochs
            max_time = subject_data['time'].max()
            epoch_starts = np.arange(0, max_time - epoch_length_s, epoch_length_s)
            
            for start_time in epoch_starts:
                end_time = start_time + epoch_length_s
                epoch_data = subject_data[
                    (subject_data['time'] >= start_time) & 
                    (subject_data['time'] < end_time)
                ].copy()
                
                if len(epoch_data) > 0:
                    # Calculate mean values for feelings in this epoch
                    epoch_summary = {}
                    epoch_summary['subject'] = subject
                    epoch_summary['epoch_start'] = start_time
                    epoch_summary['epoch_length'] = epoch_length_s
                    
                    for feeling in self.config['feelings']:
                        if feeling in epoch_data.columns:
                            epoch_summary[feeling] = epoch_data[feeling].mean()
                    
                    epochs.append(epoch_summary)
        
        return pd.DataFrame(epochs)
    
    def perform_clustering(self, df, n_clusters=None):
        """
        Perform clustering on epoched data.
        
        Args:
            df: DataFrame with epoched data
            n_clusters: Number of clusters (default from config)
        """
        if n_clusters is None:
            n_clusters = self.config['no_clust']
            
        # Extract features for clustering
        features = []
        for feeling in self.config['feelings']:
            if feeling in df.columns:
                features.append(feeling)
        
        if not features:
            print("No valid features found for clustering")
            return None, None, None
            
        X = df[features].values
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(self.config['no_dimensions_PCA'], len(features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        
        return cluster_labels, X_pca, pca
    
    def calculate_cluster_metrics(self, X, labels):
        """Calculate various cluster quality metrics."""
        if len(np.unique(labels)) < 2:
            return {'silhouette': 0, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
            
        try:
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            return {
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin
            }
        except:
            return {'silhouette': 0, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
    
    def analyze_epoch_lengths(self, conditions=['breathwork', 'meditation']):
        """
        Analyze cluster distinction across different epoch lengths.
        
        Args:
            conditions: List of conditions to analyze
        """
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Define epoch lengths to test
        epoch_lengths = {
            'breathwork': np.arange(5, 61, 2),  # 5s to 60s in 2s steps
            'meditation': np.arange(10, 121, 5)  # 10s to 120s in 5s steps
        }
        
        results = {}
        
        for condition in conditions:
            print(f"\nAnalyzing {condition} condition...")
            condition_data = self.filter_condition(df, condition)
            
            if len(condition_data) == 0:
                print(f"No data found for condition: {condition}")
                continue
                
            condition_results = {
                'epoch_lengths': [],
                'silhouette_scores': [],
                'calinski_harabasz_scores': [],
                'davies_bouldin_scores': [],
                'n_epochs': []
            }
            
            for epoch_length in epoch_lengths[condition]:
                print(f"  Processing epoch length: {epoch_length}s")
                
                # Create epochs
                epoched_data = self.create_epochs(condition_data, epoch_length)
                
                if len(epoched_data) < self.config['no_clust']:
                    print(f"    Not enough epochs ({len(epoched_data)}) for clustering")
                    continue
                
                # Perform clustering
                labels, X_pca, pca = self.perform_clustering(epoched_data)
                
                if labels is None:
                    continue
                
                # Calculate metrics
                metrics = self.calculate_cluster_metrics(X_pca, labels)
                
                # Store results
                condition_results['epoch_lengths'].append(epoch_length)
                condition_results['silhouette_scores'].append(metrics['silhouette'])
                condition_results['calinski_harabasz_scores'].append(metrics['calinski_harabasz'])
                condition_results['davies_bouldin_scores'].append(metrics['davies_bouldin'])
                condition_results['n_epochs'].append(len(epoched_data))
                
                print(f"    Epochs: {len(epoched_data)}, Silhouette: {metrics['silhouette']:.3f}")
            
            results[condition] = condition_results
        
        self.results = results
        return results
    
    def find_optimal_epochs(self):
        """Find optimal epoch lengths based on cluster metrics."""
        optimal_epochs = {}
        
        for condition, data in self.results.items():
            if not data['silhouette_scores']:
                continue
                
            # Find epoch length with highest silhouette score
            max_silhouette_idx = np.argmax(data['silhouette_scores'])
            optimal_epoch = data['epoch_lengths'][max_silhouette_idx]
            max_silhouette = data['silhouette_scores'][max_silhouette_idx]
            
            optimal_epochs[condition] = {
                'epoch_length': optimal_epoch,
                'silhouette_score': max_silhouette,
                'calinski_harabasz': data['calinski_harabasz_scores'][max_silhouette_idx],
                'davies_bouldin': data['davies_bouldin_scores'][max_silhouette_idx]
            }
            
            print(f"\nOptimal epoch length for {condition}: {optimal_epoch}s")
            print(f"  Silhouette score: {max_silhouette:.3f}")
        
        return optimal_epochs
    
    def visualize_results(self, save_plots=True):
        """Create visualizations of the analysis results."""
        if not self.results:
            print("No results to visualize. Run analyze_epoch_lengths() first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cluster Quality Metrics vs Epoch Length', fontsize=16, fontweight='bold')
        
        conditions = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Plot 1: Silhouette Score
        ax1 = axes[0, 0]
        for i, condition in enumerate(conditions):
            data = self.results[condition]
            ax1.plot(data['epoch_lengths'], data['silhouette_scores'], 
                    marker='o', linewidth=2, markersize=6, label=condition.capitalize(),
                    color=colors[i])
        ax1.set_xlabel('Epoch Length (seconds)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score vs Epoch Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Calinski-Harabasz Score
        ax2 = axes[0, 1]
        for i, condition in enumerate(conditions):
            data = self.results[condition]
            ax2.plot(data['epoch_lengths'], data['calinski_harabasz_scores'], 
                    marker='s', linewidth=2, markersize=6, label=condition.capitalize(),
                    color=colors[i])
        ax2.set_xlabel('Epoch Length (seconds)')
        ax2.set_ylabel('Calinski-Harabasz Score')
        ax2.set_title('Calinski-Harabasz Score vs Epoch Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Davies-Bouldin Score (lower is better)
        ax3 = axes[1, 0]
        for i, condition in enumerate(conditions):
            data = self.results[condition]
            ax3.plot(data['epoch_lengths'], data['davies_bouldin_scores'], 
                    marker='^', linewidth=2, markersize=6, label=condition.capitalize(),
                    color=colors[i])
        ax3.set_xlabel('Epoch Length (seconds)')
        ax3.set_ylabel('Davies-Bouldin Score')
        ax3.set_title('Davies-Bouldin Score vs Epoch Length\n(Lower is Better)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Number of Epochs
        ax4 = axes[1, 1]
        for i, condition in enumerate(conditions):
            data = self.results[condition]
            ax4.plot(data['epoch_lengths'], data['n_epochs'], 
                    marker='d', linewidth=2, markersize=6, label=condition.capitalize(),
                    color=colors[i])
        ax4.set_xlabel('Epoch Length (seconds)')
        ax4.set_ylabel('Number of Epochs')
        ax4.set_title('Number of Epochs vs Epoch Length')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = os.path.join(self.config['savelocation_TET'], 'epoch_cluster_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Create a second figure for detailed comparison
        self._create_comparison_plot(save_plots)
    
    def _create_comparison_plot(self, save_plots=True):
        """Create a detailed comparison plot highlighting optimal epochs."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        conditions = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, condition in enumerate(conditions):
            data = self.results[condition]
            
            # Plot silhouette scores
            ax.plot(data['epoch_lengths'], data['silhouette_scores'], 
                   marker='o', linewidth=3, markersize=8, label=f'{condition.capitalize()}',
                   color=colors[i], alpha=0.8)
            
            # Highlight optimal epoch
            if data['silhouette_scores']:
                max_idx = np.argmax(data['silhouette_scores'])
                optimal_epoch = data['epoch_lengths'][max_idx]
                optimal_score = data['silhouette_scores'][max_idx]
                
                ax.scatter(optimal_epoch, optimal_score, s=200, color=colors[i], 
                          marker='*', edgecolor='black', linewidth=2, 
                          label=f'{condition.capitalize()} Optimal ({optimal_epoch}s)', zorder=5)
                
                # Add annotation
                ax.annotate(f'{optimal_epoch}s\n({optimal_score:.3f})', 
                           xy=(optimal_epoch, optimal_score), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel('Epoch Length (seconds)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Optimal Epoch Length Comparison\nSilhouette Score vs Epoch Length', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add hypothesis lines
        ax.axvline(x=28, color='blue', linestyle='--', alpha=0.5, 
                  label='Breathwork Hypothesis (28s)')
        ax.axvline(x=60, color='orange', linestyle='--', alpha=0.5, 
                  label='Meditation Hypothesis (60s)')
        
        plt.tight_layout()
        
        if save_plots:
            save_path = os.path.join(self.config['savelocation_TET'], 'optimal_epoch_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self):
        """Save analysis results to CSV files."""
        if not self.results:
            print("No results to save.")
            return
        
        for condition, data in self.results.items():
            df_results = pd.DataFrame({
                'epoch_length': data['epoch_lengths'],
                'silhouette_score': data['silhouette_scores'],
                'calinski_harabasz_score': data['calinski_harabasz_scores'],
                'davies_bouldin_score': data['davies_bouldin_scores'],
                'n_epochs': data['n_epochs']
            })
            
            save_path = os.path.join(self.config['savelocation_TET'], 
                                   f'epoch_analysis_{condition}.csv')
            df_results.to_csv(save_path, index=False)
            print(f"Results for {condition} saved to: {save_path}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Epoch-Cluster Analysis...")
        print("=" * 50)
        
        # Run analysis
        results = self.analyze_epoch_lengths()
        
        if not results:
            print("Analysis failed. Check data and configuration.")
            return
        
        # Find optimal epochs
        optimal_epochs = self.find_optimal_epochs()
        
        # Create visualizations
        self.visualize_results()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 50)
        print("Analysis Complete!")
        
        # Print summary
        print("\nSUMMARY:")
        for condition, optimal in optimal_epochs.items():
            hypothesis = 28 if condition == 'breathwork' else 60
            print(f"\n{condition.upper()}:")
            print(f"  Optimal epoch length: {optimal['epoch_length']}s")
            print(f"  Hypothesis: {hypothesis}s")
            print(f"  Difference: {abs(optimal['epoch_length'] - hypothesis)}s")
            print(f"  Silhouette score: {optimal['silhouette_score']:.3f}")
        
        return optimal_epochs


def main():
    """Main function to run the analysis."""
    # Initialize analyzer
    analyzer = EpochClusterAnalysis()
    
    # Run full analysis
    optimal_epochs = analyzer.run_full_analysis()
    
    return analyzer, optimal_epochs


if __name__ == "__main__":
    analyzer, optimal_epochs = main()