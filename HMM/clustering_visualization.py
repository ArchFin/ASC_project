import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from scipy.optimize import linear_sum_assignment
from clustering_comparison import ClusteringComparison

class ClusteringVisualization:
    def __init__(self, config_file="Meditation.yaml", results_path=None):
        # Load configuration
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.feelings = self.config['feelings']
        self.no_of_jumps = self.config['no_of_jumps']
        self.colors = self.config['colours']
        self.savelocation = self.config['savelocation_TET']
        self.cluster_no = self.config['no_clust']

        # Load results
        if results_path and os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                self.results = pickle.load(f)
        else:
            print("No results file found. Running clustering comparison...")
            comparison = ClusteringComparison(config_file)
            self.results = comparison.run_all_methods(n_clusters=self.cluster_no)
            comparison.save_results()
        
        # Load original data for trajectory plotting
        from HMM_methods import csv_splitter
        csv_splitter_instance = csv_splitter(self.config['filelocation_TET'])
        self.df_csv_file_original = csv_splitter_instance.read_CSV()
        
        # Create color mapping for feelings
        self.feeling_colors = {feeling: self.colors.get(i, 'black') 
                              for i, feeling in enumerate(self.feelings)}
        
        # Apply cluster matching using Hungarian algorithm
        self.align_clusters()
    
    def align_clusters(self):
        """
        Align cluster labels across methods using Hungarian algorithm.
        Uses custom HMM as reference and matches other methods to it.
        """
        if 'custom_hmm' not in self.results:
            print("Warning: custom_hmm results not found. Skipping cluster alignment.")
            return
        
        # Use custom HMM as reference
        reference_centers = self.results['custom_hmm']['cluster_centers']
        reference_method = 'custom_hmm'
        
        print(f"Aligning clusters using {reference_method} as reference...")
        
        for method_name in self.results.keys():
            if method_name == reference_method:
                continue
                
            print(f"Aligning {method_name} clusters...")
            target_centers = self.results[method_name]['cluster_centers']
            
            # Handle different number of clusters (custom_hmm has extra transitional state)
            if method_name == 'kmeans' or method_name == 'standard_hmm':
                # If custom_hmm has extra cluster, exclude last cluster for matching
                if reference_centers.shape[0] > target_centers.shape[0]:
                    base_ref_centers = reference_centers[:-1]  # Exclude last cluster (transitional)
                else:
                    base_ref_centers = reference_centers
                perm = self.match_clusters(base_ref_centers, target_centers)
                # Create full permutation for all clusters
                full_perm = perm
                new_centers = target_centers[perm]
            else:
                # Standard matching for methods with same number of clusters
                perm = self.match_clusters(reference_centers, target_centers)
                full_perm = perm
                new_centers = target_centers[perm]
            
            # Update cluster centers
            self.results[method_name]['cluster_centers'] = new_centers
            
            # Remap labels in the data
            old_labels = self.results[method_name]['labels']
            new_labels = self.remap_labels(old_labels, full_perm)
            self.results[method_name]['labels'] = new_labels
            
            # Update labels in the dataframe
            self.results[method_name]['data']['labels'] = new_labels
            
            print(f"  Applied permutation: {full_perm}")
    
    def match_clusters(self, reference_centers, target_centers):
        """
        Match clusters using Hungarian algorithm based on centroid distances.
        
        Args:
            reference_centers: Reference cluster centers (n_ref_clusters, n_features)
            target_centers: Target cluster centers to be matched (n_target_clusters, n_features)
        
        Returns:
            permutation: Array indicating how to reorder target clusters
        """
        # Compute pairwise distances between cluster centers
        n_ref = reference_centers.shape[0]
        n_target = target_centers.shape[0]
        
        # Use Euclidean distance
        distance_matrix = np.zeros((n_ref, n_target))
        for i in range(n_ref):
            for j in range(n_target):
                distance_matrix[i, j] = np.linalg.norm(reference_centers[i] - target_centers[j])
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        return col_ind
    
    def remap_labels(self, labels, permutation):
        """
        Remap cluster labels according to permutation.
        
        Args:
            labels: Original labels array
            permutation: Permutation array from match_clusters
        
        Returns:
            new_labels: Remapped labels array
        """
        # Create mapping from old to new labels
        label_mapping = {}
        for new_idx, old_idx in enumerate(permutation):
            label_mapping[old_idx] = new_idx
        
        # Apply mapping
        new_labels = np.array([label_mapping.get(label, label) for label in labels])
        return new_labels
    
    def preprocess_trajectory_data(self):
        """Group original data for trajectory plotting"""
        if 'Week' in self.df_csv_file_original.columns and 'Med_type' in self.df_csv_file_original.columns:
            group_keys = ['Subject', 'Week', 'Session', 'Med_type']
        elif 'Week' in self.df_csv_file_original.columns:
            group_keys = ['Subject', 'Week', 'Session']
        elif 'Med_type' in self.df_csv_file_original.columns:
            group_keys = ['Subject', 'Session', 'Med_type']
        else:
            group_keys = ['Subject', 'Session']
        
        self.trajectory_groups_original = {}
        for heading, group in self.df_csv_file_original.groupby(group_keys):
            self.trajectory_groups_original[heading] = group
        
        # Group processed data for each method
        self.trajectory_groups_processed = {}
        for method_name, result in self.results.items():
            self.trajectory_groups_processed[method_name] = {}
            for heading, group in result['data'].groupby(group_keys):
                self.trajectory_groups_processed[method_name][heading] = group
    
    def format_heading(self, heading):
        """Format group heading for display"""
        if isinstance(heading, tuple):
            if len(heading) == 3:
                subj, week, session = heading
                return f"Subject {subj} {week} {str(session).zfill(2)}"
            elif len(heading) == 2:
                subj, session = heading
                return f"Subject {subj} run {str(session).zfill(2)}"
            else:
                return ' '.join(str(h) for h in heading)
        else:
            return str(heading)

    def plot_comparison_trajectories(self):
        """Plot trajectory comparisons for multiple sessions"""
        self.preprocess_trajectory_data()
        
        time_jump = 60  # Original data sampling interval (seconds)
        methods = ['kmeans', 'standard_hmm', 'custom_hmm']
        method_titles = ['K-means', 'Standard HMM', 'TS-HMM']
        
        # Select first few sessions for visualization
        session_keys = list(self.trajectory_groups_original.keys())
        
        for session_key in session_keys:
            if session_key not in self.trajectory_groups_original:
                continue
                
            original_data = self.trajectory_groups_original[session_key]
            time_array = np.arange(0, time_jump * original_data.shape[0], time_jump)
            
            # Create figure with subplots for each method
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Clustering Comparison: {self.format_heading(session_key)}', fontsize=14, fontweight='bold')
            
            for method_idx, (method_name, method_title) in enumerate(zip(methods, method_titles)):
                ax = axes[method_idx]
                
                # Plot feeling trajectories
                for feeling in self.feelings:
                    ax.plot(time_array, original_data[feeling] * 10, 
                           label=feeling, color=self.feeling_colors[feeling], linewidth=1.5)
                
                # Add cluster shading if method data exists
                cluster_patches = []
                if (method_name in self.trajectory_groups_processed and 
                    session_key in self.trajectory_groups_processed[method_name]):
                    
                    processed_data = self.trajectory_groups_processed[method_name][session_key]
                    
                    # Create color mapping for this method's clusters
                    unique_labels = sorted(processed_data['labels'].unique())
                    cluster_colors = {}
                    for i, label in enumerate(unique_labels):
                        if isinstance(self.colors, dict):
                            cluster_colors[label] = self.colors.get(label, plt.cm.tab10(i))
                        else:
                            cluster_colors[label] = plt.cm.tab10(i)
                        # Add cluster color patch for legend
                        cluster_patches.append(mpatches.Patch(color=cluster_colors[label], label=f'Cluster {label+1}'))
                    
                    # Add cluster shading
                    prev_label = processed_data['labels'].iloc[0]
                    start_index = 0
                    
                    for index, label in enumerate(processed_data['labels']):
                        if label != prev_label or index == len(processed_data) - 1:
                            end_idx = index if index != len(processed_data) - 1 else len(time_array) - 1
                            if end_idx < len(time_array):
                                ax.axvspan(time_array[start_index], time_array[end_idx],
                                          facecolor=cluster_colors.get(prev_label, 'grey'), 
                                          alpha=0.3)
                            start_index = index
                            prev_label = label
                
                # Formatting
                ax.set_title(f'{method_title}', fontweight='bold')
                ax.set_xlabel('Time (s)')
                if method_idx == 0:
                    ax.set_ylabel('Rating')
                
                # Add legend for each subplot: feelings + clusters
                handles, labels = ax.get_legend_handles_labels()
                # Only add cluster_patches if there are clusters for this method
                if cluster_patches:
                    handles.extend(cluster_patches)
                ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            session_name = ''.join(map(str, session_key)).translate({ord(c): None for c in "\\'() "})
            save_path = os.path.join(self.savelocation, f'clustering_comparison_{session_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparison plot: {save_path}")
    
    def plot_cluster_centers_comparison(self):
        """Plot cluster centers comparison across methods"""
        methods = ['kmeans', 'standard_hmm', 'custom_hmm']
        method_titles = ['K-means', 'Standard HMM', 'TS-HMM']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Cluster Centers Comparison', fontsize=16, fontweight='bold')
        
        for method_idx, (method_name, method_title) in enumerate(zip(methods, method_titles)):
            ax = axes[method_idx]
            
            if method_name in self.results:
                cluster_centers = self.results[method_name]['cluster_centers']
                n_clusters = cluster_centers.shape[0]
                
                # Create bar plot for each cluster
                x_pos = np.arange(len(self.feelings))
                bar_width = 0.8 / n_clusters
                
                for cluster_idx in range(n_clusters):
                    offset = (cluster_idx - n_clusters/2 + 0.5) * bar_width
                    color = plt.cm.tab10(cluster_idx)
                    
                    ax.bar(x_pos + offset, cluster_centers[cluster_idx], 
                          bar_width, label=f'Cluster {cluster_idx + 1}', 
                          color=color, alpha=0.7)
                
                ax.set_title(method_title, fontweight='bold')
                ax.set_xlabel('Features')
                if method_idx == 0:
                    ax.set_ylabel('Mean Value')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(self.feelings, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.savelocation, 'cluster_centers_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved cluster centers comparison: {save_path}")
    
    def plot_cluster_statistics(self):
        """Plot cluster distribution statistics"""
        methods = ['kmeans', 'standard_hmm', 'custom_hmm']
        method_titles = ['K-means', 'Standard HMM', 'TS-HMM']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Clustering Statistics Comparison', fontsize=16, fontweight='bold')
        
        for method_idx, (method_name, method_title) in enumerate(zip(methods, method_titles)):
            if method_name not in self.results:
                continue
                
            labels = self.results[method_name]['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Cluster distribution (top row)
            ax_top = axes[0, method_idx]
            bars = ax_top.bar(range(len(unique_labels)), counts, 
                             color=[plt.cm.tab10(i) for i in range(len(unique_labels))],
                             alpha=0.7)
            ax_top.set_title(f'{method_title}\nCluster Distribution')
            ax_top.set_xlabel('Cluster ID')
            ax_top.set_ylabel('Number of Points')
            ax_top.set_xticks(range(len(unique_labels)))
            ax_top.set_xticklabels([f'C{i+1}' for i in unique_labels])
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                           str(count), ha='center', va='bottom')
            
            # Cluster percentages (bottom row)
            ax_bottom = axes[1, method_idx]
            percentages = counts / np.sum(counts) * 100
            wedges, texts, autotexts = ax_bottom.pie(percentages, 
                                                    labels=[f'C{i+1}' for i in unique_labels],
                                                    autopct='%1.1f%%',
                                                    colors=[plt.cm.tab10(i) for i in range(len(unique_labels))])
            ax_bottom.set_title(f'{method_title}\nCluster Percentages')
        
        plt.tight_layout()
        save_path = os.path.join(self.savelocation, 'clustering_statistics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved clustering statistics: {save_path}")
    
    def create_summary_report(self):
        """Create a summary report of all methods"""
        report = []
        report.append("CLUSTERING METHODS COMPARISON REPORT")
        report.append("=" * 50)
        report.append("")
        
        for method_name, result in self.results.items():
            labels = result['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            report.append(f"Method: {result['method']}")
            report.append(f"Number of clusters: {len(unique_labels)}")
            report.append(f"Total data points: {len(labels)}")
            report.append(f"Cluster distribution: {dict(zip(unique_labels, counts))}")
            
            # Calculate cluster balance (coefficient of variation)
            cv = np.std(counts) / np.mean(counts)
            report.append(f"Cluster balance (CV): {cv:.3f}")
            report.append("")
        
        # Save report
        report_path = os.path.join(self.savelocation, 'clustering_comparison_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Saved comparison report: {report_path}")
        return '\n'.join(report)

    def run_full_comparison(self):
        """Run complete visualization comparison"""
        print("Creating clustering comparison visualizations...")
        
        # Plot trajectory comparisons
        self.plot_comparison_trajectories()
        
        # Plot cluster centers comparison
        self.plot_cluster_centers_comparison()
        
        # Plot clustering statistics
        self.plot_cluster_statistics()
        
        # Create summary report
        report = self.create_summary_report()
        
        print("\nComparison visualization completed!")
        print(f"All plots saved to: {self.savelocation}")
        
        return report

if __name__ == "__main__":
    # Create visualization
    visualizer = ClusteringVisualization()
    report = visualizer.run_full_comparison()
    
    print("\n" + report)