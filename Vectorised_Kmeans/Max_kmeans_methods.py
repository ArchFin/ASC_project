from sklearn.cluster import KMeans
from itertools import combinations
import pandas as pd
from scipy.spatial import distance
import os  # Added for path handling
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.patches as mpatches
from statsmodels.tsa.stattools import acf
import yaml
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler  # Added for standardization

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
           
# Finds and plots the principal components for all of the TET data

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
        df_TET_feelings_prin = pd.DataFrame(self.scaled_data @ self.principal_components.T,  # <-- KEY FIX
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


class KMeansVectorClustering:
    """
    A class to perform K-Means clustering on vectorized TET data.
    """

    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original, feelings, feelings_diffs, principal_components, no_of_jumps, colours, colours_list):
        """
        Initialise the KMeansVectorClustering class with data and parameters.

        Args:
            filelocation_TET (str): File path for TET data.
            savelocation_TET (str): Save location for output files.
            df_csv_file_original (DataFrame): Original input dataset.
            feelings (list): List of emotional states to analyse.
            feelings_diffs (list): Differences of emotional states.
            principal_components (array): Principal component transformation matrix.
            no_of_jumps (int): Step size for downsampling.
            colours (dict): Colour mapping for clusters.
            colours_list (list): List of colour values.
        """
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.df_csv_file_original = df_csv_file_original
        self.feelings = feelings
        self.feelings_diffs = feelings_diffs
        self.principal_components = principal_components
        self.no_of_jumps = no_of_jumps
        self.colours = colours
        self.colours_list = colours_list

    def preprocess_data(self):
        """
        Preprocess the data by downsampling and computing emotional state differences.
        """
        split_dict_skip = {}

        # Group by subject, week, and session, then downsample
        for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            group = group.iloc[::self.no_of_jumps].copy()

            # Compute the difference in emotional states
            for feeling in self.feelings:
                group[f'{feeling}_diff'] = -group[feeling].diff(-1)

            split_dict_skip[(subject, week, session)] = group

        # Combine processed data
        self.df_csv_file = pd.concat(split_dict_skip.values())

        # Store grouped data for later clustering
        split_dict = {key: group.copy() for key, group in self.df_csv_file.groupby(['Subject', 'Week', 'Session'])}
        self.differences_array = pd.concat([df[:-1] for df in split_dict.values()])
        self.differences_array_MI = self.differences_array.copy()
        self.differences_array_MI['number'] = range(self.differences_array.shape[0])

    def perform_clustering(self):
        """
        Perform K-Means clustering on the processed data.
        """
        wcss_best = float('inf')  # Track the best Within-Cluster Sum of Squares (WCSS)
        self.labels_fin = []
        self.cluster_centres_fin = []

        # Run K-Means 1000 times to find the best clustering
        for _ in range(1000):
            kmeans = KMeans(3)
            kmeans.fit(self.differences_array.iloc[:, -len(self.feelings)::])

            # Update best clustering if WCSS improves
            if kmeans.inertia_ < wcss_best:
                wcss_best = kmeans.inertia_
                self.labels_fin = kmeans.labels_
                self.cluster_centres_fin = kmeans.cluster_centers_

        # Store cluster labels in the dataset
        self.differences_array_MI['labels unnormalised vectors'] = self.labels_fin
        self.point_colours = [self.colours[i] for i in self.labels_fin]

    def plot_results(self):
        """
        Generate a scatter plot of clustered data using principal components.
        """
        # Project data onto principal components
        self.differences_array[["principal component 1", "principal component 2"]] = self.differences_array.iloc[:, -len(self.feelings):].dot(self.principal_components.T)

        # Scatter plot of clustered points
        plt.scatter(self.differences_array["principal component 1"], self.differences_array["principal component 2"], color=self.point_colours, s=0.5)
        plt.xlabel("Principal Component 1 (Bored/Effort)")
        plt.ylabel("Principal Component 2 (Calm)")
        plt.title("K-Means Cluster Scatter Plot")
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + 'K_means_vector_scatter_plot')
        plt.close()

        # Plot cluster centres as vectors
        cluster_centres_prin = np.transpose(self.cluster_centres_fin.dot(self.principal_components.T), (1, 0))
        for i in range(cluster_centres_prin.shape[1]):
            plt.arrow(0, 0, cluster_centres_prin[0, i], cluster_centres_prin[1, i], head_width=0.1, head_length=0.1,
                      fc=self.colours_list[i], ec=self.colours_list[i])
        plt.xlabel("Principal Component 1 (Bored*Effort)")
        plt.ylabel("Principal Component 2 (Calm)")
        plt.legend([f'Cluster {i+1}' for i in range(cluster_centres_prin.shape[1])])
        plt.title("Cluster Centres for K-Means Vector")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + 'cluster_centres_for_kmeans_vectors')
        plt.close()

    def plot_cluster_centroids(self):
        """
        Plot bar charts of cluster centroids to visualise key features.
        """
        for i in range(self.cluster_centres_fin.shape[0]):
            plt.figure()
            plt.bar(self.feelings, self.cluster_centres_fin[i])
            plt.ylim(-0.22, 0.22)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Cluster Centroid for Cluster {i+1}")
            plt.tight_layout()
            plt.savefig(self.savelocation_TET + f'K-Vector_Vector_Cluster_Centroids_{i}')
            plt.close()

    def stable_cluster_analysis(self):
        """
        Perform secondary clustering analysis on the most stable cluster.
        """
        magnitudes = [np.linalg.norm(centre) for centre in self.cluster_centres_fin]
        stable_cluster = np.argmin(magnitudes)
        self.differences_array['clust'] = self.labels_fin
        df_stable = self.differences_array[self.differences_array['clust'] == stable_cluster].copy()

        wcss_best = float('inf')
        labels_fin_stable = []
        cluster_centres_fin_stable = []

        # Perform K-Means within the stable cluster
        for _ in range(1000):
            kmeans = KMeans(2)
            kmeans.fit(df_stable[self.feelings])

            if kmeans.inertia_ < wcss_best:
                wcss_best = kmeans.inertia_
                labels_fin_stable = kmeans.labels_
                cluster_centres_fin_stable = kmeans.cluster_centers_

        # Update the cluster labels
        df_stable = df_stable.drop('clust', axis=1)
        df_stable['clust_name'] = [f'{stable_cluster+1}a' if label == 0 else f'{stable_cluster+1}b' for label in labels_fin_stable]
        df_stable['clust'] = [stable_cluster if label == 0 else 3 for label in labels_fin_stable]
        self.differences_array['clust_name'] = self.differences_array['clust'] + 1
        self.differences_array.update(df_stable)

        # Create a dictionary for cluster labels
        clust_labels = self.differences_array['clust'].unique() + 1
        clust_name_labels = self.differences_array['clust_name'].unique()
        self.dictionary_clust_labels = {clust: clust_name for clust, clust_name in zip(clust_labels, clust_name_labels)}

        # Plot stable cluster centroids
        alphabet = {0: 'a', 1: 'b'}
        for i in range(cluster_centres_fin_stable.shape[0]):
            plt.figure()
            plt.bar(self.feelings, cluster_centres_fin_stable[i])
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.title(f"Cluster Centroid for Stable Cluster {stable_cluster+1}{alphabet[i]}")
            plt.tight_layout()
            plt.savefig(self.savelocation_TET + f'K-Vector_Cluster_Centroids_for_stable_cluster_{i}')
            plt.close()

    def expand_to_original_shape(self):
        """
        Expand the downsampled clustering results back to the original DataFrame's shape.
        This method creates a new column with cluster labels by merging the downsampled labels
        into the original dataset, then uses forward/backward filling to fill the gaps.
        """
        # Copy the original DataFrame
        expanded_df = self.df_csv_file_original.copy()
        # Create a new column for the cluster labels, initializing with NaNs
        expanded_df['cluster_label'] = np.nan
        # Use the downsampled indices to assign the cluster labels
        expanded_df.loc[self.differences_array.index, 'cluster_label'] = self.differences_array['clust']
        # Fill missing values (adjust the fill method if needed)
        expanded_df['cluster_label'] = expanded_df['cluster_label'].fillna(method='ffill').fillna(method='bfill')
        return expanded_df

    def run(self):
        self.preprocess_data()
        self.perform_clustering()
        self.plot_results()
        self.plot_cluster_centroids()
        self.stable_cluster_analysis()
        return self.differences_array, self.dictionary_clust_labels


class KMeansVectorVisualizer:
    def __init__(self, filelocation_TET, savelocation_TET, differences_array, df_csv_file_original, dictionary_clust_labels, principal_components, feelings, no_of_jumps, colours):
        # Initialize class variables with provided parameters
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.differences_array = differences_array
        self.df_csv_file_original = df_csv_file_original
        self.dictionary_clust_labels = dictionary_clust_labels
        self.principal_components = principal_components
        self.feelings = feelings
        self.no_of_jumps = no_of_jumps
        # 1 is 2a, 3 is 2b, 0 is 1 and 2 is 3 this is for clust and clust_name
        # Negative Stable Cluster (green 2b), then Positive Vectoral cluster (blue 1) and Positive Stable (yellow 2a), and ends with Negative Vectoral cluster (red 3).
        # Define a color map for different clusters s18week_1run_01
        self.color_map = colours
        self.feeling_colors = {feeling: self.color_map.get(i, 'black') 
                  for i, feeling in enumerate(self.feelings)}

    def preprocess_data(self):
        # Compute principal component projections for the differences array
        self.differences_array[["principal component 1 non-diff", "principal component 2 non-diff"]] = self.differences_array[self.feelings].dot(self.principal_components.T)
        
        # Initialize dictionaries to store trajectory data
        self.traj_transitions_dict = {}
        self.traj_transitions_dict_original = {}
        
        # Group original CSV data by Subject, Week, and Session
        for heading, group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            self.traj_transitions_dict_original[heading] = group
        
        # Group transformed differences data by Subject, Week, and Session
        for heading, group in self.differences_array.groupby(['Subject', 'Week', 'Session']):
            self.traj_transitions_dict[heading] = group

    def plot_trajectories(self):
        time_jump = 28  # Original data sampling interval (seconds)

        for heading, value in self.traj_transitions_dict_original.items():
            fig, ax = plt.subplots()
            time_array = np.arange(0, time_jump * value.shape[0], time_jump)

            # Plot feeling trajectories
            for feeling in self.feelings:
                ax.plot(time_array, value[feeling] * 10, label=feeling, color=self.feeling_colors[feeling])

            # Add cluster shading if available
            if heading in self.traj_transitions_dict:
                traj_group = self.traj_transitions_dict[heading]
                # Use the original scaling for cluster boundaries:
                # Each index in traj_group corresponds to time = index * (time_jump * self.no_of_jumps)
                prev_color_val = traj_group['clust'].iloc[0]
                start_index = 0
                for index, color_val in enumerate(traj_group['clust']):
                    # Check for a change in cluster or if we are at the last point
                    if color_val != prev_color_val or index == traj_group.shape[0] - 1:
                        # Compute the end time for the shaded region.
                        if index != traj_group.shape[0] - 1:
                            end_time = index * (time_jump * self.no_of_jumps)
                        else:
                            end_time = time_array[-1]
                        start_time = start_index * (time_jump * self.no_of_jumps)
                        ax.axvspan(start_time, end_time, 
                                facecolor=self.color_map.get(prev_color_val, 'grey'), alpha=0.3)
                        start_index = index
                        prev_color_val = color_val

            # Finalize plot appearance
            combined = ''.join(map(str, heading)).translate({ord(c): None for c in "\\'() "})
            ax.set_title(combined)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Rating')
            
            # Create legend for clusters similar to the reference function
            cluster_patches = [mpatches.Patch(color=color, label=f'Cluster {cluster}')
                            for cluster, color in self.color_map.items()]
            handles, labels = ax.get_legend_handles_labels()
            handles.extend(cluster_patches)
            labels.extend([f'Cluster {label}' for label in self.dictionary_clust_labels.values()])
            ax.legend(handles=handles, labels=labels, title='Legend', 
                    bbox_to_anchor=(1.05, 1), loc='upper left')

            # Save the plot
            save_path = os.path.join(self.savelocation_TET, f'K_Vector_stable_cluster_centroids{combined}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        
    def run(self):
        # Run the preprocessing and visualization methods
        self.preprocess_data()
        self.plot_trajectories()


class JumpAnalysis:
    # Constructor to initialize the class with necessary variables.
    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original, feelings, feelings_diffs):
        self.filelocation_TET = filelocation_TET  # Path where TET files are stored
        self.savelocation_TET = savelocation_TET  # Path to save results
        self.df_csv_file_original = df_csv_file_original  # Original dataframe
        self.feelings = feelings  # List of feelings columns in the dataframe
        self.feelings_diffs = feelings_diffs  # List of differences for feelings columns

    # Method to analyse stability of clusters based on the number of time steps (jumps)
    def determine_no_jumps_stability(self):
        y_labels = []  # Stores the ratio of the dominant stable cluster
        x_labels = []  # Stores the corresponding time steps (jumps)
        
        # Loop through different values of time steps (from 1 to 29)
        for j in range(1, 30):
            split_dict_skip = {}  # Dictionary to store grouped data with skips (time steps)
            
            # Group data by subject, week, and session
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                group = group.iloc[::j].copy()  # Skip rows by j steps
                # Calculate the differences for each feeling
                for feeling in self.feelings:
                    group[f'{feeling}_diff'] = -group[feeling].diff(-1)
                split_dict_skip[(subject, week, session)] = group

            df_csv_file_new = pd.concat([df for df in split_dict_skip.values()])  # Concatenate the split data
            
            # Group data again by subject, week, and session
            split_dict = {}
            for (subject, week, session), group in df_csv_file_new.groupby(['Subject', 'Week', 'Session']):
                split_dict[(subject, week, session)] = group.copy()
                
            differences_array = pd.concat([df[:-1] for df in split_dict.values()])  # Concatenate data after skipping last row
            differences_array_MI = differences_array.copy()
            differences_array_MI['number'] = range(differences_array.shape[0])  # Add a sequence number column

            wcss_best = float('inf')  # Initialise to store the best WCSS (Within-Cluster Sum of Squares)
            labels_fin = []  # Store final cluster labels
            cluster_centres_fin = []  # Store final cluster centers
            
            # Run k-means clustering 1000 times to get the best clustering
            for _ in range(1000):
                kmeans = KMeans(3)  # Use 3 clusters for k-means
                kmeans.fit(differences_array.iloc[:, -len(self.feelings):])  # Fit using the last 14 columns (assumed to be features)
                if kmeans.inertia_ < wcss_best:  # Update if a better (lower) WCSS is found
                    wcss_best = kmeans.inertia_
                    labels_fin = kmeans.labels_
                    cluster_centres_fin = kmeans.cluster_centers_

            differences_array_MI['labels unnormalised vectors'] = labels_fin  # Add labels to data
            unique, counts = np.unique(labels_fin, return_counts=True)  # Get counts of each label
            label_counts = dict(zip(unique, counts))
            print(f'For {j} time steps, our cluster distribution is {label_counts}')  # Print cluster distribution

            # Check if the cluster with the highest count is the smallest in magnitude
            magnitudes = [np.linalg.norm(centre) for centre in cluster_centres_fin]
            max_clust = [key for key, value in label_counts.items() if value == max(counts)]
            if magnitudes[max_clust[0]] == min(magnitudes):
                print("True")
            else:
                print('False')

            # Store the ratio of the most frequent cluster to the sum of all other clusters
            y_labels.append(max(counts) / (sum(counts) - max(counts)))
            x_labels.append(j)  # Store the number of time jumps

        # Plot the relationship between the number of time jumps and the stable cluster ratio
        plt.plot(x_labels, y_labels)
        plt.title('Stable Cluster Dominance with No. of Time Steps')
        plt.xlabel('Number of Time Jumps')
        plt.ylabel('No in stable cluster:No in all other clusters')
        plt.savefig(self.savelocation_TET + f'KMeans_stable_cluster_dominance')  # Save the plot
        # plt.show()  # Show the plot

    # Method to analyse consistency of clusters based on the number of time steps
    def determine_no_jumps_consistency(self):
        x_values = []  # Stores the number of time steps (jumps)
        y_values = []  # Stores the consistency (correct assignment ratio)
        
        # Loop through different values of time steps (from 1 to 29)
        for no_of_jumps in range(1, 30):
            split_dict_skip = {}  # Dictionary to store grouped data with skips (time steps)
            
            # Group data by subject, week, and session
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                group = group.iloc[::no_of_jumps].copy()  # Skip rows by no_of_jumps steps
                # Calculate the differences for each feeling
                for feeling in self.feelings:
                    group[f'{feeling}_diff'] = -group[feeling].diff(-1)
                split_dict_skip[(subject, week, session)] = group

            df_csv_file_new = pd.concat([df for df in split_dict_skip.values()])  # Concatenate the split data
            
            # Group data again by subject, week, and session
            split_dict = {}
            for (subject, week, session), group in df_csv_file_new.groupby(['Subject', 'Week', 'Session']):
                split_dict[(subject, week, session)] = group.copy()

            differences_array = pd.concat([df[:-1] for df in split_dict.values()])  # Concatenate data after skipping last row
            differences_array_MI = differences_array.copy()
            differences_array_MI['number'] = range(differences_array.shape[0])  # Add a sequence number column

            wcss_best = float('inf')  # Initialise to store the best WCSS (Within-Cluster Sum of Squares)
            labels_fin = []  # Store final cluster labels
            cluster_centres_fin = []  # Store final cluster centers
            
            # Run k-means clustering 1000 times to get the best clustering
            for _ in range(1000):
                kmeans = KMeans(3)  # Use 3 clusters for k-means
                kmeans.fit(differences_array.iloc[:, -14:])  # Fit using the last 14 columns (assumed to be features)
                if kmeans.inertia_ < wcss_best:  # Update if a better (lower) WCSS is found
                    wcss_best = kmeans.inertia_
                    labels_fin = kmeans.labels_
                    cluster_centres_fin = kmeans.cluster_centers_

            differences_array_MI['labels unnormalised vectors'] = labels_fin  # Add labels to data
            downsampled_groups = []  # List to store downsampled data groups
            
            # Downsample each group based on no_of_jumps and store their original indices
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                downsampled = group.iloc[::no_of_jumps].copy()
                downsampled = downsampled[:-1]  # Remove last row to match lengths
                downsampled['Original_Index'] = downsampled.index
                downsampled_groups.append(downsampled)

            df_downsampled = pd.concat(downsampled_groups)  # Concatenate downsampled groups
            df_downsampled['Cluster_Label'] = labels_fin  # Add cluster labels to downsampled data
            self.df_csv_file_original['Cluster_Label'] = np.nan  # Initialise cluster label column in original data
            
            # Propagate the cluster labels back to the original dataframe
            for _, row in df_downsampled.iterrows():
                original_index = row['Original_Index']
                label = row['Cluster_Label']
                group_info = self.df_csv_file_original.loc[original_index, ['Subject', 'Week', 'Session']]
                group_mask = (self.df_csv_file_original['Subject'] == group_info['Subject']) & \
                             (self.df_csv_file_original['Week'] == group_info['Week']) & \
                             (self.df_csv_file_original['Session'] == group_info['Session'])
                group_indices = self.df_csv_file_original[group_mask].index
                pos_in_group = list(group_indices).index(original_index)
                start_idx = pos_in_group - (pos_in_group % no_of_jumps)
                end_idx = min(start_idx + no_of_jumps, len(group_indices))
                for i in range(start_idx, end_idx):
                    self.df_csv_file_original.at[group_indices[i], 'Cluster_Label'] = label

            # Calculate the feeling differences for each feeling
            for feeling in self.feelings:
                self.df_csv_file_original[f'{feeling}_diff'] = -self.df_csv_file_original[feeling].diff(-1)

            # Calculate Hughes' measure for cluster consistency
            n_entries = 0
            correct_assignments = 0
            for i, row in self.df_csv_file_original.iterrows():
                if not pd.isnull(row['Cluster_Label']) and not row[self.feelings_diffs].isnull().any():
                    entry = row[self.feelings_diffs] * no_of_jumps  # Get the entry after scaling by jumps
                    assigned_cluster = row['Cluster_Label']  # Get the assigned cluster
                    distances = np.array([euclidean(entry, centre) for centre in cluster_centres_fin])  # Calculate distances to cluster centers
                    closest_centre_idx = np.argmin(distances)  # Get the index of the closest centre
                    if closest_centre_idx == assigned_cluster:  # If the assigned cluster is correct
                        correct_assignments += 1
                    n_entries += 1

            if n_entries > 0:
                hughes_measure = correct_assignments / n_entries  # Calculate Hughes' measure
            y_values.append(hughes_measure)  # Store Hughes' measure
            x_values.append(no_of_jumps)  # Store the number of time steps (jumps)

        # Plot the relationship between the number of time steps and the Hughes' measure
        plt.plot(x_values, y_values)
        plt.title('Cluster Vectoral Consistency')
        plt.xlabel('Number of Time Steps')
        plt.ylabel('Correct Assignment Ratio')
        plt.savefig(self.savelocation_TET + f'K_Vector_Cluster_Vectoral_consistency')  # Save the plot
        # plt.show()  # Show the plot

    # Method to analyse the autocorrelation of feelings over time
    def determine_no_of_jumps_autocorrelation(self):
        split_dict = {}  # Dictionary to store grouped data by subject, week, and session
        
        # Group data by subject, week, and session
        for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            split_dict[(subject, week, session)] = group[self.feelings].copy()

        acf_results = {feeling: [] for feeling in self.feelings}  # Store autocorrelation results for each feeling
        n_lags = 30  # Number of lags for autocorrelation

        # Calculate autocorrelation for each feeling
        for key, df in split_dict.items():
            for feeling in self.feelings:
                acf_value = acf(df[feeling], nlags=n_lags, fft=True)  # Calculate autocorrelation for the feeling
                acf_results[feeling].append(acf_value)

        # Calculate average autocorrelation for each feeling
        acf_averages = {feeling: np.mean(np.vstack(acf_results[feeling]), axis=0) for feeling in self.feelings}
        
        # Plot the autocorrelation for each feeling
        plt.figure(figsize=(12, 8))
        for feeling, acf_vals in acf_averages.items():
            plt.plot(acf_vals, label=feeling)

        plt.title('Average Autocorrelation Function for Each Feeling')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend(title='Feeling', bbox_to_anchor=(1.05, 1), loc='upper left')  # Show legend
        plt.grid(True)  # Show grid
        plt.tight_layout()  # Ensure tight layout for the plot
        plt.savefig(self.savelocation_TET + f'K-Vector autocorrelation for each feeling')  # Save the plot
        # plt.show()  # Show the plot