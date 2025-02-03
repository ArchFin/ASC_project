from sklearn.cluster import KMeans
from itertools import combinations
import pandas as pd
from scipy.spatial import distance
import os
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
from hmmlearn import hmm

# Load YAML file
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

# Class to handle CSV file operations
class csv_splitter:

    def __init__(self, file_path):
        """
        Constructor to initialize the CSV file location.
        :param excel_file_name: Name of the CSV file to be processed.
        """
        self.file_path = file_path

    def read_CSV(self):
        """
        Reads the CSV file and returns it as a pandas DataFrame.
        :return: DataFrame containing CSV file data.
        """
        try:
            df_CSV_file_name = pd.read_csv(self.file_path)  # Load CSV into DataFrame
            return df_CSV_file_name  # Return the DataFrame
        
        except Exception as e:
            print(f"Error reading Excel file: {e}")  # Print error if file reading fails

    def split_by_header(self, df_CSV_file_name, heading):
        """
        Splits the DataFrame into multiple DataFrames based on unique values in a specified column.
        :param df_CSV_file_name: The DataFrame to be split.
        :param heading: Column name used to split the DataFrame.
        :return: Dictionary of DataFrames and an array containing key-value pairs.
        """
        # Check if the specified heading exists in the DataFrame
        if heading not in df_CSV_file_name.columns:
            print(f"Error: '{heading}' not found in DataFrame columns.")  # Print error if column is missing
            return None  # Return None to indicate failure

        # Get unique values from the specified column
        heading_values = df_CSV_file_name[heading].unique()

        # Create a dictionary where keys are unique values, and values are corresponding filtered DataFrames
        split_df = {value: df_CSV_file_name[df_CSV_file_name[heading] == value] for value in heading_values}

        # Convert dictionary into an array format with key-value pairs
        split_df_array = [[key, value] for key, value in split_df.items()]

        return split_df, split_df_array  # Return both dictionary and array of split DataFrames
    
# Finds and plots the principal components for all of the TET data

class principal_component_finder:

    def __init__(self, csv_file, feelings, no_dimensions, savelocation_TET):
        # Extracts relevant data from the CSV file based on given feelings
        self.csv_file_TET = csv_file[feelings]
        self.feelings = feelings
        # Computes the correlation matrix of the selected data
        corr_matrix = self.csv_file_TET.corr()

        self.savelocation  = savelocation_TET
        
        # Performs PCA on the correlation matrix with the specified number of dimensions
        pca = PCA(n_components=no_dimensions)
        self.principal_components = pca.fit_transform(corr_matrix)
        self.explained_variance_ratio = pca.explained_variance_ratio_
    
    def PCA_TOT(self):
        # Computes the transformed data in PCA space
        df_TET_feelings_prin = self.csv_file_TET.dot(self.principal_components)
        
        # Plots bar charts for each principal component
        for i in range(0, self.principal_components.shape[1]):
            y_values = []
            for j in range(0, len(self.feelings)):
                y_values.append(self.principal_components[j][i])
            
            plt.figure()
            plt.bar(self.feelings, y_values)
            plt.title(f'Principal Component {i+1}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.savelocation+ f'/principal_component{i+1}')
        
        # Scatter plot of the first two principal components
        plt.figure()
        plt.scatter(df_TET_feelings_prin[0], df_TET_feelings_prin[1], s=0.5)
        plt.xlabel('Principal Component 1 (bored/effort)')
        plt.ylabel('Principal Component 2 (calm)')
        plt.title('Plot of all the data points in PCA space')
        plt.xlim(-6, 6)
        plt.ylim(-1, 2)
        
        # Bar chart showing the explained variance ratio of each principal component
        labels = [f'Principal Component {i+1}' for i in range(self.principal_components.shape[1])]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, self.explained_variance_ratio, color='skyblue')

        # Adding title and labels
        plt.title('Explained Variance Ratio of PCA Components')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(rotation=45)  # Rotates labels to prevent overlap
        plt.tight_layout()    
        
        return self.principal_components, self.explained_variance_ratio, df_TET_feelings_prin
    
    def PCA_split(self, split_df_array):
        # Extracts relevant data for each split dataset
        split_df_array_TET = [[split_df_array[i][0], split_df_array[i][1][self.feelings]] for i in range(0, len(split_df_array))]
        split_csv_TET = {split_df_array_TET[i][0]: split_df_array_TET[i][1] for i in range(0, len(split_df_array))}
        
        # Computes the transformed data for each split dataset
        df_TET_feelings_prin_dict = {name: split_csv_TET[name].dot(self.principal_components) for name in split_csv_TET.keys()}
        
        # Plots scatter plots for each split dataset in PCA space
        for key, value in df_TET_feelings_prin_dict.items():
            plt.figure()
            plt.scatter(value[0], value[1], s=0.5)
            plt.title(key)
            plt.xlabel('Principal Component 1 (bored/effort)')
            plt.ylabel('Principal Component 2 (calm)')
            plt.xlim(-6, 6)
            plt.ylim(-1, 2)
            plt.show()
        
        return df_TET_feelings_prin_dict


class HMMClustering:
    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original, feelings, feelings_diffs, principal_components, no_of_jumps, colours, colours_list, ):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.df_csv_file_original = df_csv_file_original
        self.feelings = feelings
        self.feelings_diffs = feelings_diffs
        self.principal_components = principal_components
        self.no_of_jumps = no_of_jumps
        self.colours = colours
        self.colours_list = colours_list
        #self.n_components = n_components  # Number of hidden states

    def preprocess_data(self):
        split_dict_skip = {}
        for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            group = group.iloc[::self.no_of_jumps].copy()
            # for feeling in self.feelings:
            #     group[f'{feeling}_diff'] = -group[feeling].diff(-1)
            split_dict_skip[(subject, week, session)] = group

        self.df_csv_file = pd.concat([df for df in split_dict_skip.values()])
        split_dict = {}
        for (subject, week, session), group in self.df_csv_file.groupby(['Subject', 'Week', 'Session']):
            split_dict[(subject, week, session)] = group.copy()
        self.array = pd.concat([df[:-1] for df in split_dict.values()])
        self.array_MI = self.array.copy()
        self.array_MI['number'] = range(self.array.shape[0])

    def perform_clustering(self, min_states = 2, max_states =20):
        best_score = float("inf")
        best_model = None
        for n in range(min_states, max_states + 1):
            model = hmm.GaussianHMM(n_components=n, covariance_type="diag", n_iter=3000)
            model.fit(self.array.iloc[:, 4:4+len(self.feelings)])
            score = model.bic(self.array.iloc[:, 4:4+len(self.feelings)])  
            if score < best_score:
                best_score = score
                best_model = model
                best_n = n
        
        # Fit the model to the data
        best_model.fit(self.array.iloc[:, 4:4+len(self.feelings)])  
        
        # Predict the hidden states (cluster labels)
        self.array_MI['labels unnormalised vectors'] = best_model.predict(self.array.iloc[:, 4:4+len(self.feelings)])  

        # Save results
        self.labels_fin = self.array_MI['labels unnormalised vectors']
        self.cluster_centres_fin = model.means_  # HMM means represent the centers of the hidden states


    def plot_results(self):
        self.array[["principal componant 1", "principal componant 2"]] = self.array.iloc[:, 4:4+len(self.feelings)].dot(self.principal_components)
        plt.scatter(self.array["principal componant 1"], self.array["principal componant 2"], color=[self.colours[i] for i in self.labels_fin], s=0.5)
        plt.xlabel("principal componant 1 (bored/effort)")
        plt.ylabel("principal componant 2 (calm)")
        plt.title("scatter plot of HMM state assignments for TET data vectors")
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + f'HMM_state_scatter_plot')
        plt.show()

        cluster_centres_prin = np.transpose(self.cluster_centres_fin.dot(self.principal_components), (1, 0))
        for i in range(cluster_centres_prin.shape[1]):
            plt.arrow(0, 0, cluster_centres_prin[0, i], cluster_centres_prin[1, i], head_width=0.1, head_length=0.1, fc=self.colours_list[i], ec=self.colours_list[i])
        plt.xlabel("principal component 1 (bored*effort)")
        plt.ylabel("principal component 2 (calm)")
        plt.legend(['State {}'.format(i+1) for i in range(cluster_centres_prin.shape[1])])
        plt.title("State centers for HMM")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + f'state_centers_for_HMM')
        plt.show()

    def plot_cluster_centroids(self):
        for i in range(self.cluster_centres_fin.shape[0]):
            plt.figure()
            plt.bar(self.feelings, self.cluster_centres_fin[i])
            plt.ylim(-0.22, 0.22)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"state centroid for state {i+1}")
            plt.tight_layout()
            plt.savefig(self.savelocation_TET + f'HMM_State_Centroids_{i}')
            plt.show()

    def stable_cluster_analysis(self):
        magnitudes = [np.linalg.norm(centre) for centre in self.cluster_centres_fin]
        stable_cluster = np.argmin(magnitudes)
        self.array['clust'] = self.labels_fin
        df_stable = self.array[self.array['clust'] == stable_cluster].copy()

        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
        
        # Fit the model to the stable cluster data
        model.fit(df_stable[self.feelings])
        
        # Predict the hidden states (cluster labels)
        labels_fin_stable = model.predict(df_stable[self.feelings])  
        cluster_centres_fin_stable = model.means_

        # Update the cluster labels
        df_stable = df_stable.drop('clust', axis=1)
        df_stable['clust_name'] = [f'{stable_cluster+1}a' if label == 0 else f'{stable_cluster+1}b' for label in labels_fin_stable]
        df_stable['clust'] = [stable_cluster if label == 0 else 3 for label in labels_fin_stable]
        self.array['clust_name'] = self.array['clust'] + 1
        self.array.update(df_stable)

        # Create a dictionary for cluster labels
        clust_labels = self.array['clust'].unique() + 1
        clust_name_labels = self.array['clust_name'].unique()
        self.dictionary_clust_labels = {clust: clust_name for clust, clust_name in zip(clust_labels, clust_name_labels)}

        alphabet = {0: 'a', 1: 'b'}
        # Plot the centroids for stable states
        for i in range(cluster_centres_fin_stable.shape[0]):
            plt.figure()
            plt.bar(self.feelings, cluster_centres_fin_stable[i])
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.title(f"state centroid for state {stable_cluster+1}{alphabet[i]} HMM on stable points")
            plt.tight_layout()
            plt.savefig(self.savelocation_TET + f'HMM_state_Centroids_for_stable_state_{i}')
            plt.show()

        # Plot the stable state centroids on principal components
        cluster_centres_prin_stable = np.transpose(cluster_centres_fin_stable.dot(self.principal_components), (1, 0))
        plt.figure()
        for i in range(cluster_centres_prin_stable.shape[1]):
            state_label = f'State {stable_cluster + 1}{alphabet[i]}'
            plt.scatter(cluster_centres_prin_stable[0, i], cluster_centres_prin_stable[1, i], label=state_label)
        plt.xlabel("Principal Component 1 (Bored*Effort)")
        plt.ylabel("Principal Component 2 (Calm)")
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + f'HMM_stable_state_centroids')
        plt.show()

    def plot_cluster_counts(self):
        self.array['clust_name'] = self.array['clust_name'].astype(str)
        unique, counts = np.unique(self.array['clust_name'], return_counts=True)
        label_counts = dict(zip(unique, counts))
        labels = list(label_counts.keys())
        counts = list(label_counts.values())

        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts)
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation='vertical')
        plt.xlabel('State Labels')
        plt.ylabel('Counts')
        plt.title('Distribution of State Labels')
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + f'HMM_state_counts')
        plt.show()

    def run(self):
        self.preprocess_data()
        self.perform_clustering()
        self.plot_results()
        self.plot_cluster_centroids()
        self.stable_cluster_analysis()
        self.plot_cluster_counts()
        return self.array, self.dictionary_clust_labels

    
class Visualizer:
    def __init__(self, filelocation_TET, savelocation_TET, array, df_csv_file_original, dictionary_clust_labels, principal_components, feelings, no_of_jumps):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.array = array
        self.df_csv_file_original = df_csv_file_original
        self.dictionary_clust_labels = dictionary_clust_labels
        self.principal_components = principal_components
        self.feelings = feelings
        self.no_of_jumps = no_of_jumps
        self.color_map = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'yellow',
            4: 'pink',
        }

    def preprocess_data(self):
        self.array[["principal componant 1 non-diff", "principal componant 2 non-diff"]] = self.array[self.feelings].dot(self.principal_components)
        self.traj_transitions_dict = {}
        self.traj_transitions_dict_original = {}
        for heading, group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            self.traj_transitions_dict_original[heading] = group
        for heading, group in self.array.groupby(['Subject', 'Week', 'Session']):
            self.traj_transitions_dict[heading] = group

    def plot_trajectories(self):
        for heading, value in self.traj_transitions_dict_original.items():
            plt.figure()
            for feeling in self.feelings:
                starting_time = 0
                time_jump = 28
                time_array = np.arange(starting_time, starting_time + time_jump * value.shape[0], time_jump)
                plt.plot(time_array, value[feeling] * 10, label=feeling)
            combined = ''.join(map(str, heading))
            cleaned = combined.replace("\\", "").replace("'", "").replace(" ", "").replace("(", "").replace(")", "")
            plt.title(f'{cleaned}')
            plt.xlabel('Time')
            plt.ylabel('Rating')
            plt.tight_layout()

            prev_color_val = self.traj_transitions_dict[heading]['clust'].iloc[0]
            start_index = 0
            for index, color_val in enumerate(self.traj_transitions_dict[heading]['clust']):
                if color_val != prev_color_val or index == self.traj_transitions_dict[heading].shape[0] - 1:
                    end_index = index * (time_jump * self.no_of_jumps) if index != self.traj_transitions_dict[heading].shape[0] - 1 else time_array[-1]
                    plt.axvspan(start_index * (time_jump * self.no_of_jumps), end_index, facecolor=self.color_map[prev_color_val], alpha=0.3)
                    start_index = index
                    prev_color_val = color_val

            cluster_patches = [mpatches.Patch(color=color, label=f'Cluster {cluster}') for cluster, color in self.color_map.items()]
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.extend(cluster_patches)
            labels.extend([f'Cluster {cluster}' for cluster in self.dictionary_clust_labels.values()])
            plt.legend(handles=handles, labels=labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.savefig(self.savelocation_TET + f'HMM_stable_cluster_centroids')
            plt.show()

    def run(self):
        self.preprocess_data()
        self.plot_trajectories()

class JumpAnalysis:
    def __init__(self, filelocation_TET, savelocation_TET, df_csv_file_original, feelings, feelings_diffs):
        self.filelocation_TET = filelocation_TET
        self.savelocation_TET = savelocation_TET
        self.df_csv_file_original = df_csv_file_original
        self.feelings = feelings
        self.feelings_diffs = feelings_diffs

    def determine_no_jumps_stability(self):
        y_labels = []
        x_labels = []
        for j in range(1, 30):
            split_dict_skip = {}
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                group = group.iloc[::j].copy()
                for feeling in self.feelings:
                    group[f'{feeling}_diff'] = -group[feeling].diff(-1)
                split_dict_skip[(subject, week, session)] = group

            df_csv_file_new = pd.concat([df for df in split_dict_skip.values()])
            split_dict = {}
            for (subject, week, session), group in df_csv_file_new.groupby(['Subject', 'Week', 'Session']):
                split_dict[(subject, week, session)] = group.copy()
            array = pd.concat([df[:-1] for df in split_dict.values()])
            array_MI = array.copy()
            array_MI['number'] = range(array.shape[0])

            wcss_best = float('inf')
            labels_fin = []
            cluster_centres_fin = []
            for _ in range(1000):
                kmeans = KMeans(3)
                kmeans.fit(array.iloc[:, -14:])
                if kmeans.inertia_ < wcss_best:
                    wcss_best = kmeans.inertia_
                    labels_fin = kmeans.labels_
                    cluster_centres_fin = kmeans.cluster_centers_

            array_MI['labels unnormalised vectors'] = labels_fin
            unique, counts = np.unique(labels_fin, return_counts=True)
            label_counts = dict(zip(unique, counts))
            print(f'For {j} time steps, our cluster distribution is {label_counts}')

            magnitudes = [np.linalg.norm(centre) for centre in cluster_centres_fin]
            max_clust = [key for key, value in label_counts.items() if value == max(counts)]
            if magnitudes[max_clust[0]] == min(magnitudes):
                print("True")
            else:
                print('False')

            y_labels.append(max(counts) / (sum(counts) - max(counts)))
            x_labels.append(j)

        plt.plot(x_labels, y_labels)
        plt.title('Stable Cluster Dominance with No. of Time Steps')
        plt.xlabel('Number of Time Jumps')
        plt.ylabel('No in stable cluster:No in all other clusters')
        plt.savefig(self.savelocation_TET + f'HMM_stable_cluster_dominance')
        plt.show()

    def determine_no_jumps_consistency(self):
        x_values = []
        y_values = []
        for no_of_jumps in range(1, 30):
            split_dict_skip = {}
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                group = group.iloc[::no_of_jumps].copy()
                for feeling in self.feelings:
                    group[f'{feeling}_diff'] = -group[feeling].diff(-1)
                split_dict_skip[(subject, week, session)] = group

            df_csv_file_new = pd.concat([df for df in split_dict_skip.values()])
            split_dict = {}
            for (subject, week, session), group in df_csv_file_new.groupby(['Subject', 'Week', 'Session']):
                split_dict[(subject, week, session)] = group.copy()
            array = pd.concat([df[:-1] for df in split_dict.values()])
            array_MI = array.copy()
            array_MI['number'] = range(array.shape[0])

            wcss_best = float('inf')
            labels_fin = []
            cluster_centres_fin = []
            for _ in range(1000):
                kmeans = KMeans(3)
                kmeans.fit(array.iloc[:, -14:])
                if kmeans.inertia_ < wcss_best:
                    wcss_best = kmeans.inertia_
                    labels_fin = kmeans.labels_
                    cluster_centres_fin = kmeans.cluster_centers_

            array_MI['labels unnormalised vectors'] = labels_fin
            downsampled_groups = []
            for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
                downsampled = group.iloc[::no_of_jumps].copy()
                downsampled = downsampled[:-1]
                downsampled['Original_Index'] = downsampled.index
                downsampled_groups.append(downsampled)

            df_downsampled = pd.concat(downsampled_groups)
            df_downsampled['Cluster_Label'] = labels_fin
            self.df_csv_file_original['Cluster_Label'] = np.nan
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

            for feeling in self.feelings:
                self.df_csv_file_original[f'{feeling}_diff'] = -self.df_csv_file_original[feeling].diff(-1)

            n_entries = 0
            correct_assignments = 0
            for i, row in self.df_csv_file_original.iterrows():
                if not pd.isnull(row['Cluster_Label']) and not row[self.feelings_diffs].isnull().any():
                    entry = row[self.feelings_diffs] * no_of_jumps
                    assigned_cluster = row['Cluster_Label']
                    distances = np.array([euclidean(entry, centre) for centre in cluster_centres_fin])
                    closest_centre_idx = np.argmin(distances)
                    if closest_centre_idx == assigned_cluster:
                        correct_assignments += 1
                    n_entries += 1

            if n_entries > 0:
                hughes_measure = correct_assignments / n_entries
            y_values.append(hughes_measure)
            x_values.append(no_of_jumps)

        plt.plot(x_values, y_values)
        plt.title('Cluster Vectoral Consistency')
        plt.xlabel('Number of Time Steps')
        plt.ylabel('Correct Assignment Ratio')
        plt.savefig(self.savelocation_TET + f'HMM_Cluster_Vectoral_consistency')
        plt.show()

    def determine_no_of_jumps_autocorrelation(self):
        split_dict = {}
        for (subject, week, session), group in self.df_csv_file_original.groupby(['Subject', 'Week', 'Session']):
            split_dict[(subject, week, session)] = group[self.feelings].copy()

        acf_results = {feeling: [] for feeling in self.feelings}
        n_lags = 30

        for key, df in split_dict.items():
            for feeling in self.feelings:
                acf_value = acf(df[feeling], nlags=n_lags, fft=True)
                acf_results[feeling].append(acf_value)

        acf_averages = {feeling: np.mean(np.vstack(acf_results[feeling]), axis=0) for feeling in self.feelings}
        plt.figure(figsize=(12, 8))
        for feeling, acf_vals in acf_averages.items():
            plt.plot(acf_vals, label=feeling)

        plt.title('Average Autocorrelation Function for Each Feeling')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend(title='Feeling', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.savelocation_TET + f'HMM autocorrelation for each feeling')
        plt.show()
