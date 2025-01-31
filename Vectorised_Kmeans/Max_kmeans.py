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
#from statsmodels.tsa.stattools import acf
import yaml

# Load YAML file
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

from Max_kmeans_methods import csv_splitter, principal_component_finder, KMeansVectorClustering, KMeansVectorVisualizer, JumpAnalysis

csv_splitter_instance = csv_splitter(config['filelocation_TET'])
df_csv_file_original = csv_splitter_instance.read_CSV()
split_df, split_csv_array = csv_splitter_instance.split_by_header(df_csv_file_original,config['header'])
principal_component_finder_instance = principal_component_finder(df_csv_file_original,config['feelings'],config['no_dimensions_PCA'], config['savelocation_TET'] )
principal_components, explained_variance_ratio, df_TET_feelings_prin = principal_component_finder_instance.PCA_TOT()
df_TET_feelings_prin_dict = principal_component_finder_instance.PCA_split(split_csv_array)


kmeans_clustering_instance = KMeansVectorClustering(
    config['filelocation_TET'], 
    config['savelocation_TET'], 
    df_csv_file_original, 
    config['feelings'], 
    config['feelings_diffs'], 
    principal_components, 
    config['no_of_jumps'],
    config['colours'], 
    config['colours_list']
)

# Call the appropriate methods to get the values you need
differences_array, dictionary_clust_labels = kmeans_clustering_instance.run()  # Ensure data is processed

# Create an instance of KMeansVisualizer
visualizer = KMeansVectorVisualizer(
    filelocation_TET = config['filelocation_TET'], 
    savelocation_TET = config['savelocation_TET'],
    differences_array=differences_array, 
    df_csv_file_original=df_csv_file_original, 
    dictionary_clust_labels=dictionary_clust_labels, 
    principal_components=principal_components, 
    feelings=config['feelings'], 
    no_of_jumps=config['no_of_jumps']
).run()

jump_analysis = JumpAnalysis(
    config['filelocation_TET'], 
    config['savelocation_TET'], 
    df_csv_file_original, 
    config['feelings'], 
    config['feelings_diffs'])
jump_analysis.determine_no_jumps_stability()
jump_analysis.determine_no_jumps_consistency()
jump_analysis.determine_no_of_jumps_autocorrelation()

