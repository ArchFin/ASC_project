import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.linalg import inv, det
from scipy.special import gamma
from matplotlib.lines import Line2D
import random

# Load YAML file
with open("Breathwork.yaml", "r") as file:
    config = yaml.safe_load(file)  # Converts YAML to a Python dictionary

file_path = config['filelocation_TET']
feelings_columns = config['feelings']
num_clusters = config['no_clust']
num_states = config['no_states']


from Max_HMM_methods import TETProcessor, ClusterAnalysis, StatisticalFunctions, HMMModel, TETVisualisation

processor = TETProcessor(file_path)
processor.load_data()
processor.scale_data(feelings_columns)

cluster_analysis = ClusterAnalysis(processor.get_scaled_data())
cluster_analysis.find_optimal_clusters()
cluster_analysis.perform_clustering()

hmm_model = HMMModel(cluster_analysis.get_cluster_labels(), cluster_analysis.optimal_k)
hmm_model.train_hmm()
hmm_model.predict_states()

visualiser = TETVisualisation(hmm_model.get_state_sequence(), processor.get_scaled_data())
visualiser.visualize_clusters_and_transitions(cluster_analysis.get_cluster_labels())
