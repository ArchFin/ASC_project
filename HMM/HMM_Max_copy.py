import pandas as pd
import yaml
from sklearn.cluster import KMeans
from scipy.stats import zscore
import os
from Max_HMM_methods_copy import Visualiser, HMMModel, DataLoader
# Main execution flow remains similar to original with corrected variable passing
with open("Breathwork.yaml") as f:
    config = yaml.safe_load(f)

tet_data, sessions, weeks, subjects, unique_ids = DataLoader.load_tet_data(
    config['filelocation_TET'], config['feelings']
)
tet_data = zscore(tet_data)

# Initialize with KMeans as original
kmeans = KMeans(n_clusters=5, n_init=10, random_state=12345)
states = kmeans.fit_predict(tet_data)

hmm = HMMModel(num_states=5, num_emissions=tet_data.shape[1])
hmm.emission_means = kmeans.cluster_centers_  # Match original initialization
hmm.train(tet_data)

state_seq, log_prob = hmm.decode(tet_data)

# Save results with original structure
results_df = pd.DataFrame({
    'TETData': list(tet_data),
    'State': state_seq,
    'SessionID': unique_ids,
    'Week': weeks,
    'Subject': subjects
})

# Generate visualizations
save_dir = config['savelocation_TET']
os.makedirs(save_dir, exist_ok=True)
# Create HMM visualizations
hmm_visualiser = Visualiser(tet_data, state_seq, hmm.trans_prob, save_dir, config['feelings'])
hmm_visualiser.plot_trajectories()
hmm_visualiser.visualise_clusters(tet_data, state_seq, save_dir)
hmm_visualiser.visualise_transitions(tet_data, state_seq, save_dir)
hmm_visualiser.visualise_clusters(tet_data, state_seq, save_dir)
hmm_visualiser.visualise_transitions(tet_data, state_seq, save_dir)
hmm_visualiser.visualise_transition_matrix(hmm.trans_prob, save_dir)
hmm_visualiser.visualise_state_transition_diagram(hmm.trans_prob, save_dir)