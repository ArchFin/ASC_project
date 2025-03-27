import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

# ---------------------------------------
# 1. Load the data from CSV.
# ---------------------------------------
df = pd.read_csv("/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted.csv")

# ---------------------------------------
# 2. Define function to compute NMI.
# ---------------------------------------

def compute_pairwise_nmi(data, subjects):
    """
    Compute a pairwise Normalized Mutual Information (NMI) matrix between subjects.

    Parameters:
    - data: DataFrame containing the transition labels for a specific (Week, Session).
    - subjects: List of unique subjects in that session.

    Returns:
    - A DataFrame (subjects x subjects) containing NMI values.
    """
    nmi_matrix = pd.DataFrame(index=subjects, columns=subjects, dtype=float)

    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            if j > i:  # Only compute upper triangle
                # Reset index to align sequences by position
                series1 = data.loc[data['Subject'] == s1, 'transition_label'].reset_index(drop=True)
                series2 = data.loc[data['Subject'] == s2, 'transition_label'].reset_index(drop=True)

                # Find common indices (positions)
                min_length = min(len(series1), len(series2))
                if min_length > 0:
                    # Use only up to the minimum length
                    nmi = normalized_mutual_info_score(series1[:min_length], series2[:min_length])
                else:
                    nmi = np.nan  # No data available
                
                # Fill both (i, j) and (j, i) since NMI is symmetric
                nmi_matrix.loc[s1, s2] = nmi
                nmi_matrix.loc[s2, s1] = nmi

        # Set diagonal to 1.0 (self-comparison)
        nmi_matrix.loc[s1, s1] = 1.0  

    return nmi_matrix

# ---------------------------------------
# 3. Compute and plot NMI for each (Week, Session).
# ---------------------------------------

# Get unique (Week, Session) combinations
week_session_groups = df.groupby(['Week', 'Session'])

for (week, session), df_subset in week_session_groups:
    subjects_in_session = df_subset['Subject'].unique()
    
    # Only compute if there are at least 2 subjects
    if len(subjects_in_session) < 2:
        print(f"Skipping {week}, {session}: Not enough subjects")
        continue

    # Compute NMI matrix
    nmi_matrix = compute_pairwise_nmi(df_subset, subjects_in_session)

    # Create a heatmap for this (Week, Session) combination
    plt.figure(figsize=(10, 8))
    sns.heatmap(nmi_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
    plt.title(f"NMI Heatmap for {week}, {session}")
    plt.xlabel("Subject")
    plt.ylabel("Subject")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f'/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/NMI_between_subjects_{week}_{session}.png')


    plt.close()  # Display each heatmap separately

# Initialize a list to store aggregated NMI data
nmi_data = []

# Process each (Week, Session) group
for (week, session), df_subset in week_session_groups:
    subjects = df_subset['Subject'].unique()
    if len(subjects) < 2:
        continue  # Skip groups with <2 subjects
    
    # Compute NMI matrix
    nmi_matrix = compute_pairwise_nmi(df_subset, subjects)
    
    # Extract upper triangle values (excluding diagonal)
    mask = np.triu(np.ones(nmi_matrix.shape, dtype=bool), k=1)  # Upper triangle mask
    upper_triangle = nmi_matrix.where(mask).stack().dropna()

    # Append to aggregated data
    for (s1, s2), nmi_score in upper_triangle.items():
        nmi_data.append({
            'Week': week,
            'Session': session,
            'NMI': nmi_score
        })

# Convert to DataFrame
nmi_df = pd.DataFrame(nmi_data)

# ---------------------------------------
# 4. Plot Distributions by Week and Session
# ---------------------------------------
plt.figure(figsize=(12, 8))

# Plot the violin plot first (this creates the legend)
ax = sns.violinplot(
    data=nmi_df,
    x='Week', y='NMI', hue='Session',
    split=True, inner='quartile', palette='pastel'
)

# # Plot the swarm plot, but let it not contribute to the legend.
# # We do this by reusing the same axis and then later resetting the legend.
# sns.swarmplot(
#     data=nmi_df,
#     x='Week', y='NMI', hue='Session',
#     dodge=True, color='black', size=4, ax=ax
# )

# Capture the legend entries created by the violin plot
handles, labels = ax.get_legend_handles_labels()

# Assume the first set of handles correspond to the violin plot.
# If the swarmplot has added duplicates, they will appear later in the list.
# Here we keep only unique labels.
unique = {}
for handle, label in zip(handles, labels):
    if label not in unique:
        unique[label] = handle

# Place the legend outside the graph
ax.legend(unique.values(), unique.keys(), title='Session', bbox_to_anchor=(1.02, 0.9), loc='upper left')

plt.title('Distribution of NMI Scores by Week and Session')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/NMI_violin.png')

plt.close()