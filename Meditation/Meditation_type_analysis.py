import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap

# Load the HMM output CSV
csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv'
df = pd.read_csv(csv_path)

output_dir = '/Users/a_fin/Desktop/Year 4/Project/Data'


# Cross-frequency analysis: Med_type vs transition_label
if 'Med_type' in df.columns and 'transition_label' in df.columns:
    cross_tab = pd.crosstab(df['Med_type'], df['transition_label'])
    print('\nCross-frequency table (Med_type vs transition_label):')
    print(cross_tab)
    cross_tab_path = os.path.join(output_dir, 'Med_type_vs_transition_label_crosstab_Expert.csv')
    cross_tab.to_csv(cross_tab_path)
    # Optional: plot as heatmap
    plt.figure(figsize=(10, 6))
    plt.title('Med_type vs transition_label Frequency')
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Med_type')
    plt.xlabel('transition_label')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'Med_type_vs_transition_label_heatmap_Expert.png')
    plt.savefig(heatmap_path)
    plt.close()
    # Relative (row-normalized) heatmap
    cross_tab_rel = cross_tab.div(cross_tab.sum(axis=1), axis=0)
    plt.figure(figsize=(10, 6))
    plt.title('Med_type vs transition_label (Relative Frequency)')
    sns.heatmap(cross_tab_rel, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('Med_type')
    plt.xlabel('transition_label')
    plt.tight_layout()
    rel_heatmap_path = os.path.join(output_dir, 'Med_type_vs_transition_label_heatmap_relative_Expert.png')
    plt.savefig(rel_heatmap_path)
    plt.close()
    # Save relative crosstab to CSV
    cross_tab_rel_path = os.path.join(output_dir, 'Med_type_vs_transition_label_crosstab_relative_Expert.csv')
    cross_tab_rel.to_csv(cross_tab_rel_path)
else:
    print("Both 'Med_type' and 'transition_label' columns are required for cross-frequency analysis.")