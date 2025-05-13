import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the file path
file_path = "/Users/a_fin/Desktop/Year 4/Project/Transition analysis/transitions_summary.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Remove rows where Start Time (s) is 0
df = df[df["Start Time (s)"] != 0].copy()

# Define transition labeling function (no long/short, just state transitions)
def classify_transition_simple(row):
    from_state = row["From State"]
    to_state = row["To State"]
    return f"{from_state}→{to_state}"

# Apply the function to create a new column
# This will replace the previous 'Transition length' column
# and focus only on the state-to-state transitions
df["Transition"] = df.apply(classify_transition_simple, axis=1)

# Save the cleaned file (optional)
df.to_csv("cleaned_transitions_simple.csv", index=False)

# 1. Total count of each Transition Type (no long/short distinction)
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Transition', order=df['Transition'].value_counts().index)
plt.title("Total Count of Each State Transition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/plot_count_simple.png')

# 2. Mean event frequencies before/during/after for each transition type
summary = {}
for period in ['Before', 'During', 'After']:
    period_means = df.groupby('Transition')[[col for col in df.columns if f'{period}' in col and 'Freq' in col]].mean()
    summary[period] = period_means

# Combine into a single DataFrame for easier viewing
summary_df = pd.concat(summary, axis=1)
summary_df.to_csv('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/transition_event_frequencies.csv')

# 3. Heatmap of mean event frequencies for each transition and period
plt.figure(figsize=(16, 8))
sns.heatmap(summary_df.swaplevel(axis=1).sort_index(axis=1), annot=True, fmt='.2f', cmap='viridis')
plt.title('Mean Event Frequencies by Transition and Period')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/transition_event_heatmap.png')

# 4. Boxplots of event frequencies around each transition
conditions = list(set([col.split(' ')[1] for col in df.columns if 'Before' in col]))
plt.figure(figsize=(18, 12))
for idx, condition in enumerate(conditions, 1):
    plt.subplot(3, 3, idx)
    melt_df = df.melt(id_vars=['Transition'], 
                     value_vars=[f'Before {condition} Freq', 
                                f'During {condition} Freq',
                                f'After {condition} Freq'],
                     var_name='Period', value_name='Frequency')
    melt_df['Period'] = melt_df['Period'].str.replace(' Freq', '').str.replace(f' {condition}', '')
    sns.boxplot(x='Transition', y='Frequency', hue='Period', data=melt_df)
    plt.title(f'{condition} Frequency Distribution')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/condition_boxplots_simple.png')

# 5. Interpretation: Which transitions are most common? Which events are most modulated by transition?
# - The count plot and heatmap above show which transitions occur most frequently and how event frequencies change.
# - Look for transitions with high counts and/or large changes in event frequency before/during/after.
# - For example, if '1→2' is common and shows a spike in a particular event during the transition, this may indicate a key behavioral change.

# 6. (Optional) Add a barplot of the difference in event frequency (During - Before) for each transition and event
plt.figure(figsize=(16, 8))
diff_df = summary['During'] - summary['Before']
diff_df.plot(kind='bar', ax=plt.gca())
plt.title('Change in Event Frequency (During - Before) by Transition')
plt.ylabel('Mean Frequency Change')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/event_freq_change_barplot.png')

# 7. Rank events by frequency change for each transition and save as CSV
ranked_events = diff_df.abs().apply(lambda x: x.sort_values(ascending=False).index.tolist(), axis=1)
ranked_df = pd.DataFrame(ranked_events.tolist(), index=diff_df.index, columns=[f'Rank_{i+1}' for i in range(diff_df.shape[1])])
ranked_df.to_csv('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/event_change_ranking_by_transition.csv')

# 8. Plot event frequency trajectories (before, during, after) for each event and transition
# Extract event names from the summary DataFrame columns (remove 'Before ', 'During ', 'After ', and ' Freq')
event_names = set()
for col in summary['Before'].columns:
    if col.endswith('Freq'):
        event_names.add(col.replace(' Freq', ''))

for event in event_names:
    plt.figure(figsize=(10, 6))
    for transition in summary_df.index:
        try:
            y = [
                summary['Before'].loc[transition, f'{event} Freq'] if f'{event} Freq' in summary['Before'].columns else np.nan,
                summary['During'].loc[transition, f'{event} Freq'] if f'{event} Freq' in summary['During'].columns else np.nan,
                summary['After'].loc[transition, f'{event} Freq'] if f'{event} Freq' in summary['After'].columns else np.nan
            ]
            plt.plot(['Before', 'During', 'After'], y, marker='o', label=transition)
        except KeyError:
            continue
    plt.title(f'Event Frequency Trajectory: {event}')
    plt.ylabel('Mean Frequency')
    plt.xlabel('Period')
    plt.legend(title='Transition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/Users/a_fin/Desktop/Year 4/Project/Transition analysis/event_trajectory_{event}.png')

