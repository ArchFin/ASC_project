import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import kruskal, chi2_contingency, friedmanchisquare, wilcoxon
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Define the file path
file_path = "/Users/a_fin/Desktop/Year 4/Project/Data/transitions_summary.csv"

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

# Statistical test for transition frequency equality
transition_counts = df['Transition'].value_counts()
print("\n=== STATISTICAL TESTS FOR TRANSITION FREQUENCIES ===")
print(f"Transition counts:\n{transition_counts}")

# Chi-square goodness of fit test (equal frequency hypothesis)
expected_equal = np.full(len(transition_counts), len(df) / len(transition_counts))
chi2_stat, p_value = stats.chisquare(transition_counts.values, expected_equal)
print(f"\nChi-square test for equal transition frequencies:")
print(f"Chi2 statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.6f}")
if p_value < 0.001:
    print("*** HIGHLY SIGNIFICANT deviation from equal frequencies (p < 0.001) ***")
elif p_value < 0.01:
    print("** SIGNIFICANT deviation (p < 0.01) **")
elif p_value < 0.05:
    print("* SIGNIFICANT deviation (p < 0.05) *")
else:
    print("No significant deviation from equal frequencies")

# Calculate effect size (Cramér's V equivalent for goodness of fit)
cramers_v_gof = np.sqrt(chi2_stat / (len(df) * (len(transition_counts) - 1)))
print(f"Effect size (V): {cramers_v_gof:.4f}")

# Test if bridge transitions (involving state '3') are more frequent
bridge_transitions = transition_counts[transition_counts.index.str.contains('3')].sum()
non_bridge_transitions = transition_counts[~transition_counts.index.str.contains('3')].sum()
total_transitions = len(df)

# Binomial test for bridge transition frequency
expected_bridge_prop = 4/9  # 4 out of 9 possible transitions involve state 3 (1→3, 3→1, 2→3, 3→2)
observed_bridge_prop = bridge_transitions / total_transitions
binom_p = stats.binom_test(bridge_transitions, total_transitions, expected_bridge_prop, alternative='greater')
print(f"\nBridge state transition test:")
print(f"Observed bridge transitions: {bridge_transitions}/{total_transitions} ({observed_bridge_prop:.3f})")
print(f"Expected under null: {expected_bridge_prop:.3f}")
print(f"Binomial test p-value (one-tailed): {binom_p:.6f}")
if binom_p < 0.05:
    print("* SIGNIFICANT enrichment of bridge transitions *")
else:
    print("No significant enrichment of bridge transitions")

# 2. Mean event frequencies before/during/after for each transition type
summary = {}
for period in ['Before', 'During', 'After']:
    period_means = df.groupby('Transition')[[col for col in df.columns if f'{period}' in col and 'Freq' in col]].mean()
    summary[period] = period_means

# Combine into a single DataFrame for easier viewing
summary_df = pd.concat(summary, axis=1)
summary_df.to_csv('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/transition_event_frequencies.csv')

# Statistical testing for event frequency changes
print("\n=== STATISTICAL TESTS FOR EVENT FREQUENCY CHANGES ===")

# Get event names (without Before/During/After prefixes)
event_names = set()
for col in summary['Before'].columns:
    if col.endswith('Freq'):
        event_names.add(col.replace(' Freq', ''))

statistical_results = []

# For each event, test if frequencies change significantly across periods
for event in event_names:
    event_col = f'{event} Freq'
    if event_col in summary['Before'].columns:
        print(f"\n--- Testing {event} ---")
        
        # Prepare data for each transition type
        for transition in summary_df.index:
            try:
                before_val = summary['Before'].loc[transition, event_col]
                during_val = summary['During'].loc[transition, event_col]
                after_val = summary['After'].loc[transition, event_col]
                
                # Get individual observations for this transition and event
                transition_data = df[df['Transition'] == transition]
                before_data = transition_data[f'Before {event} Freq'].dropna()
                during_data = transition_data[f'During {event} Freq'].dropna()
                after_data = transition_data[f'After {event} Freq'].dropna()
                
                if len(before_data) > 5 and len(during_data) > 5 and len(after_data) > 5:
                    # Friedman test (non-parametric repeated measures)
                    # Note: This assumes matched observations across periods
                    min_len = min(len(before_data), len(during_data), len(after_data))
                    if min_len > 5:
                        stat, p_val = friedmanchisquare(
                            before_data[:min_len], 
                            during_data[:min_len], 
                            after_data[:min_len]
                        )
                        
                        # Post-hoc pairwise tests if significant
                        pairwise_results = {}
                        if p_val < 0.05:
                            # Wilcoxon signed-rank tests for pairwise comparisons
                            stat_bd, p_bd = wilcoxon(before_data[:min_len], during_data[:min_len], alternative='two-sided')
                            stat_ba, p_ba = wilcoxon(before_data[:min_len], after_data[:min_len], alternative='two-sided')
                            stat_da, p_da = wilcoxon(during_data[:min_len], after_data[:min_len], alternative='two-sided')
                            
                            pairwise_results = {
                                'before_vs_during_p': p_bd,
                                'before_vs_after_p': p_ba,
                                'during_vs_after_p': p_da
                            }
                        
                        # Calculate effect size (Cohen's d equivalent for non-parametric)
                        effect_size = (np.median(during_data[:min_len]) - np.median(before_data[:min_len])) / np.std(before_data[:min_len])
                        
                        result = {
                            'event': event,
                            'transition': transition,
                            'friedman_stat': stat,
                            'friedman_p': p_val,
                            'effect_size_during_vs_before': effect_size,
                            'n_observations': min_len,
                            'significant': p_val < 0.05,
                            **pairwise_results
                        }
                        
                        statistical_results.append(result)
                        
                        print(f"  {transition}: Friedman χ² = {stat:.3f}, p = {p_val:.4f}", end='')
                        if p_val < 0.05:
                            print(' *')
                            if 'before_vs_during_p' in pairwise_results and pairwise_results['before_vs_during_p'] < 0.05:
                                direction = "increase" if effect_size > 0 else "decrease"
                                print(f"    Significant {direction} from before to during (p = {pairwise_results['before_vs_during_p']:.4f})")
                        else:
                            print()
                            
            except (KeyError, ValueError) as e:
                continue

# Save statistical results
if statistical_results:
    stats_df = pd.DataFrame(statistical_results)
    stats_df.to_csv('/Users/a_fin/Desktop/Year 4/Project/Transition analysis/event_frequency_statistics.csv', index=False)
    
    # Summary of significant results
    significant_results = stats_df[stats_df['significant'] == True]
    print(f"\n=== SUMMARY ===")
    print(f"Total tests performed: {len(statistical_results)}")
    print(f"Significant results: {len(significant_results)}")
    print(f"Events with significant changes:")
    for event in significant_results['event'].unique():
        transitions_sig = significant_results[significant_results['event'] == event]['transition'].tolist()
        print(f"  {event}: {', '.join(transitions_sig)}")

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

