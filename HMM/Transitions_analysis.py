import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the file path
file_path = "/Users/a_fin/Desktop/Year 4/Project/Data/transitions_summary.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Remove rows where Start Time (s) is 0
df = df[df["Start Time (s)"] != 0].copy()

# Define transition labeling function
def classify_transition(row):
    from_state = row["From State"]
    to_state = row["To State"]
    duration = row["Duration (s)"]

    # Example classification logic (customize as needed)
    if from_state == 1 and to_state == 2:
        if duration < 100:
            return "Short 1→2"
        else:
            return "Long 1→2"
    elif from_state == 2 and to_state == 1:
        if duration < 100:
            return "Short 2→1"
        else:
            return "Long 2→1"
    else:
        return "Other"

# Apply the function to create a new column
df["Transition length"] = df.apply(classify_transition, axis=1)

# Save the cleaned file (optional)
df.to_csv("cleaned_transitions.csv", index=False)

# 1. Total count of each Transition Type
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Transition length', order=df['Transition length'].value_counts().index)
plt.title("Total Count of Each Transition Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/plot_count.png')

# 2. Interaction between Transition Type and Gradual/Abrupt
cross_tab = pd.crosstab(df['Transition length'], df['Transition Type'])
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title("Transition Type vs. Gradual/Abrupt Interaction")
plt.xlabel("Transition Type (Original)")
plt.ylabel("Custom Transition Type")
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/heatmap.png')

conditions = list(set([col.split(' ')[1] for col in df.columns if 'Before' in col]))

plt.figure(figsize=(18, 12))
for idx, condition in enumerate(conditions, 1):
    plt.subplot(3, 3, idx)  # Adjust grid size based on number of conditions
    
    # Melt dataframe for seaborn plotting
    melt_df = df.melt(id_vars=['Transition length'], 
                     value_vars=[f'Before {condition} Freq', 
                                f'During {condition} Freq',
                                f'After {condition} Freq'],
                     var_name='Period', value_name='Frequency')
    
    # Clean period labels
    melt_df['Period'] = melt_df['Period'].str.replace(' Freq', '').str.replace(f' {condition}', '')
    
    sns.boxplot(x='Transition length', y='Frequency', hue='Period', data=melt_df)
    plt.title(f'{condition} Frequency Distribution')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/condition_boxplots.png')

# 4. Temporal Patterns (Line Plot)
plt.figure(figsize=(14, 8))
for condition in conditions:
    # Calculate mean frequencies over transition duration
    df['Duration Group'] = pd.cut(df['Duration (s)'], bins=10)
    agg_df = df.groupby('Duration Group').agg({
        f'During {condition} Freq': 'mean',
        f'Before {condition} Freq': 'mean',
        f'After {condition} Freq': 'mean'
    }).reset_index()
    
    plt.plot(agg_df.index, agg_df[f'During {condition} Freq'], 
             label=f'{condition} During', linestyle='--')
    plt.plot(agg_df.index, agg_df[f'Before {condition} Freq'], 
             label=f'{condition} Before')
    plt.plot(agg_df.index, agg_df[f'After {condition} Freq'], 
             label=f'{condition} After', linestyle=':')

plt.title('Temporal Patterns of Condition Frequencies')
plt.xlabel('Duration Group (Deciles)')
plt.ylabel('Mean Frequency')
plt.xticks(range(10), [f'Group {i+1}' for i in range(10)])
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/temporal_patterns.png')

# 5. Condition Frequency Clustered Bar Chart
plt.figure(figsize=(16, 10))
melt_df = df.melt(id_vars=['Transition length'], 
                value_vars=[f'{period} {cond} Freq' 
                          for cond in conditions 
                          for period in ['Before', 'During', 'After']],
                var_name='Condition_Period', 
                value_name='Frequency')

# Split combined column into separate variables
melt_df[['Period', 'Condition']] = melt_df['Condition_Period'].str.split(' ', expand=True)[[0, 1]]

sns.barplot(x='Transition length', y='Frequency', 
           hue='Condition', ci='sd', 
           data=melt_df)
plt.title('Condition Frequencies by Period and Transition Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/clustered_barchart.png')

# 6. Transition Duration vs. Condition Frequencies (Pair Plot)
condition_vars = [f'During {cond} Freq' for cond in conditions]
plt.figure(figsize=(16, 12))
sns.pairplot(df, vars=condition_vars, 
            hue='Transition length', 
            plot_kws={'alpha': 0.6},
            height=3)
plt.suptitle('Condition Frequency Correlations by Transition Type', y=1.02)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/pairplot.png')

# 7. Small Multiples for Each Transition Type
g = sns.FacetGrid(df, col='Transition length', col_wrap=4, height=4, aspect=1.2)
g.map_dataframe(lambda data, color: sns.lineplot(
    x='Duration (s)', 
    y=df[[f'During {cond} Freq' for cond in conditions]].mean(axis=1),
    data=data, ci=None))
g.set_titles("{col_name}")
g.set_axis_labels("Duration (s)", "Mean Condition Frequency")
plt.suptitle('Duration vs. Condition Frequency by Transition Type', y=1.05)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/small_multiples.png')


# 8. Transition Type Duration Comparison
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Transition Type', y='Duration (s)', 
              palette=['#1f77b4', '#ff7f0e'], inner='quartile')
plt.title("Duration Distribution by Transition Type")
plt.xlabel("Transition Type")
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/duration_comparison.png')

# 9. Transition Type Frequency by Condition Period
condition_periods = ['Before', 'During', 'After']
plt.figure(figsize=(16, 10))
for idx, period in enumerate(condition_periods, 1):
    plt.subplot(2, 2, idx)
    period_data = df[[f'{period} {cond} Freq' for cond in conditions] + ['Transition Type']]
    period_data = period_data.melt(id_vars='Transition Type', var_name='Condition', value_name='Frequency')
    period_data['Condition'] = period_data['Condition'].str.replace(f'{period} ', '').str.replace(' Freq', '')
    
    sns.barplot(x='Condition', y='Frequency', hue='Transition Type', data=period_data,
               palette=['#1f77b4', '#ff7f0e'], ci='sd')
    plt.title(f'{period} Transition Frequency by Condition')
    plt.xticks(rotation=45)
    plt.legend(title='Transition Type')
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/condition_period_type.png')

# 10. Transition Type Proportion Radar Chart
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Calculate proportions
prop_df = df.groupby('Transition length')['Transition Type'] \
          .value_counts(normalize=True).unstack().fillna(0)

# Convert to radians
categories = list(prop_df.index)
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plot gradual transitions
values = prop_df['gradual'].tolist()
values += values[:1]
ax.plot(angles, values, color='#1f77b4', linewidth=2, linestyle='solid', label='Gradual')
ax.fill(angles, values, color='#1f77b4', alpha=0.25)

# Plot abrupt transitions
values = prop_df['abrupt'].tolist()
values += values[:1]
ax.plot(angles, values, color='#ff7f0e', linewidth=2, linestyle='solid', label='Abrupt')
ax.fill(angles, values, color='#ff7f0e', alpha=0.25)

plt.xticks(angles[:-1], categories, size=12)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=10)
plt.ylim(0,1)
plt.title("Transition Type Distribution by Transition Category", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/radar_chart.png')

# 11. Transition Type Temporal Distribution
plt.figure(figsize=(14, 8))
hourly_data = df.copy()
hourly_data['Start Time (s)'] = hourly_data['Start Time (s)'].astype(int)  # Ensure integer values

sns.histplot(data=hourly_data, x='Start Time (s)', hue='Transition Type',
            palette=['#1f77b4', '#ff7f0e'], multiple='stack',
            bins=50, edgecolor='white')  # Adjust bin size as needed
plt.title("Temporal Distribution of Transition Types")
plt.xlabel("Start Time (seconds)")
plt.ylabel("Count of Transitions")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/Interpretation/temporal_distribution.png')
