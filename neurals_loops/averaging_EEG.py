import pandas as pd

# 1. Load your data (adjust the path/sep as needed)
df = pd.read_excel('/Users/a_fin/Desktop/Year 4/Project/Data/psd_metrics_combined.xlsx')

# 2. Define the power/metric columns to average
metrics = [
    'Offset',
    'Exponent',
    'Delta_Power',
    'Theta_Power',
    'Alpha_Power',
    'Beta_Power',
    'Gamma_Power'
]

# 3. Group by epoch, subject, week, and run, then take the mean across channels
grouped = (
    df
    .groupby(['epoch', 'subject', 'week', 'run'], as_index=False)[metrics]
    .mean()
)

# 4. Rename each averaged metric by appending '_glob_chans'
grouped = grouped.rename(
    columns={col: f"{col}_glob_chans" for col in metrics}
)

# 5. Sort by subject, week, and run
grouped = grouped.sort_values(['subject', 'week', 'run']).reset_index(drop=True)

#Save to CSV or inspect
grouped.to_excel('/Users/a_fin/Desktop/Year 4/Project/Data/psd_metrics_combined_avg.xlsx', index=False)
print(grouped)