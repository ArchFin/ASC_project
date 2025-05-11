import pandas as pd

# Load the uploaded CSV files
file_path_1 = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions_2.csv'
data_1 = pd.read_csv(file_path_1)[['subject', 'week', 'run', 'number', 'transition_label']]

file_path_2 = '/Users/a_fin/Desktop/Year 4/Project/Data/averaged_neurals.csv'
data_2 = pd.read_csv(file_path_2)

# Convert the merging columns to strings to ensure consistency
for col in ['subject', 'week', 'run', 'number']:
    data_1[col] = data_1[col].astype(str)
    data_2[col] = data_2[col].astype(str)

# Merge the DataFrames on the specified columns using an inner join
merged_df = pd.merge(data_1, data_2, on=['subject', 'week', 'run', 'number'], how='inner')

# Save the processed data to a CSV file
output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/neural_data_complete.csv'
merged_df.to_csv(output_path, index=False)

print(f"Averaged data saved to: {output_path}")
