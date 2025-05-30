import pandas as pd

# Load the uploaded CSV file
file_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv'
data = pd.read_csv(file_path)


# Group by subject, week, and run, then apply the averaging function
#grouped = data.groupby(['Subject', 'Week', 'Session'])
#averaged_data = pd.concat([average_epochs(group) for _, group in grouped], ignore_index=True)
data['subject'] = data['Subject'].str.rstrip('\\')  # Remove trailing backslash
data['week'] = data['Week'].str.extract(r'(\d+)', expand=False)  # Extract number from 'week_#'
data['run'] = data['Session'].str.extract(r'(\d+)', expand=False).str.lstrip('0')  # Extract number and remove leading zero

# Add a counting column that restarts for each unique combination
data['number'] = data.groupby(['subject', 'week', 'run']).cumcount()

# Save the processed data to a CSV file
output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions_numbered.csv'
data.to_csv(output_path, index=False)

print(f"Averaged data saved to: {output_path}")
