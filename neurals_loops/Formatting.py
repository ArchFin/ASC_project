import pandas as pd

# Load the uploaded CSV file
file_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv'
data = pd.read_csv(file_path)

# Define the fixed epoch ranges for each week
week_ranges = {
    1: 301,
    2: 679,
    3: 952,
    4: 315
}

def average_epochs(group):
    week = group['Week'].iloc[0]  # Extract week from the group
    max_epoch = week_ranges.get(week, 0)
    results = []

    # Iterate over fixed ranges in steps of 7, ensuring correct number of rows
    for start in range(1, max_epoch + 1, 7):
        end = min(start + 6, max_epoch)
        subset = group[(group['epoch'] >= start) & (group['epoch'] <= end)]

        # If no epochs in the range, create a NaN-filled row
        if subset.empty:
            avg_dict = {col: 'NaN' for col in group.columns if col != 'epoch'}
            avg_dict['epoch'] = f"{start}-{end}"
        else:
            # Calculate the mean for numeric columns
            avg_dict = subset.mean(numeric_only=True).to_dict()
            avg_dict['epoch'] = f"{start}-{end}"

        # Keep track of subject, week, and run
        avg_dict['subject'] = group['Subject'].iloc[0]
        avg_dict['week'] = week
        avg_dict['run'] = group['Session'].iloc[0]
        results.append(avg_dict)

    return pd.DataFrame(results)

# Group by subject, week, and run, then apply the averaging function
#grouped = data.groupby(['Subject', 'Week', 'Session'])
#averaged_data = pd.concat([average_epochs(group) for _, group in grouped], ignore_index=True)
data['subject'] = data['Subject'].str.rstrip('\\')  # Remove trailing backslash
data['week'] = data['Week'].str.extract(r'(\d+)', expand=False)  # Extract number from 'week_#'
data['run'] = data['Session'].str.extract(r'(\d+)', expand=False).str.lstrip('0')  # Extract number and remove leading zero

# Add a counting column that restarts for each unique combination
data['number'] = data.groupby(['Subject', 'Week', 'Session']).cumcount()

# Save the processed data to a CSV file
output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions_numbered.csv'
data.to_csv(output_path, index=False)

print(f"Averaged data saved to: {output_path}")
