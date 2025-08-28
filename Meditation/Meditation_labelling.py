import os
import pandas as pd

# Paths
csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/Meditation_TET_data.csv'
meditation_dir = '/Users/a_fin/Desktop/Year 4/Project/Meditation'
output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/Meditation_TET_data_labelled_noThought.csv'

df = pd.read_csv(csv_path)
labels = []

for idx, row in df.iterrows():
    subject = row['Subject']
    session = row['Session']
    # Remove '_TET.mat' from session to get session_name
    session_name = str(session).replace('_TET.mat', '')
    # Try direct file first
    txt_file = os.path.join(meditation_dir, f'T_{subject}.txt')
    label_found = None
    file_found = False
    if os.path.exists(txt_file):
        file_found = True
        with open(txt_file, 'r') as f:
            for line in f:
                # Look for a line that contains the session_name
                if session_name in line:
                    # Extract the fourth comma-separated value (Breath)
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        label_found = parts[6].strip()  # 4th value for direct file
                    break
    else:
        # Try subdirectory
        sub_dir = os.path.join(meditation_dir, subject)
        txt_file_sub = os.path.join(sub_dir, f'T_{subject}.txt')
        if os.path.exists(txt_file_sub):
            file_found = True
            with open(txt_file_sub, 'r') as f:
                for line in f:
                    # Look for a line that contains the session_name
                    if session_name in line:
                        # Extract the last comma-separated value (Breath) for subdirectory file
                        parts = line.strip().split(',')
                        if len(parts) >= 1:
                            label_found = parts[-1].strip()
                        break
    labels.append(label_found)

df['Med_type'] = labels

# Standardize label names
label_map = {
    'Breathing': 'Breath',
    'Compassion': 'Loving Kindness',
    'LK': 'Loving Kindness',
    'OM': 'Open Monitoring'
}
df['Med_type'] = df['Med_type'].replace(label_map)

# Exclude all rows where Med_type is 'Thought'
filtered_df = df[df['Med_type'] != 'Thought']

filtered_df.to_csv(output_path, index=False)
print(f"Saved labeled data to {output_path}")