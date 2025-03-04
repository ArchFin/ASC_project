import os
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_TET_data(main_folder, participant_folders):
    all_TET_data = []
    participant_ids = []  # To store participant identifiers
    meditation_styles = []
    session_ids = []
    
    for participant_folder in participant_folders:
        # Construct path to 20-SubjExp subfolder
        full_path = os.path.join(main_folder, participant_folder, '20-SubjExp')
        
        # Find all _TET.mat files in the folder
        for file_name in os.listdir(full_path):
            if file_name.endswith('_TET.mat'):
                file_path = os.path.join(full_path, file_name)
                print(f"Now reading {file_path}")
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'Subjective' in f:
                            # Load the dataset and convert to a NumPy array.
                            data = np.array(f['Subjective'])
                            data = data.T  # Transpose to match MATLAB's orientation if needed
                            
                            # Check that data has at least 13 columns (for columns 0-12)
                            if data.shape[1] < 13:
                                print(f"Insufficient columns in {file_path}")
                                continue

                            # Subsample: take every 28th row from the data
                            data = data[::7, :]
                            
                            # Extract columns 0-11
                            TET_data = data[:, :12]
                            
                            # If any previous data exists, ensure the dimensions match
                            if all_TET_data and TET_data.shape[1] != all_TET_data[0].shape[1]:
                                print(f"Dimension mismatch in {file_path}")
                                continue
                            
                            all_TET_data.append(TET_data)
                            
                            # Metadata handling:
                            num_rows = TET_data.shape[0]
                            meditation_styles.extend(data[:, 12].tolist())
                            session_ids.extend([file_name] * num_rows)
                            participant_ids.extend([participant_folder] * num_rows)
                        else:
                            print(f"'Subjective' not found in {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Stack the data if any was loaded
    if all_TET_data:
        all_TET_data = np.vstack(all_TET_data)
    else:
        all_TET_data = np.array([])
    
    return (
        all_TET_data,
        np.array(participant_ids),
        np.array(meditation_styles),
        np.array(session_ids)
    )

# Feelings labels for the 12 TET dimensions
feelings = [
"Aperture",
"Boredom",
"Clarity",
"Conflict",
"Dereification",
"Wakefulness",
"Emotion",
"Effort",
"Stability",
"MetaAwareness",
"ObjectOrientation",
"Source",
]

main_folder = '/Users/a_fin/Desktop/Year 4/Project/Meditation'
participant_folders = [
    '1425',
    '1733_BandjarmasinKomodoDragon',
    '1871',
    '1991_MendozaCow',
    '2222_JiutaiChicken',
    '2743_HuaianKoi',
    '184_WestYorkshireWalrus',
    '1465_WashingtonQuelea',
    '1867_BucharestTrout',
    '1867_GoianiaCrane',
    '3604_LichuanHookworm',
    '3604_ShangquiHare',
    '3614_BrisbaneHornet',
    '3614_VientianeWhippet',
    '3938_YingchengSeaLion',
    '4765_NouakchottMoose',
    '5644_AkesuCoral',
    '5892_LvivRooster',
    '5892_NonthaburiHalibut',
    '7135_TampicoWallaby',
    '8681_NanchangAlbatross',
    '8725_ShishouMosquito',
    '8725_YangchunCobra'
]

all_TET_data, participant_ids, meditation_styles, session_ids = load_TET_data(main_folder, participant_folders)
    
# Normalize data if any data was loaded
if all_TET_data.size > 0:
    all_TET_data = MinMaxScaler().fit_transform(all_TET_data)
else:
    print("No TET data loaded!")
    
# Convert to a pandas DataFrame with normalized data
df = pd.DataFrame(all_TET_data)

# Insert 'Subject' and 'Session' at the front of the DataFrame
df.insert(0, 'Subject', participant_ids)
df.insert(1, 'Session', session_ids)

# Rename numeric columns to the feeling labels
df.columns = ["Subject", "Session"] + feelings

# Print the first few rows to verify the columns
print(df.head())

# Save the final DataFrame to a CSV file
output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/Meditation_TET_data.csv'
df.to_csv(output_path, index=False)
print(f"Final product saved to {output_path}")