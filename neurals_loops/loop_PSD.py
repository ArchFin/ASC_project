## Loops through all preprocessed (& epoched) .set files in a folder and extracts PSD values

import os
import pandas as pd
from concog_dreem_lib import process_psd

# Set input and output folder
folder_path = "C:/Users/benni/Desktop/Part II EEG/full_loop_resting23_results"
output_path = "C:/Users/benni/Desktop/Part II EEG/neural_data/PSD_metrics_r23.xlsx"

# Define relevant parameters for this neural feature
epoch_length = '4secs'

# Create dataframes to which results of each loop (i.e., per file) will be saved
all_psd, all_gaps = [], []

# Loop through each .set file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.set'):
        file_path = os.path.join(folder_path, file_name)

        print(f"Processing: {file_name}")

        # Apply function to extract neural feature
        psd_results = process_psd(
            original_set_path=file_path,
            processed_set_path=file_path,
            epoch_length=epoch_length,
        )

        # Save results to overall dataframe
        if psd_results is not None:
            # Add participant ID column
            participant_id = file_name.replace('_epoched.set', '')
            psd_results['measures']['Participant'] = participant_id
            psd_results['gaps']['Participant'] = participant_id

            all_psd.append(psd_results['measures'])
            all_gaps.append(psd_results['gaps'])

        else:
            print(f"Processing failed for {file_name}")

# Concatenate all results and save to a single Excel file
if all_psd:
    df_psd = pd.concat(all_psd, ignore_index=True)
    df_gaps = pd.concat(all_gaps, ignore_index=True)

    with pd.ExcelWriter(output_path) as writer:
        df_psd.to_excel(writer, sheet_name='PSD', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps', index=False)

    print(f"All results saved to {output_path}")
else:
    print("No valid data to save.")

print("PSD processing loop complete")