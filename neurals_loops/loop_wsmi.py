## Loops through all preprocessed (& epoched) .set files in a folder and extracts wSMI values for given tau

import os
import pandas as pd
from concog_dreem_lib import process_wsmi

# Set input/output folders
folder_path = "C:/Users/benni/Desktop/Part II EEG/loop_epoch_results"  # Folder containing .set files
output_path = "C:/Users/benni/Desktop/Part II EEG/neural_data/wsmi_metrics.xlsx"

# Define relevant parameters for this neural feature
epoch_length = '4secs'
kernel = 4
tau = 2 # I changed this each time manually, but you could automate this into a loop if you want something neater
method_params = {"bypass_csd": True} # (Optional, see Max's code for the wSMI function for more info)

custom_channel_groups = {                   # Use as needed given your channel setup
    'fo_chans': [0, 1, 4, 5],
    'ff_chans': [2, 3, 6],
    'oo_chans': [7],
    'glob_chans': list(range(8))
}

# Create dataframes to which results of each loop (i.e., per file) will be saved
all_wsmi, all_gaps = [], []

# Loop through each .set file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.set'):
        file_path = os.path.join(folder_path, file_name)

        print(f"Processing: {file_name}")

        # Apply function to extract neural feature
        wsmi_results = process_wsmi(
            original_set_path=file_path,
            processed_set_path=file_path,
            epoch_length=epoch_length,
            custom_channel_groups=custom_channel_groups,
            kernel=kernel,
            tau=tau,
            method_params=method_params
        )

        #Save results to overall dataframe
        if wsmi_results is not None:
            # Add participant ID column
            participant_id = file_name.replace('_mr_60sec_epochs.set', '')
            wsmi_results['wsmi']['Participant'] = participant_id
            wsmi_results['gaps']['Participant'] = participant_id

            all_wsmi.append(wsmi_results['wsmi'])
            all_gaps.append(wsmi_results['gaps'])

        else:
            print(f"Processing failed for {file_name}")

# Concatenate all results and save to a single Excel file
if all_wsmi:
    df_wsmi = pd.concat(all_wsmi, ignore_index=True)
    df_gaps = pd.concat(all_gaps, ignore_index=True)

    with pd.ExcelWriter(output_path) as writer:
        df_wsmi.to_excel(writer, sheet_name='Measures', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps', index=False)

    print(f"All results saved to {output_path}")
else:
    print("No valid data to save.")

print("wsmi processing loop complete")