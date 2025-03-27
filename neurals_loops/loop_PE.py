## Loops through all preprocessed (& epoched) .set files in a folder and extracts permutation entropy values for given tau

import os
import pandas as pd
from concog_dreem_lib import process_permutation_entropy  # Import function for desired neural feature

# Set input/output folders
folder_path = "C:/Users/benni/Desktop/Part II EEG/full_loop_narrative23_results"
output_path = "C:/Users/benni/Desktop/Part II EEG/neural_data/PE_metrics_n23_tau8.xlsx"

# Define relevant parameters for this neural feature
epoch_length = '4secs'
kernel = 4
tau = 8 # I changed this each time manually, but you could automate this into a loop if you want something neater

custom_channel_groups = {                   # Use as needed given your channel setup
    'fo_chans': [0, 1, 4, 5],
    'ff_chans': [2, 3, 6],
    'oo_chans': [7],
    'glob_chans': list(range(8))
}

# Create dataframes to which results of each loop (i.e., per file) will be saved
all_pe, all_gaps = [], []

# Loop through each .set file in input folder and apply function to extract neural feature
for file_name in os.listdir(folder_path):
    if file_name.endswith('.set'):
        file_path = os.path.join(folder_path, file_name)

        print(f"Processing: {file_name}")

        # Apply function to extract neural feature
        pe_results = process_permutation_entropy(
            original_set_path=file_path,
            processed_set_path=file_path,   # Using same file for both
            epoch_length=epoch_length,
            custom_channel_groups=custom_channel_groups,
            kernel=kernel,
            tau=tau
        )

        # Save results to overall dataframe
        if pe_results is not None:
            # Add participant ID column
            participant_id = file_name.replace('_mr_60sec_epochs.set', '')
            # Save results
            pe_results['pe']['Participant'] = participant_id
            pe_results['gaps']['Participant'] = participant_id
            all_pe.append(pe_results['pe'])
            all_gaps.append(pe_results['gaps'])

        else:
            print(f"Processing failed for {file_name}")

# Concatenate all results and save to a single Excel file
if all_pe:
    df_pe = pd.concat(all_pe, ignore_index=True)
    df_gaps = pd.concat(all_gaps, ignore_index=True)

    with pd.ExcelWriter(output_path) as writer:
        df_pe.to_excel(writer, sheet_name='Measures', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps', index=False)

    print(f"All results saved to {output_path}")
else:
    print("No valid data to save.")

print("Permutation Entropy processing loop complete")
