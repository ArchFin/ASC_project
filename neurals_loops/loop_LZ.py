## Loops through all preprocessed (& epoched) .set files in a folder and extracts LZC and LZsum values

import os
import pandas as pd
from concog_dreem_lib import process_LZ78

# Set input/output folders
folder_path = "C:/Users/benni/Desktop/Part II EEG/full_loop_resting23_results"
output_path = "C:/Users/benni/Desktop/Part II EEG/neural_data/LZ_metrics_r23.xlsx"

# Define relevant parameters for this neural feature
epoch_length = '4secs'
custom_channel_groups = {                   # Use as needed given your channel setup
    'fo_chans': [0, 1, 4, 5],
    'ff_chans': [2, 3, 6],
    'oo_chans': [7],
    'glob_chans': list(range(8))
}

# Create dataframes to which results of each loop (i.e., per file) will be saved
all_lzc, all_lzsum, all_gaps = [], [], []

# Loop through each .set file in input folder and apply function to extract neural feature
for file_name in os.listdir(folder_path):
    if file_name.endswith('.set'):
        file_path = os.path.join(folder_path, file_name)

        print(f"Processing: {file_name}")

        # Apply function to extract neural feature
        lz_results = process_LZ78(
            original_set_path=file_path,
            processed_set_path=file_path,  # Using the same file for both
            epoch_length=epoch_length,
            custom_channel_groups=custom_channel_groups
        )

        # Save results to overall dataframe
        if lz_results is not None:
            # Add participant ID column
            participant_id = file_name.replace('_epoched.set', '')
            lz_results['lz_c']['Participant'] = participant_id
            lz_results['lz_sum']['Participant'] = participant_id
            lz_results['gaps']['Participant'] = participant_id

            # Add results
            all_lzc.append(lz_results['lz_c'])
            all_lzsum.append(lz_results['lz_sum'])
            all_gaps.append(lz_results['gaps'])

        else:
            print(f"Processing failed for {file_name}")

# Concatenate all results and save to a single Excel file
if all_lzc:
    df_lzc = pd.concat(all_lzc, ignore_index=True)
    df_lzsum = pd.concat(all_lzsum, ignore_index=True)
    df_gaps = pd.concat(all_gaps, ignore_index=True)

    with pd.ExcelWriter(output_path) as writer:
        df_lzc.to_excel(writer, sheet_name='LZc', index=False)
        df_lzsum.to_excel(writer, sheet_name='LZsum', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps', index=False)

    print(f"All results saved to {output_path}")
else:
    print("No valid data to save.")

print("LZ processing loop complete")