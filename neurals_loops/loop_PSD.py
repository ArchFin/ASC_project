import os
import pandas as pd
from concog_dreem_lib import process_psd

# List of subjects, weeks, and runs to process
subjects = ['s01', 's02', 's03', 's04', 's07', 's08', 's11', 's13', 's14', 's17', 's18', 's19', 's21']
weeks = [1, 2, 3, 4]
runs = [1, 2, 3, 4, 5, 6, 7]

# Base directory for the data files
base_path = "/Users/a_fin/Library/CloudStorage/OneDrive-UniversityofCambridge/Evan Lewis Healey's files - Dreem_Pilot"

epoch_length = '4secs'
custom_channel_groups = {
    'fo_chans': [0, 1, 4, 5],
    'ff_chans': [2, 3, 6],
    'oo_chans': [7],
    'glob_chans': list(range(8))
}

# Lists to store results from each file processing
all_psd, all_gaps = [], []

for subject in subjects:
    for week in weeks:
        for run in runs:
            # Construct folder names and file names based on subject, week, and run.
            week_folder = f"week_{week}"
            base_filename = f"{subject}_wk{week}_{run}"
            original_file = f"{base_filename}_hp_4sec_labelled_cut.set"
            processed_file = f"{base_filename}_hp_4sec_labelled_cut_reref_rej_epoch.set"
            
            # Build full paths using os.path.join for cross-platform compatibility
            original_set_path = os.path.join(base_path, subject, week_folder, "5-labelled&cut", original_file)
            processed_set_path = os.path.join(base_path, subject, week_folder, "6-reref_rej_epoch", "4sec", processed_file)
            
            # Check if the original file exists; if not, skip to the next iteration
            if not os.path.exists(original_set_path):
                print(f"Original file not found: {original_set_path}")
                continue
            
            print(f"Processing: {base_filename}")
            
            # Process the file
            psd_results = process_psd(
                original_set_path=original_set_path,
                processed_set_path=processed_set_path,
                epoch_length=epoch_length,
            )
            
            # Save results if processing was successful
            if psd_results is not None:
                # Use base_filename as the participant ID
                participant_id = base_filename
                psd_results['measures']['Participant'] = participant_id
                psd_results['gaps']['Participant'] = participant_id
                
                all_psd.append(psd_results['measures'])
                all_gaps.append(psd_results['gaps'])
                
                print(f"Processed {subject} week {week} run {run}")
            else:
                print(f"Processing failed for {base_filename}")

# After processing, combine and save the results if any processing was successful.
if all_psd:
    df_psd = pd.concat(all_psd, ignore_index=True)
    df_gaps = pd.concat(all_gaps, ignore_index=True)
    
    output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/psd_metrics_combined.xlsx'
    
    with pd.ExcelWriter(output_path) as writer:
        df_psd.to_excel(writer, sheet_name='PSD', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps', index=False)
    
    print(f"\nCombined results have been saved to {output_path}")
else:
    print("No valid PSD processing results were obtained.")