import os
import pandas as pd
from concog_dreem_lib import process_wsmi

# List of subjects, weeks, and runs to process
subjects = ['s01', 's02', 's03', 's04', 's07', 's08', 's11', 's13', 's14', 's17', 's18', 's19', 's21']
weeks = [1, 2, 3, 4]
runs = [1, 2, 3, 4, 5, 6, 7]

# Base directory for the data files
base_path = "/Users/a_fin/Library/CloudStorage/OneDrive-UniversityofCambridge/Evan Lewis Healey's files - Dreem_Pilot"

# Define parameters for wSMI (similar to permutation entropy parameters)
epoch_length = '4secs'
kernel = 4
tau = 8
method_params = {"bypass_csd": True}  # Optional parameters

custom_channel_groups = {
    'fo_chans': [0, 1, 4, 5],
    'ff_chans': [2, 3, 6],
    'oo_chans': [7],
    'glob_chans': list(range(8))
}

# Create lists to store results from each file processing
all_wsmi, all_gaps = [], []

# Output path for saving the combined Excel file
output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_metrics_combined.xlsx'

for subject in subjects:
    for week in weeks:
        for run in runs:
            # Construct folder names and file names based on subject, week, and run.
            week_folder = f"week_{week}"
            # Naming convention: e.g., s01_wk1_1
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
            
            # Process the file to extract wSMI values (using full paths)
            wsmi_results = process_wsmi(
                original_set_path=original_set_path,
                processed_set_path=processed_set_path,
                epoch_length=epoch_length,
                custom_channel_groups=custom_channel_groups,
                kernel=kernel,
                tau=tau,
                method_params=method_params
            )
            
            if wsmi_results is not None:
                # Retrieve DataFrames for wsmi and gaps
                df_wsmi = wsmi_results['wsmi']
                df_gaps = wsmi_results['gaps']
                
                # Add metadata columns to identify the source
                df_wsmi['subject'] = subject
                df_wsmi['week'] = week
                df_wsmi['run'] = run

                df_gaps['subject'] = subject
                df_gaps['week'] = week
                df_gaps['run'] = run
                
                # Save the results along with metadata
                all_wsmi.append(df_wsmi)
                all_gaps.append(df_gaps)
                
                print(f"Processed {subject} week {week} run {run}")
            else:
                print(f"Processing wSMI failed for {subject} week {week} run {run}")

# After processing, combine and save the results if any processing was successful.
if all_wsmi:
    combined_df_wsmi = pd.concat(all_wsmi, ignore_index=True)
    combined_df_gaps = pd.concat(all_gaps, ignore_index=True)
    
    with pd.ExcelWriter(output_path) as writer:
        combined_df_wsmi.to_excel(writer, sheet_name='Measures', index=False)
        combined_df_gaps.to_excel(writer, sheet_name='Gaps', index=False)
    
    print(f"\nCombined wSMI results have been saved to {output_path}")
else:
    print("No valid wSMI processing results were obtained.")

print("wSMI processing loop complete")
