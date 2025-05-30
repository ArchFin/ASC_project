import os
import pandas as pd
from concog_dreem_lib import process_LZ78

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

# List to store results from each file processing
results_list = []

for subject in subjects:
    for week in weeks:
        for run in runs:
            # Construct folder names and file names based on subject, week, and run.
            week_folder = f"week_{week}"
            # The naming convention: e.g., s01_wk1_1
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
            
            # Process the file (assumes process_LZ78 is defined/imported)
            lz_results = process_LZ78(
                original_set_path=original_set_path,
                processed_set_path=processed_set_path,
                epoch_length=epoch_length,
                custom_channel_groups=custom_channel_groups
            )
            
            if lz_results is not None:
                # Retrieve DataFrames
                df_lzc = lz_results['lz_c']
                df_lzsum = lz_results['lz_sum']
                df_gaps = lz_results['gaps']
                
                # Add metadata columns to help identify the source
                for df in [df_lzc, df_lzsum, df_gaps]:
                    df['subject'] = subject
                    df['week'] = week
                    df['run'] = run
                
                # Save the results along with metadata
                results_list.append({
                    'Subject': subject,
                    'Week': week,
                    'Session': run,
                    'lz_c': df_lzc,
                    'lz_sum': df_lzsum,
                    'gaps': df_gaps
                })
                
                print(f"Processed {subject} week {week} run {run}")
            else:
                print(f"Processing LZ78 failed for {subject} week {week} run {run}")

# After processing, combine and save the results if any processing was successful.
if results_list:
    # Combine all lz_c, lz_sum, and gaps DataFrames from the results list
    combined_df_lzc = pd.concat([res['lz_c'] for res in results_list], ignore_index=True)
    combined_df_lzsum = pd.concat([res['lz_sum'] for res in results_list], ignore_index=True)
    combined_df_gaps = pd.concat([res['gaps'] for res in results_list], ignore_index=True)
    
    output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/lz_metrics_combined.xlsx'
    
    with pd.ExcelWriter(output_path) as writer:
        combined_df_lzc.to_excel(writer, sheet_name='LZc', index=False)
        combined_df_lzsum.to_excel(writer, sheet_name='LZsum', index=False)
        combined_df_gaps.to_excel(writer, sheet_name='Gaps', index=False)
    
    print(f"\nCombined results have been saved to {output_path}")
else:
    print("No valid LZ78 processing results were obtained.")
