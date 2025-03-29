import os
import pandas as pd
from concog_dreem_lib import process_permutation_entropy  # Import function for permutation entropy

# List of subjects, weeks, and runs to process
subjects = ['s01', 's02', 's03', 's04', 's07', 's08', 's11', 's13', 's14', 's17', 's18', 's19', 's21']
weeks = [1, 2, 3, 4]
runs = [1, 2, 3, 4, 5, 6, 7]

# Base directory for the data files
base_path = "/Users/a_fin/Library/CloudStorage/OneDrive-UniversityofCambridge/Evan Lewis Healey's files - Dreem_Pilot"

# Define parameters for permutation entropy
epoch_length = '4secs'
kernel = 4
tau = 8

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
            
            # Process the file for permutation entropy (using full paths)
            pe_results = process_permutation_entropy(
                original_set_path=original_set_path,
                processed_set_path=processed_set_path,
                epoch_length=epoch_length,
                custom_channel_groups=custom_channel_groups,
                kernel=kernel,
                tau=tau
            )
            
            if pe_results is not None:
                # Retrieve DataFrames for permutation entropy and gaps
                df_pe = pe_results['pe']
                df_gaps = pe_results['gaps']
                
                # Optionally add metadata columns to help identify the source
                df_pe['subject'] = subject
                df_pe['week'] = week
                df_pe['run'] = run
                
                df_gaps['subject'] = subject
                df_gaps['week'] = week
                df_gaps['run'] = run
                
                # Save the results along with metadata
                results_list.append({
                    'subject': subject,
                    'week': week,
                    'run': run,
                    'pe': df_pe,
                    'gaps': df_gaps
                })
                
                print(f"Processed {subject} week {week} run {run}")
            else:
                print(f"Processing permutation entropy failed for {subject} week {week} run {run}")

# After processing, combine and save the results if any processing was successful.
if results_list:
    # Combine all permutation entropy and gaps DataFrames from the results list
    combined_df_pe = pd.concat([res['pe'] for res in results_list], ignore_index=True)
    combined_df_gaps = pd.concat([res['gaps'] for res in results_list], ignore_index=True)
    
    output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined.xlsx'
    
    with pd.ExcelWriter(output_path) as writer:
        combined_df_pe.to_excel(writer, sheet_name='PermutationEntropy', index=False)
        combined_df_gaps.to_excel(writer, sheet_name='Gaps', index=False)
    
    print(f"\nCombined permutation entropy results have been saved to {output_path}")
else:
    print("No valid permutation entropy processing results were obtained.")

print("Permutation Entropy processing loop complete")
