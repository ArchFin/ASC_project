import os
import math
import pandas as pd
from concog_dreem_lib import process_wsmi

subjects = ['s01']#, 's02']#, 's03', 's04', 's07', 's08', 's13', 's14', 's17', 's18', 's19', 's21']
weeks    = [1, 2, 3, 4]
runs     = [1, 2, 3, 4, 5, 6, 7]

base_path     = "/Users/a_fin/Library/CloudStorage/OneDrive-UniversityofCambridge/Evan Lewis Healey's files - Dreem_Pilot"
epoch_length  = '4secs'
kernel        = 3
tau           = 1
method_params = {"bypass_csd": True}

custom_channel_groups = {
    'fo_chans':   [0, 1, 4, 5],
    'ff_chans':   [2, 3, 6],
    'oo_chans':   [7],
    'glob_chans': list(range(8))
}

# collect every wsmi DataFrame
all_wsmi = []

for subject in subjects:
    for week in weeks:
        for run in runs:
            week_folder   = f"week_{week}"
            base_filename = f"{subject}_wk{week}_{run}"
            orig_path     = os.path.join(base_path, subject, week_folder, "5-labelled&cut",
                                         base_filename + "_hp_4sec_labelled_cut.set")
            proc_path     = os.path.join(base_path, subject, week_folder, "6-reref_rej_epoch", "4sec",
                                         base_filename + "_hp_4sec_labelled_cut_reref_rej_epoch.set")

            if not os.path.exists(orig_path):
                continue

            wsmi_results = process_wsmi(
                original_set_path=orig_path,
                processed_set_path=proc_path,
                epoch_length=epoch_length,
                custom_channel_groups=custom_channel_groups,
                kernel=kernel,
                tau=tau,
                method_params=method_params
            )
            if wsmi_results is None:
                continue

            df = wsmi_results['wsmi']
            # add meta so we can group on it
            df['subject'] = subject
            df['week']    = week
            df['run']     = run
            # df already has an 'epoch' column
            all_wsmi.append(df)

# concatenate everything
combined = pd.concat(all_wsmi, ignore_index=True)

metrics = [
    'wSMI',
    'SMI'
]

# Group by epoch, subject, week, and run, then take the mean across channels
grouped = (
    combined
    .groupby(['subject', 'week', 'run', 'epoch'], as_index=False)[metrics]
    .mean()
)

# 4. Rename each averaged metric by appending '_glob_chans'
grouped = grouped.rename(
    columns={col: f"{col}_glob_chans" for col in metrics}
)

# write out
out_path = f"/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_{tau}_global.xlsx"
grouped.to_excel(out_path, index=False)
print(f"Saved per‐epoch global averages to {out_path}")
