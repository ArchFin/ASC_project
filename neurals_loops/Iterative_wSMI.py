import os
import pandas as pd
import math
from concog_dreem_lib import process_wsmi


# — your original parameters —
subjects = ['s10']
#           ['s01', 's02'] # ['s03', 's04'] # ['s07','s08'] # ['s13', 's14'] # ['s17', 's18'] # 
weeks    = [1,2,3,4]
runs     = [1,2,3,4,5,6,7]

base_path     = "/Users/a_fin/Library/CloudStorage/OneDrive-UniversityofCambridge/Evan Lewis Healey's files - Dreem_Pilot"
epoch_length  = '4secs'
kernel        = 3
tau           = 8
method_params = {"bypass_csd": True}

custom_channel_groups = {
    'fo_chans':   [0, 1, 4, 5],
    'ff_chans':   [2, 3, 6],
    'oo_chans':   [7],
    'glob_chans': list(range(8))
}

# — load existing results —
input_path = f"/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_{tau}_global"
xls        = pd.ExcelFile(f"{input_path}_2.xlsx")
df_wsmi    = xls.parse('Sheet1')   # or 'Measures' if only one sheet
#df_gaps    = xls.parse('Gaps')

# Check keys
for col in ('subject', 'week', 'run', 'epoch'):
    if col not in df_wsmi.columns:
        raise ValueError(f"Column '{col}' missing in existing Measures sheet")

# Find which (sub,week,run) still need running
processed = set(zip(df_wsmi['subject'], df_wsmi['week'], df_wsmi['run']))
all_combos = {
    (sub, wk, rn)
    for sub in subjects
    for wk in weeks
    for rn in runs
}
to_rerun = sorted(all_combos - processed)
print(f"Need to rerun {len(to_rerun)} combinations.")

new_wsmi_list = []

for subject, week, run in to_rerun:
    week_folder = f"week_{week}"
    base_fn     = f"{subject}_wk{week}_{run}"
    orig_set    = os.path.join(base_path, subject, week_folder, "5-labelled&cut",
                               f"{base_fn}_hp_4sec_labelled_cut.set")
    proc_set    = os.path.join(base_path, subject, week_folder, "6-reref_rej_epoch", "4sec",
                               f"{base_fn}_hp_4sec_labelled_cut_reref_rej_epoch.set")

    if not os.path.exists(orig_set):
        print(f"  SKIP missing file: {orig_set}")
        continue

    wsmi_res = process_wsmi(
        original_set_path=orig_set,
        processed_set_path=proc_set,
        epoch_length=epoch_length,
        custom_channel_groups=custom_channel_groups,
        kernel=kernel,
        tau=tau,
        method_params=method_params
    )

    if wsmi_res is None:
        print(f"  FAILED for {subject} wk{week} run{run}")
        continue

    dwsmi = wsmi_res['wsmi']
    #dgaps = wsmi_res['gaps']
    # add meta
    #for df in (dwsmi):
    dwsmi['subject'] = subject
    dwsmi['week']    = week
    dwsmi['run']     = run

    metrics = [
    'wSMI',
    'SMI'
    ]

    # Group by epoch, subject, week, and run, then take the mean across channels
    grouped = (
        dwsmi
        .groupby(['subject', 'week', 'run', 'epoch'], as_index=False)[metrics]
        .mean()
    )

    # Rename each averaged metric by appending '_glob_chans'
    grouped = grouped.rename(
        columns={col: f"{col}_glob_chans" for col in metrics}
    )

    new_wsmi_list.append(grouped)
    #new_gaps_list.append(dgaps)
    print(f"  Completed {subject} wk{week} run{run}")

# append any new runs
if new_wsmi_list:
    df_wsmi = pd.concat([df_wsmi] + new_wsmi_list, ignore_index=True)
    #df_gaps = pd.concat([df_gaps] + new_gaps_list, ignore_index=True)

    # sort for cleanliness
    df_wsmi = df_wsmi.sort_values(['subject','week','run','epoch'], ignore_index=True)


    # write out only the summary
    output_path = f'{input_path}_2.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        df_wsmi.to_excel(writer, sheet_name='Sheet1', index=False)
        # you can still write gaps raw if you like:
        #df_gaps.to_excel(writer, sheet_name='Gaps', index=False)

    print(f"✅ Updated global-summary written to {output_path}")

else:
    print("✅ No new runs—or all reruns failed—so nothing to update.")
