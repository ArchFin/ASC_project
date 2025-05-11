import os
import pandas as pd
from concog_dreem_lib import process_permutation_entropy

# — your original parameters —
subjects = ['s10']#, 's02', 's03', 's04', 's07', 's08', 's13', 's14','s17', 's18', 's19', 's21']
            #
weeks    = [1, 2, 3, 4]
runs     = [1, 2, 3, 4, 5, 6, 7]

base_path     = "/Users/a_fin/Library/CloudStorage/OneDrive-UniversityofCambridge/Evan Lewis Healey's files - Dreem_Pilot"
epoch_length  = '4secs'
kernel        = 3
tau           = 8

custom_channel_groups = {
    'fo_chans':  [0, 1, 4, 5],
    'ff_chans':  [2, 3, 6],
    'oo_chans':  [7],
    'glob_chans': list(range(8))
}

# — load existing results —
input_path = f'/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_{tau}'
xls        = pd.ExcelFile(f"{input_path}.xlsx")
df_pe      = xls.parse('PermutationEntropy')
df_gaps    = xls.parse('Gaps')

# ensure the key columns exist
for col in ('subject', 'week', 'run', 'epoch'):
    if col not in df_pe.columns:
        raise ValueError(f"Column '{col}' missing from one of your sheets.")

# build set of already‐processed combos
processed = set(zip(df_pe['subject'], df_pe['week'], df_pe['run']))

# build the full expected set
all_combos = {
    (sub, wk, rn)
    for sub in subjects
    for wk  in weeks
    for rn  in runs
}

# find which to rerun
to_rerun = sorted(all_combos - processed)
print(f"Need to rerun {len(to_rerun)} combinations.")

new_pe_list, new_gaps_list = [], []

for subject, week, run in to_rerun:
    week_folder = f"week_{week}"
    base_fn     = f"{subject}_wk{week}_{run}"
    orig_set    = os.path.join(base_path, subject, week_folder,
                               "5-labelled&cut",
                               f"{base_fn}_hp_4sec_labelled_cut.set")
    proc_set    = os.path.join(base_path, subject, week_folder,
                               "6-reref_rej_epoch","4sec",
                               f"{base_fn}_hp_4sec_labelled_cut_reref_rej_epoch.set")
    if not os.path.exists(orig_set):
        print(f"  SKIP missing file: {orig_set}")
        continue

    pe_res = process_permutation_entropy(
        original_set_path=orig_set,
        processed_set_path=proc_set,
        epoch_length=epoch_length,
        custom_channel_groups=custom_channel_groups,
        kernel=kernel,
        tau=tau
    )
    if pe_res is None:
        print(f"  FAILED for {subject} wk{week} run{run}")
        continue

    dpe  = pe_res['pe']
    dgps = pe_res['gaps']
    for df in (dpe, dgps):
        df['subject'] = subject
        df['week']    = week
        df['run']     = run

    new_pe_list .append(dpe)
    new_gaps_list.append(dgps)
    print(f"  Completed {subject} wk{week} run{run}")

# if we got anything new, append, sort, and save
if new_pe_list:
    # concatenate
    df_pe   = pd.concat([df_pe]   + new_pe_list,  ignore_index=True)
    df_gaps = pd.concat([df_gaps] + new_gaps_list, ignore_index=True)

    # sort by epoch, subject, week, run
    sort_cols = ['subject', 'week', 'run','epoch']
    df_pe   = df_pe.sort_values(sort_cols, ignore_index=True)

    # write back
    with pd.ExcelWriter(f"{input_path}.xlsx") as writer:
        df_pe  .to_excel(writer, sheet_name='PermutationEntropy', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps',               index=False)

    print(f"✅ Updated and sorted Excel saved to {input_path}")
else:
    print("✅ No new runs were missing or all reruns failed—no update needed.")
