import os
import pandas as pd
from concog_dreem_lib import process_permutation_entropy

# — your original parameters —
subjects = ['1425',
'1465_WashingtonQuelea',
'1733_BandjarmasinKomodoDragon',
'184_WestYorkshireWalrus',
'1867_BucharestTrout',
'1867_GoianiaCrane',
'1871',
'1991_MendozaCow',
'2222_JiutaiChicken',
'2743_HuaianKoi',
'3604_LichuanHookworm',
'3604_ShangquiHare',
'3614_BrisbaneHornet',
'3614_VientianeWhippet',
'3938_YingchengSeaLion',
'4765_NouakchottMoose',
'5644_AkesuCoral',
'5892_LvivRooster',
'5892_NonthaburiHalibut',
'7135_TampicoWallaby',
'8681_NanchangAlbatross',
'8725_ShishouMosquito',
'8725_YangchunCobra']

base_path = "/Users/a_fin/Desktop/Year 4/Project/Meditation/DreemEEG"
epoch_dir_candidates = ["6-rej_epoch", "5-rej_epoch"]
epoch_length  = '4secs'
kernel        = 3
tau           = 8

custom_channel_groups = {
    'fo_chans':  [0, 1, 4, 5],
    'ff_chans':  [2, 3, 6],
    'oo_chans':  [7],
    'glob_chans': list(range(8))
}

# CSV containing sessions per subject
meta_csv = "/Users/a_fin/Desktop/Year 4/Project/Data/Meditation_TET_data.csv"
SUBJECT_COL = "Subject"
SESSION_COL = "Session"

# ------------- load sessions map -------------
if not os.path.exists(meta_csv):
    raise FileNotFoundError(f"CSV not found: {meta_csv}")

df_meta = pd.read_csv(meta_csv)

if SUBJECT_COL not in df_meta.columns or SESSION_COL not in df_meta.columns:
    raise ValueError(f"CSV must contain columns '{SUBJECT_COL}' and '{SESSION_COL}'")

# Normalize sessions to the numeric prefix only (e.g., "688320_TET.mat" -> "688320")
df_meta[SESSION_COL] = (
    df_meta[SESSION_COL]
    .astype(str)
    .str.extract(r'^(\d+)')[0]  # take leading digits
)

# Build mapping: subject -> sorted unique list of sessions
sessions_by_subject = (
    df_meta
    .dropna(subset=[SUBJECT_COL, SESSION_COL])
    .groupby(SUBJECT_COL)[SESSION_COL]
    .apply(lambda s: sorted(pd.unique(s)))
    .to_dict()
)

# — load existing results —
input_path = f'/Users/a_fin/Desktop/Year 4/Project/Data/med_pe_metrics_combined_{tau}'
xls        = pd.ExcelFile(f"{input_path}.xlsx")
df_pe      = xls.parse('PermutationEntropy')
df_gaps    = xls.parse('Gaps')

# ensure the key columns exist
for col in ('subject', 'session', 'run', 'epoch'):
    if col not in df_pe.columns:
        raise ValueError(f"Column '{col}' missing from one of your sheets.")

# build set of already‐processed combos (subject, session)
processed = set(zip(df_pe['subject'], df_pe['session']))

# build the full expected set from the CSV metadata
all_combos = set()
for subject in subjects:
    subj_sessions = sessions_by_subject.get(subject, [])
    for session in subj_sessions:
        all_combos.add((subject, str(session)))

# find which to rerun
to_rerun = sorted(all_combos - processed)
print(f"Need to rerun {len(to_rerun)} combinations.")

new_pe_list, new_gaps_list = [], []

for subject, session in to_rerun:
    session_str = str(session)

    # Candidate filename patterns
    filename_patterns = [
        "{session}_cut_4sec_rej_epoch.set",
        "{session}_hp_4sec_labelled_rej_epoch.set",
    ]

    original_set_path = None
    processed_set_path = None
    chosen_epoch_dir = None
    chosen_filename = None

    # Try both 6-rej_epoch and 5-rej_epoch
    for d in epoch_dir_candidates:
        dir_path = os.path.join(base_path, subject, d)
        if not os.path.isdir(dir_path):
            continue

        # Try known patterns first
        for pat in filename_patterns:
            candidate = os.path.join(dir_path, pat.format(session=session_str))
            if os.path.exists(candidate):
                chosen_epoch_dir = d
                chosen_filename = os.path.basename(candidate)
                original_set_path = candidate
                processed_set_path = candidate
                break

        if original_set_path:
            break

        # Fallback: scan for any file starting with the session id and ending with '_rej_epoch.set'
        try:
            matches = [
                f for f in os.listdir(dir_path)
                if f.startswith(session_str) and f.endswith("_rej_epoch.set")
            ]
        except FileNotFoundError:
            matches = []

        if matches:
            # Prefer 'cut' over 'hp' if multiple
            matches.sort(key=lambda x: (0 if "cut" in x else 1, x))
            chosen = matches[0]
            chosen_epoch_dir = d
            chosen_filename = chosen
            original_set_path = os.path.join(dir_path, chosen)
            processed_set_path = original_set_path
            break

    if not original_set_path:
        print(f"  SKIP missing file for {subject} session {session_str}")
        continue

    pe_res = process_permutation_entropy(
        original_set_path=original_set_path,
        processed_set_path=processed_set_path,
        epoch_length=epoch_length,
        custom_channel_groups=custom_channel_groups,
        kernel=kernel,
        tau=tau
    )
    if pe_res is None:
        print(f"  FAILED for {subject} session {session_str}")
        continue

    dpe  = pe_res['pe']
    dgps = pe_res['gaps']
    for df in (dpe, dgps):
        df['subject'] = subject
        df['session'] = session_str
        df['run']     = session_str

    new_pe_list .append(dpe)
    new_gaps_list.append(dgps)
    print(f"  Completed {subject} session {session_str}")

# if we got anything new, append, sort, and save
if new_pe_list:
    # concatenate
    df_pe   = pd.concat([df_pe]   + new_pe_list,  ignore_index=True)
    df_gaps = pd.concat([df_gaps] + new_gaps_list, ignore_index=True)

    # sort by epoch, subject, session, run
    sort_cols = ['subject', 'session', 'run','epoch']
    df_pe   = df_pe.sort_values(sort_cols, ignore_index=True)

    # write back
    with pd.ExcelWriter(f"{input_path}.xlsx") as writer:
        df_pe  .to_excel(writer, sheet_name='PermutationEntropy', index=False)
        df_gaps.to_excel(writer, sheet_name='Gaps',               index=False)

    print(f"✅ Updated and sorted Excel saved to {input_path}")
else:
    print("✅ No new runs were missing or all reruns failed—no update needed.")
