import os
import pandas as pd
from concog_dreem_lib import process_LZ78

# List of subjects, weeks, and runs to process
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


# Base directory for the data files
base_path = "/Users/a_fin/Desktop/Year 4/Project/Meditation/DreemEEG"
epoch_dir_candidates = ["6-rej_epoch", "5-rej_epoch"]

epoch_length = '4secs'
custom_channel_groups = {
    'fo_chans': [0, 1, 4, 5],
    'ff_chans': [2, 3, 6],
    'oo_chans': [7],
    'glob_chans': list(range(8))
}

# CSV containing sessions per subject
meta_csv = "/Users/a_fin/Desktop/Year 4/Project/Data/Meditation_TET_data.csv"
SUBJECT_COL = "Subject"   # <-- change if needed (e.g., "SubjectID")
SESSION_COL = "Session"   # <-- change if needed (e.g., "Session")

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

# ------------- processing loop -------------
results_list = []

for subject in subjects:
    subj_sessions = sessions_by_subject.get(subject, [])
    if not subj_sessions:
        print(f"No sessions listed in CSV for subject {subject}; skipping.")
        continue
    for session in subj_sessions:
            # Ensure session is a string (filenames may expect numbers or strings)
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
                tried = []
                for d in epoch_dir_candidates:
                    for pat in filename_patterns:
                        tried.append(os.path.join(base_path, subject, d, pat.format(session=session_str)))
                print(f"Original file not found for {subject} session {session_str}. Tried patterns: {tried}")
                continue
            else:
                print(f"Using '{chosen_epoch_dir}/{chosen_filename}' for {subject} session {session_str}")

            lz_results = process_LZ78(
                original_set_path=original_set_path,
                processed_set_path=processed_set_path,
                epoch_length=epoch_length,
                custom_channel_groups=custom_channel_groups
            )

            if lz_results is None:
                print(f"Processing LZ78 failed for {subject} session {session_str}")
                continue

            df_lzc = lz_results['lz_c']
            df_lzsum = lz_results['lz_sum']
            df_gaps = lz_results['gaps']

            # Add metadata
            for df in (df_lzc, df_lzsum, df_gaps):
                df['subject'] = subject
                df['session'] = session_str  # keep explicit
                df['run'] = session_str       # optional: keep 'run' for compatibility

            results_list.append({
                'Subject': subject,
                'Session': session_str,
                'lz_c': df_lzc,
                'lz_sum': df_lzsum,
                'gaps': df_gaps
            })

            print(f"Processed {subject} session {session_str}")

# ------------- save combined outputs -------------
if results_list:
    combined_df_lzc = pd.concat([res['lz_c'] for res in results_list], ignore_index=True)
    combined_df_lzsum = pd.concat([res['lz_sum'] for res in results_list], ignore_index=True)
    combined_df_gaps = pd.concat([res['gaps'] for res in results_list], ignore_index=True)

    output_path = '/Users/a_fin/Desktop/Year 4/Project/Data/med_lz_metrics_combined.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        combined_df_lzc.to_excel(writer, sheet_name='LZc', index=False)
        combined_df_lzsum.to_excel(writer, sheet_name='LZsum', index=False)
        combined_df_gaps.to_excel(writer, sheet_name='Gaps', index=False)

    print(f"\nCombined results have been saved to {output_path}")
else:
    print("No valid LZ78 processing results were obtained.")