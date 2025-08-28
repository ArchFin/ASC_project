import os
import math
import pandas as pd
from concog_dreem_lib import process_wsmi

# List of subjects to process
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
kernel = 3
tau = 1
method_params = {"bypass_csd": True}

custom_channel_groups = {
    'fo_chans':   [0, 1, 4, 5],
    'ff_chans':   [2, 3, 6],
    'oo_chans':   [7],
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

# collect every wsmi DataFrame
all_wsmi = []

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
            print(f"Original file not found for {subject} session {session_str}")
            continue
        else:
            print(f"Using '{chosen_epoch_dir}/{chosen_filename}' for {subject} session {session_str}")

        wsmi_results = process_wsmi(
            original_set_path=original_set_path,
            processed_set_path=processed_set_path,
            epoch_length=epoch_length,
            custom_channel_groups=custom_channel_groups,
            kernel=kernel,
            tau=tau,
            method_params=method_params
        )
        if wsmi_results is None:
            print(f"Processing wSMI failed for {subject} session {session_str}")
            continue

        df = wsmi_results['wsmi']
        # add meta so we can group on it
        df['subject'] = subject
        df['session'] = session_str  # keep explicit
        df['run'] = session_str       # optional: keep 'run' for compatibility
        # df already has an 'epoch' column
        all_wsmi.append(df)
        print(f"Processed {subject} session {session_str}")

# concatenate everything
combined = pd.concat(all_wsmi, ignore_index=True)

metrics = [
    'wSMI',
    'SMI'
]

# Group by epoch, subject, session, and run, then take the mean across channels
grouped = (
    combined
    .groupby(['subject', 'session', 'run', 'epoch'], as_index=False)[metrics]
    .mean()
)

# 4. Rename each averaged metric by appending '_glob_chans'
grouped = grouped.rename(
    columns={col: f"{col}_glob_chans" for col in metrics}
)

# write out
out_path = f"/Users/a_fin/Desktop/Year 4/Project/Data/med_wsmi_{tau}_global.xlsx"
grouped.to_excel(out_path, index=False)
print(f"Saved per‐epoch global averages to {out_path}")
