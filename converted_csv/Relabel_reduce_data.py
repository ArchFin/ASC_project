import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
INPUT_CSV = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/converted_csv/combined_all_subjects.csv'
OUTPUT_CSV = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/converted_csv/combined_all_subjects_re.csv'
TARGET_EPOCH_SECONDS = 60
EXPECTED_SESSION_SECONDS = 900  # used only as guidance if needed

# Read and tidy
df = pd.read_csv(INPUT_CSV)

# Drop unwanted columns and rename
df = df.drop(columns=['session_type', 'source_file', 'subject_id'], errors='ignore')
df = df.rename(columns={'filename': 'Session', 'subject_directory': 'Subject'})

# Ensure Subject and Session exist
if 'Subject' not in df.columns or 'Session' not in df.columns:
    raise KeyError('Input CSV must contain Subject and Session columns')

# Reindex to preserve original order per session
# If you have a time column, you may want to sort by it here.

# Helper: epoch and average a single session group
def epoch_and_average_group(group, epoch_seconds=TARGET_EPOCH_SECONDS, session_seconds=EXPECTED_SESSION_SECONDS):
    n = len(group)
    if n == 0:
        return pd.DataFrame()

    # Infer sampling rate (samples per second) from observed points and expected session length
    if session_seconds and session_seconds > 0:
        sampling_rate = n / float(session_seconds)  # samples per second
    else:
        sampling_rate = 1.0  # fallback: assume 1 sample/sec

    # Compute how many points should go into each epoch (rounded, at least 1)
    points_per_epoch = max(1, int(round(epoch_seconds * sampling_rate)))

    # Number of epochs (ceil to cover all samples)
    epoch_count = int(np.ceil(n / points_per_epoch))

    # Create contiguous chunks of indices of length points_per_epoch
    indices = group.index.to_numpy()
    rows = []
    for i in range(0, n, points_per_epoch):
        idxs = indices[i:i + points_per_epoch]
        chunk = group.loc[idxs]
        # Compute mean for numeric columns
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
        means = chunk[numeric_cols].mean()
        row = means.to_dict()
        # Add metadata
        row['Subject'] = chunk['Subject'].iloc[0]
        row['Session'] = chunk['Session'].iloc[0]
        row['epoch'] = (i // points_per_epoch) + 1
        row['epoch_size'] = len(idxs)
        rows.append(row)

    return pd.DataFrame(rows)

# Process all sessions
out_frames = []
for (sub, sess), grp in df.groupby(['Subject', 'Session'], sort=False):
    grp = grp.copy()
    # If there's a timestamp column, sort here (e.g., grp.sort_values('time', inplace=True))
    averaged = epoch_and_average_group(grp, TARGET_EPOCH_SECONDS)
    out_frames.append(averaged)

if out_frames:
    df_epoched = pd.concat(out_frames, ignore_index=True)
    # Order columns: Subject, Session, epoch, epoch_size, then features
    cols = ['Subject', 'Session', 'epoch', 'epoch_size'] + [c for c in df_epoched.columns if c not in ['Subject','Session','epoch','epoch_size']]
    df_epoched = df_epoched[cols]
    # Save
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_epoched.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved epoch-averaged CSV: {OUTPUT_CSV} (rows: {len(df_epoched)})")
else:
    print('No data to process')
