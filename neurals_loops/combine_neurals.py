import pandas as pd
import os
from functools import reduce

def merge_excel_files(file_paths, output_csv,
                      key_cols=('epoch', 'week', 'subject', 'run'),
                      how='outer'):
    dfs = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"⚠️  File not found: {fp}")
            continue
        try:
            df = pd.read_excel(fp, engine='openpyxl')
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            continue
        
        basename = os.path.splitext(os.path.basename(fp))[0]
        other_cols = [c for c in df.columns if c not in key_cols]
        rename_map = {c: f"{basename}__{c}" for c in other_cols}
        df = df.rename(columns=rename_map)
        
        dfs.append(df)
        print(f"Loaded {fp} ({df.shape[0]} rows, {df.shape[1]} cols)")

    if not dfs:
        raise ValueError("No valid DataFrames were loaded. Check your file paths.")

    merged = reduce(
        lambda left, right: pd.merge(left, right, on=key_cols, how=how),
        dfs
    )

    merged.to_csv(output_csv, index=False)
    print(f"\nMerged {len(dfs)} files → {merged.shape[0]} rows × {merged.shape[1]} cols")
    print(f"Output written to: {output_csv}")


if __name__ == "__main__":
    files_to_merge = [
        r'/Users/a_fin/Desktop/Year 4/Project/Data/lz_metrics_combined.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_1_wide.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_2_wide.xlsx', 
        r'/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_4_wide.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_8_wide.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/psd_metrics_combined_avg.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_1_global_2.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_2_global_2.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_4_global_2.xlsx',
        r'/Users/a_fin/Desktop/Year 4/Project/Data/wsmi_8_global_2.xlsx',
    ]

    merge_excel_files(
        file_paths=files_to_merge,
        output_csv=r'/Users/a_fin/Desktop/Year 4/Project/Data/total_neurals.csv',
        key_cols=('epoch', 'week', 'subject', 'run'),
        how='outer'  # or 'inner' if you only want rows common to all files
    )
