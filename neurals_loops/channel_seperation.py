import pandas as pd

def pivot_channel_groups_allow_duplicates(
    df: pd.DataFrame,
    index_cols=('epoch', 'subject', 'week', 'run'),
    channel_col='channel_group',
    value_col='PE',
    drop_helper=True
) -> pd.DataFrame:
    """
    Pivot a DataFrame so that each channel_group becomes its own column,
    but *keep* every row even if you had duplicate (index,channel_group)
    combinations.
    """
    # 1) make a copy so we don’t clobber the original
    df2 = df.copy()

    # 2) within each group of (index_cols + channel_col), tag duplicates 0,1,2,...
    helper = '_dup_id'
    df2[helper] = (
        df2
        .groupby(list(index_cols) + [channel_col])
        .cumcount()
    )

    # 3) pivot including the helper in the index
    wide = (
        df2
        .pivot(
            index=list(index_cols) + [helper],
            columns=channel_col,
            values=value_col
        )
        .reset_index()
    )

    # 4) clean up
    wide.columns.name = None     # drop the "channel_group" name
    if drop_helper:
        wide = wide.drop(columns=[helper])

    return wide


if __name__ == "__main__":
    # --- 1) Load your data ---
    # Replace the path with your actual Excel file location
    df = pd.read_excel(
        '/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_8.xlsx'
    )

    # --- 2) Pivot, keeping all duplicates ---
    wide = pivot_channel_groups_allow_duplicates(df)

    # --- 3) Sort by subject, week, run, then epoch ---
    wide = wide.sort_values(by=['subject', 'week', 'run', 'epoch'])

    # (Optional) Inspect the first few rows
    print(wide.head())

    # --- 4) Save the result ---
    output_path = (
        '/Users/a_fin/Desktop/Year 4/Project/Data/'
        'pe_metrics_combined_8_wide.xlsx'
    )
    wide.to_excel(output_path, index=False)

    print(f"Wide-format data saved to '{output_path}'")
