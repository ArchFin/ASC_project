"""
density_plot.py

Reads a CSV file into a pandas DataFrame and produces a density plot
for the specified column.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # 1. Load your data
    df = pd.read_excel("/Users/a_fin/Desktop/Year 4/Project/Data/pe_metrics_combined_1.xlsx")  

    # 2. Specify which column to plot
    col = "glob_chans"  

    # 3. Create the density plot
    plt.figure(figsize=(8, 6))                     # set figure size
    sns.kdeplot(data=df, x=col, fill=True,         # draw filled KDE
                bw_adjust=1,                     # bandwidth adjustment (1=default)
                clip=(df[col].min(), df[col].max()))  # optional: clip to data range

    # 4. Tidy up and display
    plt.title(f"Density Plot of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()