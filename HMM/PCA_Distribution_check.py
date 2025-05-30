import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# --------------------------------------------------------------------
# Adjust this path to point to your local CSV file:
csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted.csv'
# --------------------------------------------------------------------

# 1) Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# 2) Specify the two principal‐component columns
pc_columns = ['principal component 1', 'principal component 2']

for pc in pc_columns:
    # 3) Extract non‐null values of this principal component
    data = df[pc].dropna().values

    # 4) Fit a Normal (Gaussian) distribution: returns (mu, sigma)
    mu, sigma = norm.fit(data)

    # 5) Fit a Student’s t‐distribution: returns (df, loc, scale)
    df_t, loc_t, scale_t = t.fit(data)

    # 6) Create an array of x‐values spanning the data range
    x_min = data.min() - (data.std() * 0.5)
    x_max = data.max() + (data.std() * 0.5)
    x = np.linspace(x_min, x_max, 300)

    # 7) Compute the fitted PDFs on the x‐grid
    pdf_norm = norm.pdf(x, loc=mu, scale=sigma)
    pdf_t    = t.pdf(x, df_t, loc=loc_t, scale=scale_t)

    # 8) Plot histogram of the raw data (density=True normalizes to a PDF)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black', label='Data histogram')

    # 9) Overlay the fitted Normal and Student’s t curves
    plt.plot(x, pdf_norm,
             lw=2,
             label=f'Normal fit  (μ={mu:.3f},  σ={sigma:.3f})')
    plt.plot(x, pdf_t,
             lw=2,
             linestyle='--',
             label=f"Student’s t fit  (df={df_t:.2f},  loc={loc_t:.3f},  scale={scale_t:.3f})")

    # 10) Add labels, title, and legend
    plt.title(f'Distribution of "{pc}" with Fitted PDFs')
    plt.xlabel(pc)
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # 11) Show (or save) the figure
    plt.savefig(f"/Users/a_fin/Desktop/Year 4/Project/Data/{pc}_fit_comparison.png", dpi=300)