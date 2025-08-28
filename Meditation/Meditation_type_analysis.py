import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import chi2, norm
import warnings

# Load the HMM output CSV
csv_path = '/Users/a_fin/Desktop/Year 4/Project/Data/HMM_output_adjusted_notransitions.csv'
df = pd.read_csv(csv_path)

output_dir = '/Users/a_fin/Desktop/Year 4/Project/Data'


# Cross-frequency analysis: Med_type vs transition_label
if 'Med_type' in df.columns and 'transition_label' in df.columns:
    cross_tab = pd.crosstab(df['Med_type'], df['transition_label'])
    print('\nCross-frequency table (Med_type vs transition_label):')
    print(cross_tab)
    cross_tab_path = os.path.join(output_dir, 'Med_type_vs_transition_label_crosstab_Expert.csv')
    cross_tab.to_csv(cross_tab_path)
    
    # Statistical testing
    print('\n=== STATISTICAL TESTS ===')
    
    # 1. Chi-square test of independence
    try:
        chi2_stat, chi2_p, dof, expected = chi2_contingency(cross_tab)
        print(f'\nChi-square test of independence:')
        print(f'Chi2 statistic: {chi2_stat:.4f}')
        print(f'p-value: {chi2_p:.6f}')
        print(f'Degrees of freedom: {dof}')
        print(f'Expected frequencies minimum: {expected.min():.2f}')
        
        if chi2_p < 0.001:
            print('*** HIGHLY SIGNIFICANT association (p < 0.001) ***')
        elif chi2_p < 0.01:
            print('** SIGNIFICANT association (p < 0.01) **')
        elif chi2_p < 0.05:
            print('* SIGNIFICANT association (p < 0.05) *')
        else:
            print('No significant association (p >= 0.05)')
            
        # Effect size (Cramér's V)
        n = cross_tab.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(cross_tab.shape) - 1)))
        print(f'Cramér\'s V (effect size): {cramers_v:.4f}')
        if cramers_v < 0.1:
            print('  -> Small effect')
        elif cramers_v < 0.3:
            print('  -> Medium effect')
        else:
            print('  -> Large effect')
            
    except ValueError as e:
        print(f'Chi-square test failed: {e}')
    
    # 2. Post-hoc / cell-wise tests using adjusted residuals + multiple-testing correction
    #    This is more appropriate than testing each Med_type against a uniform distribution.
    if chi2_p < 0.05:
        print('\n--- Post-hoc cell-wise comparisons (adjusted residuals) ---')
        obs = cross_tab.values.astype(float)
        exp = expected  # from chi2_contingency
        row_totals = cross_tab.sum(axis=1).values.reshape(-1, 1)
        col_totals = cross_tab.sum(axis=0).values.reshape(1, -1)
        n_total = obs.sum()
        # Haberman's adjusted residuals: (obs - exp) / sqrt( exp * (1 - row_total/n) * (1 - col_total/n) )
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.sqrt(exp * (1 - row_totals / n_total) * (1 - col_totals / n_total))
            adj_resid = (obs - exp) / denom
            adj_resid = np.nan_to_num(adj_resid, nan=0.0, posinf=0.0, neginf=0.0)
        p_vals = 2 * (1 - norm.cdf(np.abs(adj_resid)))
        p_flat = p_vals.flatten()
        try:
            from statsmodels.stats.multitest import multipletests
            _HAS_STATSMODELS = True
        except Exception:
            _HAS_STATSMODELS = False
        if _HAS_STATSMODELS:
            reject, p_adj_flat, _, _ = multipletests(p_flat, alpha=0.05, method='fdr_bh')
            print('Applied Benjamini-Hochberg FDR correction to cell-wise tests.')
        else:
            m = len(p_flat)
            p_adj_flat = np.minimum(p_flat * m, 1.0)
            reject = p_adj_flat < 0.05
            print('statsmodels not installed; applied Bonferroni correction as fallback.')
        med_types = list(cross_tab.index)
        cluster_labels = list(cross_tab.columns)
        rows = []
        for i, med in enumerate(med_types):
            for j, cl in enumerate(cluster_labels):
                p_unc = p_vals[i, j]
                p_adj = p_adj_flat[i * len(cluster_labels) + j]
                z = adj_resid[i, j]
                direction = 'over-represented' if z > 0 else 'under-represented' if z < 0 else 'no-difference'
                sig = '*' if p_adj < 0.05 else ''
                rows.append({
                    'Med_type': med,
                    'cluster_label': cl,
                    'observed': int(obs[i, j]),
                    'expected': float(exp[i, j]),
                    'adjusted_residual': float(z),
                    'p_uncorrected': float(p_unc),
                    'p_adjusted': float(p_adj),
                    'significant': bool(p_adj < 0.05),
                    'direction': direction
                })
                if p_adj < 0.05:
                    print(f'  {med} - Cluster {cl}: {direction} (z = {z:.2f}, p_adj = {p_adj:.4f})')
        cell_stats_df = pd.DataFrame(rows)
        cell_stats_path = os.path.join(output_dir, 'Med_type_vs_cluster_cellwise_stats_Expert.csv')
        cell_stats_df.to_csv(cell_stats_path, index=False)
        print(f'Cell-wise results saved to: {cell_stats_path}')
    else:
        print('Overall test not significant: skipping cell-wise post-hoc tests.')
    # 3. Save statistical results
    stats_results = {
        'test': 'Chi-square independence',
        'chi2_statistic': chi2_stat,
        'p_value': chi2_p,
        'degrees_of_freedom': dof,
        'cramers_v': cramers_v,
        'interpretation': 'Significant association' if chi2_p < 0.05 else 'No significant association',
        'effect_size_interpretation': 'Small' if cramers_v < 0.1 else ('Medium' if cramers_v < 0.3 else 'Large')
    }
    
    stats_df = pd.DataFrame([stats_results])
    stats_path = os.path.join(output_dir, 'Med_type_vs_transition_label_statistics_Expert.csv')
    stats_df.to_csv(stats_path, index=False)
    
    # Optional: plot as heatmap
    plt.figure(figsize=(10, 6))
    plt.title(f'Meditation type vs Cluster label Frequency\n(χ² = {chi2_stat:.2f}, p = {chi2_p:.4f}, Cramér\'s V = {cramers_v:.3f})')
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Meditation type')
    plt.xlabel('Cluster label')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'Meditation_type_vs_cluster_label_heatmap_Expert.png')
    plt.savefig(heatmap_path)
    plt.close()
    # Relative (row-normalized) heatmap
    cross_tab_rel = cross_tab.div(cross_tab.sum(axis=1), axis=0)
    plt.figure(figsize=(10, 6))
    plt.title(f'Med_type vs transition_label (Relative Frequency)\n(χ² = {chi2_stat:.2f}, p = {chi2_p:.4f})')
    sns.heatmap(cross_tab_rel, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('Med_type')
    plt.xlabel('transition_label')
    plt.tight_layout()
    rel_heatmap_path = os.path.join(output_dir, 'Med_type_vs_transition_label_heatmap_relative_Expert.png')
    plt.savefig(rel_heatmap_path)
    plt.close()
    # Save relative crosstab to CSV
    cross_tab_rel_path = os.path.join(output_dir, 'Med_type_vs_transition_label_crosstab_relative_Expert.csv')
    cross_tab_rel.to_csv(cross_tab_rel_path)
    # Heatmap of adjusted residuals (helps show which cells drive the effect)
    try:
        plt.figure(figsize=(10, 6))
        if 'adj_resid' in locals():
            df_adj = pd.DataFrame(adj_resid, index=cross_tab.index, columns=cross_tab.columns)
            sns.heatmap(df_adj, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
            plt.title('Adjusted residuals (Haberman) — positive = over-represented')
            plt.ylabel('Meditation type')
            plt.xlabel('Cluster Label')
            plt.tight_layout()
            resid_heatmap_path = os.path.join(output_dir, 'Med_type_cluster_adjusted_residuals_heatmap_Expert.png')
            plt.savefig(resid_heatmap_path)
            plt.close()
    except Exception as e:
        print(f'Could not produce adjusted residuals heatmap: {e}')
else:
    print("Both 'Med_type' and 'transition_label' columns are required for cross-frequency analysis.")