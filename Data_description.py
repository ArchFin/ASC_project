import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from HMM.HMM_methods import csv_splitter  # reuse your loader

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_config(cfg_path="Breathwork.yaml"):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def basic_overview(df):
    overview = {}
    overview['n_rows'] = int(df.shape[0])
    overview['n_columns'] = int(df.shape[1])
    overview['columns'] = list(df.columns)
    # group keys heuristics
    for key in ['Subject','Week','Session','Med_type']:
        if key in df.columns:
            overview['has_' + key.lower()] = True
    return overview

def missingness_table(df):
    miss = df.isna().sum().to_dict()
    miss_pct = (df.isna().mean() * 100).round(2).to_dict()
    return {'missing_counts': miss, 'missing_pct': miss_pct}

def feature_summary(df, feature_cols):
    # defensive: if no features, return empty summary
    if not feature_cols:
        return {'descriptive': {}, 'skew': {}, 'kurtosis': {}}

    # Use describe() to get standard percentiles (25%, 50%, 75%) reliably
    desc_df = df[feature_cols].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    # compute skewness and kurtosis separately
    skew = df[feature_cols].skew().to_dict()
    kurt = df[feature_cols].kurtosis().to_dict()
    return {'descriptive': desc_df, 'skew': skew, 'kurtosis': kurt}

def per_subject_stats(df, subject_col='Subject'):
    if subject_col not in df.columns:
        return {}
    counts = df.groupby(subject_col).size()
    return {
        'n_subjects': int(counts.shape[0]),
        'per_subject_counts': counts.describe().to_dict(),
        'per_subject_counts_full': counts.to_dict()
    }

def per_week_stats(df, subject_col='Subject', week_col='Week', session_col='Session', time_per_sample_sec=28):
    """Return list of per-week summaries (n_subjects, n_sessions, n_samples, total_time_min)."""
    if week_col not in df.columns:
        return []
    weeks = []
    grouped = df.groupby(week_col)
    for wk, g in grouped:
        n_subjects = int(g[subject_col].nunique()) if subject_col in g.columns else None
        # count unique sessions within week (subject+session pairs)
        if subject_col in g.columns and session_col in g.columns:
            n_sessions = int(g[[subject_col, session_col]].drop_duplicates().shape[0])
        elif session_col in g.columns:
            n_sessions = int(g[session_col].nunique())
        else:
            n_sessions = int(g.shape[0])
        n_samples = int(g.shape[0])
        total_time_min = round(n_samples * time_per_sample_sec / 60.0, 2)
        weeks.append({'week': wk, 'n_subjects': n_subjects, 'n_sessions': n_sessions,
                      'n_samples': n_samples, 'total_time_min': total_time_min})
    return weeks

def per_medtype_stats(df, medtype_col='Med_type', subject_col='Subject', session_col='Session', time_per_sample_sec=28):
    """Return list of per-medtype summaries (n_subjects, n_sessions, n_samples, total_time_min)."""
    if medtype_col not in df.columns:
        return []
    medtypes = []
    grouped = df.groupby(medtype_col)
    for mt, g in grouped:
        n_subjects = int(g[subject_col].nunique()) if subject_col in g.columns else None
        if subject_col in g.columns and session_col in g.columns:
            n_sessions = int(g[[subject_col, session_col]].drop_duplicates().shape[0])
        elif session_col in g.columns:
            n_sessions = int(g[session_col].nunique())
        else:
            n_sessions = int(g.shape[0])
        n_samples = int(g.shape[0])
        total_time_min = round(n_samples * time_per_sample_sec / 60.0, 2)
        medtypes.append({'medtype': mt, 'n_subjects': n_subjects, 'n_sessions': n_sessions,
                         'n_samples': n_samples, 'total_time_min': total_time_min})
    return medtypes


def generate_latex_table(summary, week_stats, medtype_stats, outdir, filename='dataset_description_table.tex', caption='Dataset description'):
    """Create a simple LaTeX table summarising overall metrics and per-week/medtype rows and save to file."""
    ensure_dir(outdir)

    # Overall metrics
    overall = summary.get('overview', {})
    n_samples = overall.get('n_rows', '')
    n_columns = overall.get('n_columns', '')
    n_subjects = summary.get('per_subject', {}).get('n_subjects', '')
    n_features = len(summary.get('feature_summary', {}).get('descriptive', {}))
    total_time_min = ''
    if 'total_time_min' in summary:
        total_time_min = summary['total_time_min']

    # Build LaTeX
    lines = []
    lines.append('% Auto-generated dataset description table')
    lines.append('\begin{table}[htbp]')
    lines.append('\centering')
    lines.append(f'\caption{{{caption}}}')
    lines.append('\begin{tabular}{lr}')
    lines.append('\hline')
    lines.append('Metric & Value\\')
    lines.append('\hline')
    lines.append(f'Number of samples & {n_samples}\\')
    lines.append(f'Number of features & {n_features}\\')
    lines.append(f'Number of subjects & {n_subjects}\\')
    lines.append(f'Number of columns & {n_columns}\\')
    if total_time_min != '':
        lines.append(f'Total recording time (min) & {total_time_min}\\')
    lines.append('\hline')
    lines.append('\end{tabular}')

    # Add per-week table below if available
    if week_stats:
        lines.append('\vspace{1em}')
        lines.append('\begin{tabular}{lrrr}')
        lines.append('\hline')
        lines.append('Week & Subjects & Sessions & Time (min)\\')
        lines.append('\hline')
        for wk in week_stats:
            lines.append(f"{wk['week']} & {wk['n_subjects']} & {wk['n_sessions']} & {wk['total_time_min']}\\")
        lines.append('\hline')
        lines.append('\end{tabular}')

    # Add per-medtype table below if available
    if medtype_stats:
        lines.append('\vspace{1em}')
        lines.append('\begin{tabular}{lrrr}')
        lines.append('\hline')
        lines.append('Med\_type & Subjects & Sessions & Time (min)\\')
        lines.append('\hline')
        for mt in medtype_stats:
            # escape percent signs or special chars if necessary
            lines.append(f"{mt['medtype']} & {mt['n_subjects']} & {mt['n_sessions']} & {mt['total_time_min']}\\")
        lines.append('\hline')
        lines.append('\end{tabular}')

    lines.append('\label{tab:dataset_description}')
    lines.append('\end{table}')

    tex_path = os.path.join(outdir, filename)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    return tex_path

def plot_feature_distributions(df, feature_cols, outdir):
    ensure_dir(outdir)
    # hist + KDE grid
    n = len(feature_cols)
    cols = 3
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(4*cols, 3*rows))
    for i, f in enumerate(feature_cols):
        ax = plt.subplot(rows, cols, i+1)
        sns.histplot(df[f].dropna(), kde=True, ax=ax, color='C0')
        ax.set_title(f)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'feature_histograms.png'), dpi=200)
    plt.close()
    # boxplots
    plt.figure(figsize=(max(10, len(feature_cols)*0.4), 6))
    sns.boxplot(data=df[feature_cols].melt(var_name='feature', value_name='value'),
                x='feature', y='value')
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature boxplots')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'feature_boxplots.png'), dpi=200)
    plt.close()

def plot_correlation(df, feature_cols, outdir):
    ensure_dir(outdir)
    corr = df[feature_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='RdBu_r', center=0, annot=False, fmt='.2f')
    plt.title('Feature correlation (Pearson)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'feature_correlation.png'), dpi=200)
    plt.close()
    return corr

def plot_pca_variance(df, feature_cols, outdir, n_components=10):
    ensure_dir(outdir)
    X = df[feature_cols].dropna().values
    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X)
    var = pca.explained_variance_ratio_
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(var)+1), np.cumsum(var), marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'pca_explained_variance.png'), dpi=200)
    plt.close()
    return var

def class_balance_plots(df, label_col='labels', outdir='.'):
    ensure_dir(outdir)
    if label_col not in df.columns:
        return None
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x=[f'C{i+1}' for i in counts.index], y=counts.values, palette='tab10')
    plt.ylabel('Count')
    plt.title('Class balance')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'class_balance.png'), dpi=200)
    plt.close()
    return counts.to_dict()

def example_time_series(df, feature_cols, outdir, n_examples=3):
    ensure_dir(outdir)
    # Pick n_examples subjects / sessions
    if 'Subject' in df.columns and 'Session' in df.columns:
        groups = df.groupby(['Subject','Session'])
    elif 'Subject' in df.columns:
        groups = df.groupby('Subject')
    else:
        groups = [('all', df)]
    examples = list(groups)[:n_examples]
    for ix, (name, g) in enumerate(examples):
        plt.figure(figsize=(12,4))
        for f in feature_cols[:6]:  # show first 6 features to avoid clutter
            plt.plot(g[f].values, label=f)
        plt.legend(ncol=2)
        plt.title(f'Example time series: {name}')
        plt.tight_layout()
        fname = f'example_timeseries_{ix+1}.png'
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()

def save_summary_report(summary, outdir):
    ensure_dir(outdir)
    with open(os.path.join(outdir, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    # human-readable txt
    with open(os.path.join(outdir, 'dataset_summary.txt'), 'w') as f:
        for k,v in summary.items():
            f.write(f"{k}:\n{v}\n\n")

def run_description(cfg_path="Meditation.yaml", outdir=None):
    cfg = load_config(cfg_path)
    if outdir is None:
        outdir = cfg.get('savelocation_TET','./results_dataset_description')
    ensure_dir(outdir)
    # load data via csv_splitter (keeps your existing file parsing)
    reader = csv_splitter(cfg['filelocation_TET'])
    df = reader.read_CSV()
    if df is None:
        raise RuntimeError("Failed to read CSV via csv_splitter.")
    # determine feature columns (feelings in config preferred)
    if 'feelings' in cfg:
        features = [f for f in cfg['feelings'] if f in df.columns]
    else:
        # fallback: all numeric cols except known grouping/label cols
        exclude = {'Subject','Session','Week','Med_type','labels'}
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    summary = {}
    summary['overview'] = basic_overview(df)
    summary['missingness'] = missingness_table(df)
    summary['feature_summary'] = feature_summary(df, features)
    summary['per_subject'] = per_subject_stats(df, subject_col='Subject')

    # compute total time (use config time step if provided)
    time_per_sample = cfg.get('time_step_seconds', cfg.get('time_jump', 28))
    total_time_min = round(df.shape[0] * time_per_sample / 60.0, 2)
    summary['total_time_min'] = total_time_min

    # per-week summaries
    week_stats = per_week_stats(df, subject_col='Subject', week_col='Week', session_col='Session', time_per_sample_sec=time_per_sample)
    summary['week_stats'] = week_stats

    # per-medtype summaries
    medtype_stats = per_medtype_stats(df, medtype_col='Med_type', subject_col='Subject', session_col='Session', time_per_sample_sec=time_per_sample)
    summary['medtype_stats'] = medtype_stats

    # plots and matrices
    plot_feature_distributions(df, features, outdir)
    corr = plot_correlation(df, features, outdir)
    pca_var = plot_pca_variance(df, features, outdir, n_components=min(10, len(features)))
    class_counts = class_balance_plots(df, label_col='labels', outdir=outdir)
    example_time_series(df, features, outdir, n_examples=3)
    summary['correlation_rank'] = corr.abs().sum().sort_values(ascending=False).to_dict()
    summary['pca_variance'] = pca_var.tolist()
    if class_counts is not None:
        summary['class_counts'] = class_counts

    # produce LaTeX table for paper
    tex_path = generate_latex_table(summary, week_stats, medtype_stats, outdir)
    summary['latex_table'] = tex_path

    save_summary_report(summary, outdir)
    print(f"Saved dataset description and plots to {outdir}")
    return summary

if __name__ == "__main__":
    run_description()