"""Plots for split-half analysis:
- Bar plots per metric group with 95% CI (raw + Spearman-Brown corrected), one per level_type.
- Scatter plots of half A vs half B for the first 10 iterations per metric.
- Distribution plots (histograms) of r across iterations.
"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Correlations.analysis.split_half.analysis import LEVEL_TYPES
from src.Correlations.analysis.split_half.data import text_id_cols
from src.Correlations.define_cols import (
    MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3,
)

METRIC_GROUPS = {
    'main':         MAIN_RT_COLS,
    'sm_fixation':  SM_RT_COLS_SET1,
    'sm_firstpass': SM_RT_COLS_SET2,
    'sm_late':      SM_RT_COLS_SET3,
}

METRIC_LABELS = {
    'mean_nonzero_TF': 'TF',
    'SkipRateTotal': 'SR',
    'RegRateTotal': 'RR',
    'mean_nonzero_FF': 'FF',
    'mean_FD': 'FD',
    'mean_NF': 'NF',
    'mean_FirstPassGD': 'First-Pass GD',
    'SkipRateFirstPass': 'First-Pass Skip Rate',
    'RegRateFirstPass': 'First-Pass Reg Rate',
    'mean_GD': 'GD',
    'mean_HigherPassFixation': 'Higher-Pass Fix',
    'reading_speed': 'Reading Speed',
}


# ---------------- Bar plots ----------------

def _err_from_ci(mean: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    lower = np.clip(mean - ci_low, 0, None)
    upper = np.clip(ci_high - mean, 0, None)
    return np.vstack([lower, upper])


def _annotate_bars(ax, bars, values, ci_lows=None, ci_highs=None, fontsize=6):
    """Add mean-r and CI bounds text above each bar (above the error bar top, on a white background)."""
    for i, (bar, v) in enumerate(zip(bars, values)):
        if np.isnan(v):
            continue
        lines = [f'{v:.2f}']
        top = max(v, 0)
        if ci_lows is not None and ci_highs is not None:
            lo, hi = ci_lows[i], ci_highs[i]
            if not (np.isnan(lo) or np.isnan(hi)):
                lines.append(f'[{lo:.2f},{hi:.2f}]')
                top = max(top, hi)
        ax.text(bar.get_x() + bar.get_width() / 2, top + 0.02, '\n'.join(lines),
                ha='center', va='bottom', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))


CI_METHODS = ['none', 'percentile', 'tstd']
CI_METHOD_LABELS = {
    'none': 'no CI',
    'percentile': '95% percentile CI',
    'tstd': '95% t-based CI (mean ± t·std)',
}


RAW_COLOR = '#A6CEE3'    # light blue
SB_COLOR  = '#1F78B4'    # dark blue


def plot_bar_group(
    summary_df: pd.DataFrame,
    group_name: str,
    metrics: List[str],
    level_type: str,
    corr_kind: str,
    out_path: Path,
    ci_method: str = 'percentile',
):
    """One PDF per (group, level_type, corr_kind, ci_method). 1 col × 2 rows (resolutions).
    Bars: raw (light blue) and Spearman-Brown (dark blue). Optional 95% CI error bars.

    ci_method: 'none' (no error bars), 'percentile' (ci_low/high cols), 'tstd' (ci_*_tstd cols).
    """
    resolutions = ['paragraph', 'sentence']

    if ci_method == 'percentile':
        low_col, high_col, sb_low_col, sb_high_col = 'ci_low', 'ci_high', 'sb_ci_low', 'sb_ci_high'
    elif ci_method == 'tstd':
        low_col, high_col, sb_low_col, sb_high_col = 'ci_low_tstd', 'ci_high_tstd', 'sb_ci_low_tstd', 'sb_ci_high_tstd'
    elif ci_method == 'none':
        low_col = high_col = sb_low_col = sb_high_col = None
    else:
        raise ValueError(f"unknown ci_method: {ci_method}")

    if ci_method == 'none':
        figsize = (max(7, 2.2 * len(metrics)), 3.8)
        annot_fontsize = 9
    else:
        figsize = (max(12, 3.6 * len(metrics)), 5)
        annot_fontsize = 6

    fig, axes = plt.subplots(
        nrows=1, ncols=len(resolutions),
        figsize=figsize,
        squeeze=False,
    )

    x = np.arange(len(metrics))
    width = 0.38

    for ri, resolution in enumerate(resolutions):
        ax = axes[0, ri]
        sub = summary_df[
            (summary_df['resolution'] == resolution)
            & (summary_df['corr_kind'] == corr_kind)
            & (summary_df['level_type'] == level_type)
            & (summary_df['metric'].isin(metrics))
        ].set_index('metric').reindex(metrics)

        raw_mean = sub['mean_r'].to_numpy(float)
        sb_mean = sub['sb_mean_r'].to_numpy(float)
        if ci_method == 'none':
            raw_err = None
            sb_err = None
        else:
            raw_err = _err_from_ci(raw_mean, sub[low_col].to_numpy(float), sub[high_col].to_numpy(float))
            sb_err = _err_from_ci(sb_mean, sub[sb_low_col].to_numpy(float), sub[sb_high_col].to_numpy(float))

        raw_label = f'{corr_kind.capitalize()} r'
        raw_bars = ax.bar(x - width/2, raw_mean, width, yerr=raw_err, capsize=3,
                          label=raw_label, color=RAW_COLOR, alpha=0.95)
        sb_bars = ax.bar(x + width/2, sb_mean, width, yerr=sb_err, capsize=3,
                         label='Spearman-Brown', color=SB_COLOR, alpha=0.95)
        if ci_method == 'none':
            _annotate_bars(ax, raw_bars, raw_mean, fontsize=annot_fontsize)
            _annotate_bars(ax, sb_bars, sb_mean, fontsize=annot_fontsize)
        else:
            _annotate_bars(ax, raw_bars, raw_mean,
                           sub[low_col].to_numpy(float), sub[high_col].to_numpy(float),
                           fontsize=annot_fontsize)
            _annotate_bars(ax, sb_bars, sb_mean,
                           sub[sb_low_col].to_numpy(float), sub[sb_high_col].to_numpy(float),
                           fontsize=annot_fontsize)

        prefix = 'Δ' if level_type == 'diff' else ''
        labels = [f'{prefix}{METRIC_LABELS.get(m, m)}' for m in metrics]
        rotation = 0 if max(len(l) for l in labels) <= 5 else 30
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation, ha='center' if rotation == 0 else 'right')
        if ci_method == 'none':
            ax.set_ylim(0, 1.15)
            ax.set_title(resolution.capitalize())
        else:
            ax.set_ylim(-0.5, 1.35)
            ax.set_ylabel(f'{corr_kind.capitalize()} r')
            ax.set_title(f'{resolution.capitalize()} | {corr_kind.capitalize()}')
        ax.grid(axis='y', alpha=0.3)

    # Shared legend below the two subplots.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), frameon=False)

    if ci_method != 'none':
        n_iter = int(summary_df['n_iter'].max()) if 'n_iter' in summary_df.columns else 0
        fig.suptitle(
            f'Split-half {level_type} | {group_name} metrics | {corr_kind} | {CI_METHOD_LABELS[ci_method]} | N={n_iter}',
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    else:
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_bars(summary_df: pd.DataFrame, plots_root: Path):
    """Write bar plots to three subfolders under plots_root: no_ci/, percentile_ci/, tstd_ci/.
    Separate PDFs per (group, level_type, corr_kind)."""
    for ci_method in CI_METHODS:
        out_dir = plots_root / f'{ci_method}_ci' if ci_method != 'none' else plots_root / 'no_ci'
        out_dir.mkdir(parents=True, exist_ok=True)
        for group_name, metrics in METRIC_GROUPS.items():
            present = [m for m in metrics if m in summary_df['metric'].unique()]
            if not present:
                continue
            for level_type in LEVEL_TYPES:
                for corr_kind in ['pearson', 'spearman']:
                    out_path = out_dir / f'bars_{group_name}_{level_type}_{corr_kind}.pdf'
                    plot_bar_group(summary_df, group_name, present, level_type, corr_kind, out_path, ci_method=ci_method)


# ---------------- Scatter plots ----------------

def _halves_ab_for_scatter(sub_iter: pd.DataFrame, level_type: str, tic: List[str]) -> pd.DataFrame:
    """Reshape scatter_df rows for one iter/metric/resolution into [A, B] pairs per text (or per text+level for Adv+Ele)."""
    if level_type == 'Adv+Ele':
        long = pd.concat([
            sub_iter[tic + ['half', 'Adv']].rename(columns={'Adv': 'val'}).assign(level='Adv'),
            sub_iter[tic + ['half', 'Ele']].rename(columns={'Ele': 'val'}).assign(level='Ele'),
        ], ignore_index=True)
        wide = long.pivot_table(index=tic + ['level'], columns='half', values='val').reset_index()
    else:
        wide = sub_iter.pivot_table(index=tic, columns='half', values=level_type).reset_index()
    return wide.dropna(subset=['A', 'B'])


def plot_scatter_group(
    scatter_df: pd.DataFrame,
    group_name: str,
    metrics: List[str],
    resolution: str,
    level_type: str,
    out_path: Path,
    n_iters: int = 10,
):
    """Grid of scatters for a metric group, one (resolution, level_type).

    Layout: rows = iterations, cols = metrics.
    """
    tic = text_id_cols(resolution)
    sub_group = scatter_df[
        (scatter_df['resolution'] == resolution) & (scatter_df['metric'].isin(metrics))
    ]
    metrics_present = [m for m in metrics if m in sub_group['metric'].unique()]
    if not metrics_present:
        return

    iters = sorted(sub_group['iter'].unique())[:n_iters]
    nrows = len(iters)
    ncols = len(metrics_present)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)

    for c, metric in enumerate(metrics_present):
        for r, it in enumerate(iters):
            ax = axes[r, c]
            sub_iter = sub_group[(sub_group['metric'] == metric) & (sub_group['iter'] == it)]
            wide = _halves_ab_for_scatter(sub_iter, level_type, tic)
            if len(wide) < 2:
                ax.set_visible(False)
                continue

            if level_type == 'Adv+Ele':
                for lvl, color in [('Adv', '#E45756'), ('Ele', '#4C78A8')]:
                    w = wide[wide['level'] == lvl]
                    ax.scatter(w['A'], w['B'], s=8, alpha=0.5, color=color, label=lvl)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7, loc='lower right')
            else:
                ax.scatter(wide['A'], wide['B'], s=8, alpha=0.5, color='#4C78A8')

            a = wide['A'].to_numpy()
            b = wide['B'].to_numpy()
            r_val = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
            lo = float(min(a.min(), b.min()))
            hi = float(max(a.max(), b.max()))
            pad = 0.02 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=0.6, alpha=0.6)
            ax.set_title(f'r={r_val:.2f} | n={len(wide)}', fontsize=8)
            if r == 0:
                ax.set_title(f'{METRIC_LABELS.get(metric, metric)}\nr={r_val:.2f} | n={len(wide)}', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'iter {it}\nHalf B', fontsize=8)
            if r == nrows - 1:
                ax.set_xlabel('Half A', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle(f'{group_name} | {resolution} | {level_type} | first {len(iters)} of N iterations', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_scatters(scatter_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for group_name, metrics in METRIC_GROUPS.items():
        present = [m for m in metrics if m in scatter_df['metric'].unique()]
        if not present:
            continue
        for resolution in ['paragraph', 'sentence']:
            metrics_for_res = [m for m in present if not (resolution == 'sentence' and m == 'reading_speed')]
            if not metrics_for_res:
                continue
            for level_type in LEVEL_TYPES:
                out_path = out_dir / f'scatter_{group_name}_{resolution}_{level_type}.pdf'
                plot_scatter_group(
                    scatter_df, group_name, metrics_for_res, resolution, level_type, out_path,
                )


# ---------------- Distribution plots (r histograms across iterations) ----------------

def plot_r_distributions_group(
    raw_df: pd.DataFrame,
    group_name: str,
    metrics: List[str],
    corr_kind: str,
    out_path: Path,
):
    """Grid of histograms of r per (metric, resolution) for each level_type. Rows: level_type, cols: metric×resolution."""
    resolutions = ['paragraph', 'sentence']
    col_keys = [(m, r) for m in metrics for r in resolutions]

    nrows = len(LEVEL_TYPES)
    ncols = len(col_keys)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.4 * nrows), squeeze=False)

    rcol = f'{corr_kind}_r'
    for r_idx, lt in enumerate(LEVEL_TYPES):
        for c_idx, (metric, resolution) in enumerate(col_keys):
            ax = axes[r_idx, c_idx]
            sub = raw_df[
                (raw_df['metric'] == metric)
                & (raw_df['resolution'] == resolution)
                & (raw_df['level_type'] == lt)
            ]
            rs = sub[rcol].dropna().to_numpy()
            if len(rs) == 0:
                ax.set_visible(False)
                continue
            ax.hist(rs, bins=20, color='#4C78A8', alpha=0.85, edgecolor='white')
            mean_r = float(np.mean(rs))
            ax.axvline(mean_r, color='red', linewidth=1, label=f'mean={mean_r:.2f}')
            ax.axvline(float(np.percentile(rs, 2.5)), color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(float(np.percentile(rs, 97.5)), color='gray', linestyle='--', linewidth=0.8)
            ax.set_xlim(-1, 1)
            ax.tick_params(axis='both', labelsize=7)
            if r_idx == 0:
                ax.set_title(f'{METRIC_LABELS.get(metric, metric)}\n{resolution}', fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(f'{lt}\ncount', fontsize=8)
            ax.legend(fontsize=6, loc='upper left')

    n_iter = int(raw_df['iter'].nunique()) if 'iter' in raw_df.columns else 0
    fig.suptitle(f'r distribution across N={n_iter} iterations — {group_name} | {corr_kind}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_r_distributions(raw_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for group_name, metrics in METRIC_GROUPS.items():
        present = [m for m in metrics if m in raw_df['metric'].unique()]
        if not present:
            continue
        for corr_kind in ['pearson', 'spearman']:
            out_path = out_dir / f'r_dist_{group_name}_{corr_kind}.pdf'
            plot_r_distributions_group(raw_df, group_name, present, corr_kind, out_path)
