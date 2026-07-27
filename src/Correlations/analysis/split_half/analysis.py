"""Split-half analysis core: split subjects, aggregate, correlate halves."""
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.Correlations.analysis.split_half.data import text_id_cols

LEVEL_TYPES = ['Adv+Ele', 'diff']


def spearman_brown(r: float) -> float:
    """Spearman-Brown prophecy to estimate full-sample reliability from a half-half correlation."""
    if pd.isna(r) or r == -1:
        return np.nan
    return 2 * r / (1 + r)


def stratified_split_per_batch(
    subject_batch_df: pd.DataFrame,
    metadata: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign each subject to half 'A' or 'B', stratified by (batch, L1_or_L2)."""
    df = subject_batch_df.drop_duplicates().merge(metadata, on='subject_id', how='left')
    df['L1_or_L2'] = df['L1_or_L2'].fillna('unknown')

    assignments = []
    for (batch, lang), g in df.groupby(['batch', 'L1_or_L2']):
        subs = g['subject_id'].to_numpy()
        perm = rng.permutation(len(subs))
        half = len(subs) // 2
        half_a = set(subs[perm[:half]])
        g2 = g.copy()
        g2['half'] = g2['subject_id'].map(lambda s: 'A' if s in half_a else 'B')
        assignments.append(g2)
    return pd.concat(assignments, ignore_index=True)[['subject_id', 'batch', 'half']]


def aggregate_per_text_level(
    subj_df: pd.DataFrame,
    metric: str,
    resolution: Literal["sentence", "paragraph"],
) -> pd.DataFrame:
    """Mean metric across subjects per (text..., level, half)."""
    tic = text_id_cols(resolution)
    return (
        subj_df.groupby(tic + ['level', 'half'], dropna=False)[metric]
        .mean()
        .reset_index()
    )


def pivot_levels_per_half(
    agg_df: pd.DataFrame,
    metric: str,
    resolution: Literal["sentence", "paragraph"],
) -> pd.DataFrame:
    """Returns a long-ish df with columns [text_id..., half, Adv, Ele, diff]."""
    tic = text_id_cols(resolution)
    pivoted = agg_df.pivot_table(
        index=tic + ['half'], columns='level', values=metric
    ).reset_index()
    pivoted = pivoted.dropna(subset=['Adv', 'Ele'])
    pivoted['diff'] = pivoted['Adv'] - pivoted['Ele']
    return pivoted[tic + ['half', 'Adv', 'Ele', 'diff']]


def _halves_ab_for_level_type(
    level_df: pd.DataFrame,
    level_type: str,
    resolution: Literal["sentence", "paragraph"],
) -> pd.DataFrame:
    """Return a df with columns [A, B] of per-text(-and-level) values per half.

    - level_type == 'diff': one row per text, value = Adv - Ele.
    - level_type == 'Adv+Ele': two rows per text (one per level), pooled.
    """
    tic = text_id_cols(resolution)
    if level_type == 'Adv+Ele':
        long = pd.concat([
            level_df[tic + ['half', 'Adv']].rename(columns={'Adv': 'val'}).assign(level='Adv'),
            level_df[tic + ['half', 'Ele']].rename(columns={'Ele': 'val'}).assign(level='Ele'),
        ], ignore_index=True)
        wide = long.pivot_table(index=tic + ['level'], columns='half', values='val').reset_index()
    else:
        wide = level_df.pivot_table(index=tic, columns='half', values=level_type).reset_index()
    return wide.dropna(subset=['A', 'B'])


def correlate_halves_all_levels(
    level_df: pd.DataFrame,
    resolution: Literal["sentence", "paragraph"],
) -> List[dict]:
    """For each level_type in {Adv+Ele, diff}, correlate half A vs half B."""
    out = []
    for lt in LEVEL_TYPES:
        wide = _halves_ab_for_level_type(level_df, lt, resolution)
        n = len(wide)
        if n < 3 or wide['A'].nunique() < 2 or wide['B'].nunique() < 2:
            out.append({
                'level_type': lt,
                'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan, 'n_texts': n,
            })
            continue
        pr, pp = pearsonr(wide['A'], wide['B'])
        sr, sp = spearmanr(wide['A'], wide['B'])
        out.append({
            'level_type': lt,
            'pearson_r': pr, 'pearson_p': pp,
            'spearman_r': sr, 'spearman_p': sp, 'n_texts': n,
        })
    return out


def run_split_half_iterations(
    subject_dfs: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    resolution: Literal["sentence", "paragraph"],
    n_iter: int,
    base_seed: int = 42,
    scatter_iters: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run N split-half iterations.

    Returns
    -------
    raw_df : one row per (iter, metric, resolution, level_type).
    scatter_df : per-text per-half values for the first `scatter_iters` iterations,
        with columns [iter, metric, resolution, *text_id_cols, half, Adv, Ele, diff].
    """
    pairs = pd.concat(
        [df[['subject_id', 'batch']] for df in subject_dfs.values()], ignore_index=True
    ).drop_duplicates()

    results = []
    scatter_chunks = []
    for i in tqdm(range(n_iter), desc=f"Split-half {resolution}"):
        rng = np.random.default_rng(base_seed + i)
        split_df = stratified_split_per_batch(pairs, metadata, rng)

        for metric, subj_df in subject_dfs.items():
            merged = subj_df.merge(split_df, on=['subject_id', 'batch'], how='left')
            if merged['half'].isna().any():
                merged = merged.dropna(subset=['half'])
            agg = aggregate_per_text_level(merged, metric, resolution)
            level_df = pivot_levels_per_half(agg, metric, resolution)

            for row in correlate_halves_all_levels(level_df, resolution):
                results.append({
                    'iter': i,
                    'metric': metric,
                    'resolution': resolution,
                    **row,
                })

            if i < scatter_iters:
                chunk = level_df.copy()
                chunk['iter'] = i
                chunk['metric'] = metric
                chunk['resolution'] = resolution
                scatter_chunks.append(chunk)

    raw_df = pd.DataFrame(results)
    scatter_df = pd.concat(scatter_chunks, ignore_index=True) if scatter_chunks else pd.DataFrame()
    return raw_df, scatter_df


def _t_based_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Parametric t-based CI for the mean (matches utils.stat_analysis.get_mean_ci with sem_or_std='std')."""
    data = values[~np.isnan(values)]
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    mean_val = float(np.mean(data))
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    margin = t_crit * float(np.std(data, ddof=1))
    return (mean_val - margin, mean_val + margin)


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across iterations.

    Columns:
      mean_r, ci_low, ci_high                 — percentile CI over iterations
      ci_low_tstd, ci_high_tstd               — parametric t-based CI (mean ± t*std)
      sb_mean_r, sb_ci_low, sb_ci_high        — Spearman-Brown corrected, percentile CI
      sb_ci_low_tstd, sb_ci_high_tstd         — Spearman-Brown corrected, t-based CI
    """
    rows = []
    for (metric, resolution, level_type), g in raw_df.groupby(['metric', 'resolution', 'level_type']):
        for corr_kind in ['pearson', 'spearman']:
            rs = g[f'{corr_kind}_r'].to_numpy(dtype=float)
            rs_clean = rs[~np.isnan(rs)]
            if len(rs_clean) == 0:
                continue
            mean_r = float(np.mean(rs_clean))
            ci_low = float(np.percentile(rs_clean, 2.5))
            ci_high = float(np.percentile(rs_clean, 97.5))
            t_low, t_high = _t_based_ci(rs_clean)

            sb_rs = np.array([spearman_brown(r) for r in rs_clean], dtype=float)
            sb_rs = sb_rs[~np.isnan(sb_rs)]
            if len(sb_rs):
                sb_mean = float(np.mean(sb_rs))
                sb_ci_low = float(np.percentile(sb_rs, 2.5))
                sb_ci_high = float(np.percentile(sb_rs, 97.5))
                sb_t_low, sb_t_high = _t_based_ci(sb_rs)
            else:
                sb_mean = sb_ci_low = sb_ci_high = sb_t_low = sb_t_high = np.nan

            rows.append({
                'metric': metric,
                'resolution': resolution,
                'level_type': level_type,
                'corr_kind': corr_kind,
                'n_iter': len(rs_clean),
                'mean_r': mean_r,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'ci_low_tstd': t_low,
                'ci_high_tstd': t_high,
                'sb_mean_r': sb_mean,
                'sb_ci_low': sb_ci_low,
                'sb_ci_high': sb_ci_high,
                'sb_ci_low_tstd': sb_t_low,
                'sb_ci_high_tstd': sb_t_high,
                'mean_n_texts': float(g['n_texts'].mean()),
            })
    return pd.DataFrame(rows)
