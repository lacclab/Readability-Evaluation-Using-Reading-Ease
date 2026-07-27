"""Per-subject RT data loaders for split-half analysis."""
from pathlib import Path
from typing import List, Literal

import pandas as pd

from src.Correlations.define_cols import MAIN_RT_COLS, SM_RT_COLS

# Map: RT metric name -> (eye-metric file suffix, column in that file)
RT_METRIC_INFO = {
    'mean_nonzero_TF':         ('TF',                 'mean_nonzero_TF'),
    'SkipRateTotal':           ('SR',                 'SkipRateTotal'),
    'RegRateTotal':            ('RR',                 'RegRateTotal'),
    'mean_nonzero_FF':         ('FF',                 'mean_nonzero_FF'),
    'mean_FD':                 ('FD',                 'mean_FD'),
    'mean_NF':                 ('NF',                 'mean_NF'),
    'mean_FirstPassGD':        ('FirstPassGD',        'mean_FirstPassGD'),
    'SkipRateFirstPass':       ('SR',                 'SkipRateFirstPass'),
    'RegRateFirstPass':        ('RR',                 'RegRateFirstPass'),
    'mean_GD':                 ('GD',                 'mean_GD'),
    'mean_HigherPassFixation': ('HigherPassFixation', 'mean_HigherPassFixation'),
}

READING_SPEED = 'reading_speed'
ALL_RT_METRICS = MAIN_RT_COLS + SM_RT_COLS


def text_id_cols(resolution: Literal["sentence", "paragraph"]) -> List[str]:
    return ['text_id', 'align_idx'] if resolution == 'sentence' else ['text_id']


def load_participants_metadata(src_path: Path, reader_type: str) -> pd.DataFrame:
    path = src_path / f"Participants_Metadata/data/{reader_type}/participant_metadata_processed.csv"
    md = pd.read_csv(path)
    return md[['subject_id', 'L1_or_L2']].dropna(subset=['subject_id'])


def load_rt_metric_subject_df(
    src_path: Path,
    resolution: Literal["sentence", "paragraph"],
    reader_type: str,
    reading_regime: str,
    metric: str,
) -> pd.DataFrame:
    """Per-subject per-(text, level) RT data for a single metric.

    Returns df with columns [subject_id, *text_id_cols, level, batch, <metric>].
    """
    tic = text_id_cols(resolution)

    if metric == READING_SPEED:
        if resolution != 'paragraph':
            raise ValueError("reading_speed is only available at paragraph resolution")
        path = src_path / f"Eye_metrics/data/{reader_type}/reading_speed/{reading_regime}/speed_by=subject_id_text_level.csv"
        df = pd.read_csv(path)
        df = df[df['reading_regime'] == reading_regime].copy()
        df = df.rename(columns={'words_per_sec_based_P_RT': metric})
        df['batch'] = df['text_id'].str.split('_').str[0].astype(int)
        keep = ['subject_id'] + tic + ['level', 'batch', metric]
        return df[keep]

    if metric not in RT_METRIC_INFO:
        raise KeyError(f"Unknown RT metric: {metric}")
    suffix, col = RT_METRIC_INFO[metric]
    path = src_path / f"Eye_metrics/data/{reader_type}/{reading_regime}/metric_tables/{resolution}_{suffix}_agg_by_subject_df.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={col: metric})
    keep = ['subject_id'] + tic + ['level', 'batch', metric]
    return df[keep]


def load_all_rt_metric_dfs(
    src_path: Path,
    resolution: Literal["sentence", "paragraph"],
    reader_type: str,
    reading_regime: str,
    metrics: List[str],
) -> dict:
    """Load per-subject data for each metric into a dict {metric: df}."""
    return {
        m: load_rt_metric_subject_df(src_path, resolution, reader_type, reading_regime, m)
        for m in metrics
    }
