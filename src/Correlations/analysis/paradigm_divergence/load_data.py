"""
Data loading and preparation functions for paradigm divergence analysis.
"""

from pathlib import Path
import numpy as np
from scipy.stats import zscore

from src.Correlations.calc_correlations import (
    _get_ele_adv_metrics_df, _add_diff_metrics, _add_reading_comprehension_metrics,
)
from src.Correlations.define_cols import (
    MAIN_SURP_COLS, OPPOSITE_DIRECTION_METRICS,
)
from src.utils.data_utils import get_text_id_cols
from src.Alignment_Sentences.text_comparison.examples.generate_examples import (
    WORD_DIFF_COLS, WORD_DIFF_BAR_RANGES,
    SIMILARITY_SHORT_NAMES, SIMILARITY_BAR_RANGES,
)

from src.Correlations.analysis.paradigm_divergence.constants import (
    ALL_RT_COLS, FORMULA_COLS, COMPREHENSION_COLS,
)


def load_paragraph_metrics(src_path: Path):
    """Load paragraph metrics with ET, formulas, and comprehension.

    Returns (all_metrics_df, et_cols, formula_cols, comp_cols).
    """
    resolution = "paragraph"
    reader_type = "L1_and_L2"
    reading_regime = "FirstReading"

    level_metrics_df = _get_ele_adv_metrics_df(
        src_path, resolution, reading_regime, reader_type,
        surp_cols_to_run=MAIN_SURP_COLS, pred_type="RT"
    )

    merge_cols = get_text_id_cols(resolution) + ["level"]
    level_metrics_df = _add_reading_comprehension_metrics(
        src_path, level_metrics_df, resolution,
        reading_regime, reader_type, merge_cols
    )

    et_cols = [c for c in ALL_RT_COLS if c in level_metrics_df.columns]
    formula_cols = [c for c in FORMULA_COLS if c in level_metrics_df.columns]
    comp_cols = [c for c in COMPREHENSION_COLS if c in level_metrics_df.columns]
    all_cols = et_cols + formula_cols + comp_cols

    all_metrics_df = _add_diff_metrics(resolution, all_cols, level_metrics_df)
    return all_metrics_df, et_cols, formula_cols, comp_cols


def build_example_metrics(text_id, word_diff_and_similarity_df, extra_text_lines=None):
    """Build left_metrics (word-diff) and right_metrics (similarity) for a text.

    The full corpus DataFrame is used to compute range bars (corpus min/max)
    so each example shows where it falls relative to all paragraphs.

    Args:
        text_id: paragraph text_id to look up.
        word_diff_and_similarity_df: Full corpus DataFrame from _load_merged_data
            containing word-diff and similarity metrics for all paragraphs.
            Used both to look up this text's values and to compute corpus-level
            range bars (min/max markers).
        extra_text_lines: list of plain-text strings to prepend to left_metrics
                          (e.g., "Total Fixation Time diff: +32.6").

    Returns (text_ele, text_adv, left_metrics, right_metrics) or Nones if not found.
    """
    text_row = word_diff_and_similarity_df[word_diff_and_similarity_df["text_id"] == text_id]
    if text_row.empty:
        return None, None, None, None
    text_row = text_row.iloc[0]

    # Left panel: optional extra lines + word-diff metrics with range bars
    left_metrics = list(extra_text_lines or [])
    for col in WORD_DIFF_COLS:
        if col in word_diff_and_similarity_df.columns:
            val = float(text_row[col])
            corpus_vals = word_diff_and_similarity_df[col].values
            bar_range = WORD_DIFF_BAR_RANGES.get(col)
            bmin = bar_range[0] if bar_range else float(np.nanmin(corpus_vals))
            bmax = bar_range[1] if bar_range else float(np.nanmax(corpus_vals))
            left_metrics.append({
                'label': col, 'value': val, 'all_values': corpus_vals,
                'bar_min': bmin, 'bar_max': bmax,
            })

    # Right panel: similarity metrics with range bars
    right_metrics = []
    for full_col, short_name in SIMILARITY_SHORT_NAMES.items():
        if full_col in word_diff_and_similarity_df.columns:
            val = float(text_row[full_col])
            corpus_vals = word_diff_and_similarity_df[full_col].values
            bar_range = SIMILARITY_BAR_RANGES.get(short_name, (0, 1))
            right_metrics.append({
                'label': short_name, 'value': val, 'all_values': corpus_vals,
                'bar_min': bar_range[0], 'bar_max': bar_range[1],
            })

    return text_row["text_ele"], text_row["text_adv"], left_metrics, right_metrics


def compute_formula_composite(df, formula_cols):
    """Average z-scored formula diffs into a single composite.

    Returns df with 'formula_composite_z' column added.
    """
    z_cols = []
    for col in formula_cols:
        diff_col = f"diff_{col}"
        if diff_col in df.columns:
            vals = df[diff_col].copy()
            if col in OPPOSITE_DIRECTION_METRICS:
                vals = -vals
            df[f"_z_{diff_col}"] = zscore(vals, nan_policy='omit')
            z_cols.append(f"_z_{diff_col}")
    df["formula_composite_z"] = df[z_cols].mean(axis=1)
    return df
