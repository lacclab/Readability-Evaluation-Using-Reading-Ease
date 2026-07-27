"""
Find text_ids where different readability paradigms diverge.

Paradigms compared:
1. Eyetracking (ET): RT-based readability (diff of mean_nonzero_TF, SkipRateTotal, RegRateTotal)
2. Traditional formulas: annotation/formula-based readability scores (flesch, dale-chall, etc.)
3. Psycholinguistic features: idea density, integration cost, word frequency, etc.
4. Comprehension questions: comprehension_score (paragraph resolution only)

For each text, we compute the diff (Adv - Ele) for each metric, rank texts,
and find cases where paradigms disagree on which texts are "more readable".

Results saved to results/rank_based/.
"""

from pathlib import Path
import pandas as pd
from typing import Literal, List
from loguru import logger

from src.Correlations.calc_correlations import (
    _get_ele_adv_metrics_df, _add_diff_metrics, _add_reading_comprehension_metrics
)
from src.Correlations.define_cols import (
    MAIN_SURP_COLS, OPPOSITE_DIRECTION_METRICS,
)
from src.utils.data_utils import get_text_id_cols

from src.Correlations.analysis.paradigm_divergence.constants import (
    ET_COLS, FORMULA_COLS, PSYCHOLING_COLS, LLM_COLS, COMPREHENSION_COLS,
)

RESULTS_DIR = Path(__file__).parent / "results" / "rank_based"


def _flip_opposite_direction(diff_col_values: pd.Series, metric_name: str) -> pd.Series:
    """Flip sign for metrics where higher value = easier text."""
    if metric_name in OPPOSITE_DIRECTION_METRICS:
        return -diff_col_values
    return diff_col_values


def _compute_paradigm_rank(
    all_metrics_df: pd.DataFrame,
    cols: List[str],
    text_id_cols: List[str],
) -> pd.DataFrame:
    """
    For a group of columns, compute a composite percentile rank per text.
    Each diff column is flipped if opposite direction, then ranked (percentile).
    Averaged across columns in the paradigm to get a single composite rank.
    """
    diff_cols = [f"diff_{col}" for col in cols if f"diff_{col}" in all_metrics_df.columns]
    if not diff_cols:
        return pd.DataFrame()

    rank_df = all_metrics_df[text_id_cols].copy()
    used_cols = []
    for diff_col in diff_cols:
        metric_name = diff_col.replace("diff_", "")
        vals = all_metrics_df[diff_col].copy()
        vals = _flip_opposite_direction(vals, metric_name)
        rank_df[f"rank_{metric_name}"] = vals.rank(pct=True, na_option='keep') * 100
        used_cols.append(f"rank_{metric_name}")

    rank_df["composite_rank"] = rank_df[used_cols].mean(axis=1)
    return rank_df[text_id_cols + ["composite_rank"]]


def find_divergent_texts(
    src_path: Path,
    resolution: Literal["sentence", "paragraph"],
    reader_type: str = "L1_and_L2",
    reading_regime: str = "FirstReading",
):
    logger.info(f"=== Finding divergent texts: {resolution=}, {reader_type=} ===")

    level_metrics_df = _get_ele_adv_metrics_df(
        src_path, resolution, reading_regime, reader_type,
        surp_cols_to_run=MAIN_SURP_COLS, pred_type="RT"
    )

    et_cols = [c for c in ET_COLS if c in level_metrics_df.columns]
    formula_cols = [c for c in FORMULA_COLS if c in level_metrics_df.columns]
    psycholing_cols = [c for c in PSYCHOLING_COLS if c in level_metrics_df.columns]
    llm_cols = [c for c in LLM_COLS if c in level_metrics_df.columns]
    all_cols = et_cols + formula_cols + psycholing_cols + llm_cols

    include_comprehension = False
    comp_cols = []
    if resolution == "paragraph":
        try:
            merge_cols = get_text_id_cols(resolution) + ["level"]
            level_metrics_df = _add_reading_comprehension_metrics(
                src_path, level_metrics_df, resolution, reading_regime, reader_type, merge_cols
            )
            comp_cols = [c for c in COMPREHENSION_COLS if c in level_metrics_df.columns]
            if comp_cols:
                all_cols += comp_cols
                include_comprehension = True
                logger.info(f"Comprehension columns loaded: {comp_cols}")
        except Exception as e:
            logger.warning(f"Could not load comprehension metrics: {e}")

    all_metrics_df = _add_diff_metrics(resolution, all_cols, level_metrics_df)
    text_id_cols = get_text_id_cols(resolution)

    logger.info(f"Total texts: {len(all_metrics_df)}")

    # Compute composite rank per paradigm
    paradigms = {
        "ET": et_cols,
        "Formula": formula_cols,
        "Psycholinguistic": psycholing_cols,
        "LLM": llm_cols,
    }
    if include_comprehension:
        paradigms["Comprehension"] = comp_cols

    rank_dfs = {}
    for paradigm_name, cols in paradigms.items():
        rdf = _compute_paradigm_rank(all_metrics_df, cols, text_id_cols)
        if not rdf.empty:
            rdf = rdf.rename(columns={"composite_rank": f"rank_{paradigm_name}"})
            rank_dfs[paradigm_name] = rdf
            logger.info(f"  {paradigm_name}: {len(cols)} cols, "
                        f"{rdf[f'rank_{paradigm_name}'].notna().sum()} ranked texts")

    merged = all_metrics_df[text_id_cols].copy()
    for paradigm_name, rdf in rank_dfs.items():
        merged = merged.merge(rdf, on=text_id_cols, how="left")

    # Compute divergence scores
    rank_cols = [f"rank_{p}" for p in rank_dfs.keys()]
    merged["rank_std"] = merged[rank_cols].std(axis=1)

    for p1, p2 in [("ET", "Formula"), ("Psycholinguistic", "Formula"),
                    ("ET", "Psycholinguistic"), ("ET", "LLM"),
                    ("ET", "Comprehension"), ("Formula", "Comprehension")]:
        c1, c2 = f"rank_{p1}", f"rank_{p2}"
        if c1 in merged.columns and c2 in merged.columns:
            merged[f"{p1}_minus_{p2}"] = merged[c1] - merged[c2]

    # Raw diff values for interpretability
    for col in all_cols:
        diff_col = f"diff_{col}"
        if diff_col in all_metrics_df.columns:
            merged[diff_col] = all_metrics_df[diff_col].values

    merged = merged.sort_values("rank_std", ascending=False)

    # Psycholinguistic uniquely aligned with ET
    if all(c in merged.columns for c in ["ET_minus_Psycholinguistic", "ET_minus_Formula"]):
        merged["psycholing_uniquely_aligned_with_ET"] = (
            merged["ET_minus_Formula"].abs() - merged["ET_minus_Psycholinguistic"].abs()
        )

    return merged


def run_analysis(
    src_path: Path,
    resolutions: List[str] = ["paragraph", "sentence"],
    reader_type: str = "L1_and_L2",
    reading_regime: str = "FirstReading",
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for resolution in resolutions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Resolution: {resolution}")
        logger.info(f"{'='*60}")

        merged = find_divergent_texts(src_path, resolution, reader_type, reading_regime)

        output_file = RESULTS_DIR / f"paradigm_divergence_{reader_type}_{reading_regime}_{resolution}.csv"
        merged.to_csv(output_file, index=False)
        logger.info(f"Saved full results to {output_file}")

        n_top = max(1, int(len(merged) * 0.2))
        top_divergent = merged.head(n_top)
        top_file = RESULTS_DIR / f"top_divergent_{reader_type}_{reading_regime}_{resolution}.csv"
        top_divergent.to_csv(top_file, index=False)
        logger.info(f"Saved top {n_top} divergent texts to {top_file}")

        # Summary
        rank_cols = [c for c in merged.columns if c.startswith("rank_") and c != "rank_std"]
        logger.info(f"\n--- Paradigm rank correlations ({resolution}) ---")
        for i, c1 in enumerate(rank_cols):
            for c2 in rank_cols[i+1:]:
                corr = merged[c1].corr(merged[c2])
                logger.info(f"  {c1} vs {c2}: r={corr:.3f}")

        if "psycholing_uniquely_aligned_with_ET" in merged.columns:
            top_aligned = merged.nlargest(5, "psycholing_uniquely_aligned_with_ET")
            logger.info(f"\n--- Top 5 texts: Psycholinguistic uniquely aligned with ET ({resolution}) ---")
            for _, row in top_aligned.iterrows():
                id_str = "_".join([str(row[c]) for c in get_text_id_cols(resolution)])
                logger.info(
                    f"  text={id_str} | ET_rank={row.get('rank_ET', 'N/A'):.1f} | "
                    f"Formula_rank={row.get('rank_Formula', 'N/A'):.1f} | "
                    f"Psycholing_rank={row.get('rank_Psycholinguistic', 'N/A'):.1f}"
                )


if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    run_analysis(
        src_path,
        resolutions=["paragraph", "sentence"],
        reader_type="L1_and_L2",
        reading_regime="FirstReading",
    )
