"""
Generate PNG diff visualizations for texts where RT and Comprehension diverge.

Picks top-5 texts in each divergence direction (small RT + large Comp, and vice versa)
and renders a colored word-diff PNG with similarity/word-diff metrics on the right panel.

Results saved under results/RT_vs_{comp_col}/text_examples/{rt_col}/.
"""

from pathlib import Path
from loguru import logger

from src.Correlations.analysis.paradigm_divergence.constants import RT_DISPLAY, COMP_DISPLAY
from src.Correlations.analysis.paradigm_divergence.load_data import (
    load_paragraph_metrics, build_example_metrics,
)
from src.Alignment_Sentences.text_comparison.examples.generate_examples import (
    _render_example_png, _load_merged_data,
)
from src.utils.data_utils import get_text_id_cols

RESULTS_DIR = Path(__file__).parent / "results"
N_EXAMPLES = 5

def _find_divergent_text_ids(df, col_x, col_y, id_col="text_id_str", n=N_EXAMPLES):
    """Find top-n texts where two metrics diverge in each direction.

    Uses absolute values normalized to [0,1] so that the two metrics
    are on a comparable scale regardless of units.

    Returns (high_y_low_x_ids, high_x_low_y_ids).
    """
    x_abs = df[col_x].abs()
    y_abs = df[col_y].abs()
    x_range = x_abs.max() - x_abs.min()
    y_range = y_abs.max() - y_abs.min()

    if x_range == 0 or y_range == 0:
        return [], []

    x_norm = (x_abs - x_abs.min()) / x_range
    y_norm = (y_abs - y_abs.min()) / y_range
    residual = y_norm - x_norm  # positive = y relatively larger

    high_y_ids = df.loc[residual.nlargest(n).index, id_col].tolist()
    high_x_ids = df.loc[residual.nsmallest(n).index, id_col].tolist()
    return high_y_ids, high_x_ids


CATEGORY_LABELS = {
    "small_RT_large_Comp": "Small Reading Time diff, Large Comprehension diff",
    "large_RT_small_Comp": "Large Reading Time diff, Small Comprehension diff",
}


def _format_comp_diff(comp_col, comp_diff_val):
    """Format comprehension diff value (scale to % for accuracy)."""
    if comp_col == "comprehension_score":
        return comp_diff_val * 100, "%"
    return comp_diff_val, ""


def generate_divergent_examples(
    src_path: Path,
    rt_cols_to_run: list = None,
    comp_cols_to_run: list = None,
    n_examples: int = N_EXAMPLES,
):
    # Load readability metrics (ET + comprehension diffs)
    df, et_cols, _, comp_cols = load_paragraph_metrics(src_path)
    text_id_cols = get_text_id_cols("paragraph")
    df["text_id_str"] = df[text_id_cols].astype(str).agg("_".join, axis=1)

    # Load word-diff & similarity metrics for all paragraphs (with text content).
    # The full corpus df is passed to build_example_metrics so that
    # range bars show where each example falls relative to the corpus.
    word_diff_and_similarity_df = _load_merged_data(src_path)

    rt_cols_to_run = rt_cols_to_run or et_cols
    comp_cols_to_run = comp_cols_to_run or comp_cols

    for rt_col in rt_cols_to_run:
        diff_rt = f"diff_{rt_col}"
        if diff_rt not in df.columns:
            continue
        rt_label = RT_DISPLAY.get(rt_col, rt_col)

        for comp_col in comp_cols_to_run:
            diff_comp = f"diff_{comp_col}"
            if diff_comp not in df.columns:
                continue
            comp_label = COMP_DISPLAY.get(comp_col, comp_col)

            clean = df[[diff_rt, diff_comp, "text_id_str"]].dropna()
            if len(clean) < 10:
                continue

            # Find divergent texts
            small_rt_ids, large_rt_ids = _find_divergent_text_ids(
                clean, diff_rt, diff_comp, n=n_examples
            )

            out_dir = RESULTS_DIR / f"RT_vs_{comp_col}" / "text_examples" / rt_col
            out_dir.mkdir(parents=True, exist_ok=True)

            examples = (
                [(tid, "small_RT_large_Comp") for tid in small_rt_ids] +
                [(tid, "large_RT_small_Comp") for tid in large_rt_ids]
            )

            for text_id, category in examples:
                row = clean[clean["text_id_str"] == text_id]
                if row.empty:
                    continue
                row = row.iloc[0]

                rt_diff_val = row[diff_rt]
                comp_display, comp_unit = _format_comp_diff(comp_col, row[diff_comp])

                text_ele, text_adv, left_metrics, right_metrics = build_example_metrics(
                    text_id, word_diff_and_similarity_df,
                    extra_text_lines=[
                        f"{rt_label} diff: {rt_diff_val:+.2f}",
                        f"{comp_label} diff: {comp_display:+.2f}{comp_unit}",
                    ],
                )
                if text_ele is None:
                    logger.warning(f"Text {text_id} not found in text data")
                    continue

                _render_example_png(
                    out_path=out_dir / f"{category}_{text_id}.png",
                    text_ele=text_ele,
                    text_adv=text_adv,
                    title=f"{text_id}  —  {CATEGORY_LABELS[category]}",
                    left_metrics=left_metrics,
                    right_metrics=right_metrics,
                    fontsize=10,
                    max_width=85,
                )

            logger.info(f"Saved {len(examples)} examples to {out_dir}")


if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    generate_divergent_examples(
        src_path,
        rt_cols_to_run=["mean_nonzero_TF"],
        comp_cols_to_run=["comprehension_score"],
    )
