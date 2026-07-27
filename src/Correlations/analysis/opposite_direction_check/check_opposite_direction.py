"""
Check which metrics behave as "opposite direction" — i.e., Adv values are
lower than Ele values (higher value = easier text).

For each text metric we compare the Ele vs Adv distributions directly
(mean, median, independent t-test) rather than using aligned diffs.
A metric is flagged as opposite-direction when mean_Adv < mean_Ele
and the difference is significant.

We then compare against the current list and report:
  - metrics that ARE in the list and the data confirms it       (OK)
  - metrics that ARE in the list but the data does NOT confirm  (UNEXPECTED)
  - metrics NOT in the list but the data suggests they should be (MISSING)

For MISSING and UNEXPECTED metrics, a raincloud plot is saved.
"""

from pathlib import Path
from typing import List, Literal
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import ttest_ind

from src.Correlations.calc_correlations import (
    _load_readability_metrics, _add_surprisal_metrics, _add_ppl_metrics,
    _add_integration_cost_metrics, _add_pll_metrics,
)
from src.Correlations.define_cols import (
    MAIN_TEXT_COLS, SM_TEXT_COLS, MAIN_SURP_COLS, SM_SURP_COLS,
    SM_PROMPT_COLS, OPPOSITE_DIRECTION_METRICS, TEXT_COLS_FULL_LABELS,
)
from src.readability_metrics.plot_funcs.plot_rain_clouds import (
    custom_raincloud_plot,
)
from src.constants import LEVEL_LABELS
from src.utils.plot_utils import LEVEL_COLORS
from src.utils.data_utils import get_text_id_cols

P_THRESHOLD = 0.05


def _load_text_metrics(src_path: Path, resolution: str):
    """Load text-level metrics (no eye-tracking needed)."""
    text_id_cols = get_text_id_cols(resolution)
    merge_cols = text_id_cols + ["level"]
    surp_cols = MAIN_SURP_COLS + SM_SURP_COLS

    metrics_df = _load_readability_metrics(src_path, resolution)
    metrics_df, existing_surp_cols = _add_surprisal_metrics(
        src_path, metrics_df, resolution, merge_cols, surp_cols
    )
    metrics_df = _add_ppl_metrics(metrics_df)
    metrics_df = _add_integration_cost_metrics(src_path, metrics_df, resolution, merge_cols)
    metrics_df = _add_pll_metrics(src_path, metrics_df, resolution, merge_cols)

    available_surp = [c for c in surp_cols if c in metrics_df.columns]
    return metrics_df, available_surp


def _save_raincloud(metrics_df, col, resolution, out_path):
    """Save a single-metric raincloud plot for the flagged metric."""
    display_name = TEXT_COLS_FULL_LABELS.get(col, col.replace('_', ' ').title())
    fig, axes = plt.subplots(1, 1, figsize=(7, 2.5))
    custom_raincloud_plot(
        merged_df=metrics_df,
        cols_to_plot=[col],
        display_names={col: display_name},
        axes=[axes],
        level_colors=LEVEL_COLORS,
        level_labels=LEVEL_LABELS,
    )
    fig.suptitle(f"{display_name} ({resolution})", fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def check_opposite_direction(
    src_path: Path,
    resolution: Literal["sentence", "paragraph"],
):
    logger.info(f"=== Checking opposite-direction metrics: {resolution} ===")

    metrics_df, available_surp = _load_text_metrics(src_path, resolution)

    # Columns to check
    all_text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + available_surp
    all_text_cols = [c for c in all_text_cols if c in metrics_df.columns]

    adv_df = metrics_df[metrics_df['level'] == 'Adv']
    ele_df = metrics_df[metrics_df['level'] == 'Ele']

    results = []
    for col in all_text_cols:
        adv_vals = adv_df[col].dropna()
        ele_vals = ele_df[col].dropna()
        if len(adv_vals) < 5 or len(ele_vals) < 5:
            continue

        mean_adv = adv_vals.mean()
        mean_ele = ele_vals.mean()
        median_adv = adv_vals.median()
        median_ele = ele_vals.median()
        std_adv = adv_vals.std()
        std_ele = ele_vals.std()

        t_stat, t_pval = ttest_ind(adv_vals, ele_vals)

        in_list = col in OPPOSITE_DIRECTION_METRICS
        # "opposite" = Adv mean is LOWER than Ele mean (significantly)
        looks_opposite = mean_adv < mean_ele and t_pval < P_THRESHOLD

        if in_list and looks_opposite:
            status = "OK"
        elif in_list and not looks_opposite:
            status = "UNEXPECTED"
        elif not in_list and looks_opposite:
            status = "MISSING"
        else:
            status = "OK"

        results.append({
            "metric": col,
            "in_OPPOSITE_DIRECTION_METRICS": in_list,
            "looks_opposite_direction": looks_opposite,
            "status": status,
            "n_adv": len(adv_vals),
            "n_ele": len(ele_vals),
            "mean_adv": round(mean_adv, 4),
            "mean_ele": round(mean_ele, 4),
            "median_adv": round(median_adv, 4),
            "median_ele": round(median_ele, 4),
            "std_adv": round(std_adv, 4),
            "std_ele": round(std_ele, 4),
            "ttest_stat": round(t_stat, 3),
            "ttest_pval": t_pval,
        })

    results_df = pd.DataFrame(results)
    return results_df, metrics_df


def run_analysis(
    src_path: Path,
    resolutions: List[str] = ["paragraph", "sentence"],
):
    results_dir = Path(__file__).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "report.txt"

    all_lines = []
    all_lines.append("=" * 80)
    all_lines.append("Opposite-Direction Metrics Check")
    all_lines.append("=" * 80)
    all_lines.append("")
    all_lines.append("A metric is 'opposite direction' when mean(Adv) < mean(Ele),")
    all_lines.append("i.e. higher values indicate easier text.")
    all_lines.append("")
    all_lines.append("Current OPPOSITE_DIRECTION_METRICS list:")
    for m in OPPOSITE_DIRECTION_METRICS:
        all_lines.append(f"  - {m}")
    all_lines.append("")

    for resolution in resolutions:
        results_df, metrics_df = check_opposite_direction(src_path, resolution)

        # Save full CSV
        csv_path = results_dir / f"opposite_direction_stats_{resolution}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved stats to {csv_path}")

        # Build report section grouped by status
        all_lines.append("-" * 80)
        all_lines.append(f"Resolution: {resolution}")
        all_lines.append("-" * 80)

        def _fmt_metric(row):
            lines = []
            lines.append(f"    mean:  Adv={row['mean_adv']}, Ele={row['mean_ele']}")
            lines.append(f"    med:   Adv={row['median_adv']}, Ele={row['median_ele']}")
            lines.append(f"    ttest: t={row['ttest_stat']}, p={row['ttest_pval']:.2e}")
            return "\n".join(lines)

        # OK — in list and confirmed
        ok_confirmed = results_df[
            (results_df["in_OPPOSITE_DIRECTION_METRICS"]) & (results_df["looks_opposite_direction"])
        ]
        if not ok_confirmed.empty:
            all_lines.append(f"\nOK — in list and confirmed ({len(ok_confirmed)}):")
            for _, row in ok_confirmed.iterrows():
                all_lines.append(f"  {row['metric']}")
                all_lines.append(_fmt_metric(row))

        # MISSING — not in list but data suggests it should be
        missing = results_df[results_df["status"] == "MISSING"]
        if not missing.empty:
            all_lines.append(f"\nMISSING — not in list but data suggests ({len(missing)}):")
            for _, row in missing.iterrows():
                logger.warning(f"[{resolution}] MISSING: {row['metric']}")
                all_lines.append(f"  {row['metric']}")
                all_lines.append(_fmt_metric(row))
                # Save raincloud
                plot_path = results_dir / f"raincloud_MISSING_{row['metric']}_{resolution}.pdf"
                _save_raincloud(metrics_df, row['metric'], resolution, plot_path)
                all_lines.append(f"    plot:  {plot_path.name}")
                logger.info(f"  Saved raincloud: {plot_path}")

        # UNEXPECTED — in list but data does not confirm
        unexpected = results_df[results_df["status"] == "UNEXPECTED"]
        if not unexpected.empty:
            all_lines.append(f"\nUNEXPECTED — in list but not confirmed ({len(unexpected)}):")
            for _, row in unexpected.iterrows():
                logger.warning(f"[{resolution}] UNEXPECTED: {row['metric']}")
                all_lines.append(f"  {row['metric']}")
                all_lines.append(_fmt_metric(row))
                # Save raincloud
                plot_path = results_dir / f"raincloud_UNEXPECTED_{row['metric']}_{resolution}.pdf"
                _save_raincloud(metrics_df, row['metric'], resolution, plot_path)
                all_lines.append(f"    plot:  {plot_path.name}")
                logger.info(f"  Saved raincloud: {plot_path}")

        all_lines.append("")

    # Write report
    report_text = "\n".join(all_lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Saved report to {report_path}")


if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    run_analysis(src_path, resolutions=["paragraph", "sentence"])
