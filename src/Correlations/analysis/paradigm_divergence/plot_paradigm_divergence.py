"""
Scatter plots of RT vs Comprehension and RT vs Formula divergence.

For each pair of (RT metric, Comprehension metric):
  Left subplot:  Raw Adv-level measures — "Hard/Easy to read" labels
  Right subplot: Diff (Adv−Ele) — "Large/Small diff" labels

Results saved to results/RT_vs_{comp_col}/ and results/RT_vs_formula/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import zscore

from src.utils.data_utils import get_text_id_cols

from src.Correlations.analysis.paradigm_divergence.constants import (
    RT_DISPLAY, COMP_DISPLAY,
    rt_difficulty_direction, comp_difficulty_direction,
)
from src.Correlations.analysis.paradigm_divergence.plot_utils import (
    scatter_with_labels, add_quadrant_labels, report_extremes,
)
from src.Correlations.analysis.paradigm_divergence.load_data import (
    load_paragraph_metrics, compute_formula_composite,
)

RESULTS_DIR = Path(__file__).parent / "results"


def plot_and_report(src_path):
    df, et_cols, formula_cols, comp_cols = load_paragraph_metrics(src_path)
    text_id_cols = get_text_id_cols("paragraph")
    df["text_id_str"] = df[text_id_cols].astype(str).agg("_".join, axis=1)
    df = compute_formula_composite(df, formula_cols)

    # Per-subfolder report lines: key = output dir path
    report_map = {}

    # ---- RT vs Comprehension ----
    for rt_col in et_cols:
        adv_rt = f"Adv_{rt_col}"
        diff_rt = f"diff_{rt_col}"
        if diff_rt not in df.columns or adv_rt not in df.columns:
            continue
        rt_label = RT_DISPLAY.get(rt_col, rt_col)
        rt_sign, rt_high_label, rt_low_label = rt_difficulty_direction(rt_col)

        for comp_col in comp_cols:
            adv_comp = f"Adv_{comp_col}"
            diff_comp = f"diff_{comp_col}"
            if diff_comp not in df.columns or adv_comp not in df.columns:
                continue
            comp_label = COMP_DISPLAY.get(comp_col, comp_col)
            comp_sign, comp_high_label, comp_low_label = comp_difficulty_direction(comp_col)

            out_dir = RESULTS_DIR / f"RT_vs_{comp_col}"
            out_dir.mkdir(parents=True, exist_ok=True)

            needed = [adv_rt, adv_comp, diff_rt, diff_comp, "text_id_str"]
            clean = df[needed].dropna()
            if len(clean) < 10:
                continue

            # Scale comprehension_score from 0-1 to 0-100%
            if comp_col == "comprehension_score":
                clean = clean.copy()
                clean[adv_comp] = clean[adv_comp] * 100
                clean[diff_comp] = clean[diff_comp] * 100
                comp_label = f"{comp_label} (%)"

            # Precompute diff variants
            plot_diff_rt = clean[diff_rt] * rt_sign
            plot_diff_comp = clean[diff_comp] * comp_sign
            rt_diff_label = f"{rt_label} diff"
            if rt_sign == -1:
                rt_diff_label += " (flipped)"
            comp_diff_label = f"{comp_label} diff"
            if comp_sign == -1:
                comp_diff_label += " (flipped)"

            abs_diff_rt = plot_diff_rt.abs()
            abs_diff_comp = plot_diff_comp.abs()

            adv_rt_z = pd.Series(zscore(clean[adv_rt], nan_policy='omit'), index=clean.index)
            adv_comp_z = pd.Series(zscore(clean[adv_comp], nan_policy='omit'), index=clean.index)
            diff_rt_z = pd.Series(zscore(plot_diff_rt, nan_policy='omit'), index=clean.index)
            diff_comp_z = pd.Series(zscore(plot_diff_comp, nan_policy='omit'), index=clean.index)

            pair_title = f"{rt_label} vs {comp_label}"

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(pair_title, fontsize=14, fontweight='bold', y=1.01)

            # --- Row 1: Raw values ---
            # (0,0) Raw Adv measures
            scatter_with_labels(
                axes[0, 0],
                clean[adv_rt], clean[adv_comp], clean["text_id_str"],
                xlabel=f"{rt_label}", ylabel=f"{comp_label}",
                title="Original level values",
                show_zero_lines=False, title_bold=True,
            )
            add_quadrant_labels(
                axes[0, 0],
                top_right=f"{rt_high_label}\n{comp_high_label}",
                bottom_left=f"{rt_low_label}\n{comp_low_label}",
            )

            # (0,1) Raw Diff
            r_diff, p_diff = scatter_with_labels(
                axes[0, 1],
                plot_diff_rt, plot_diff_comp, clean["text_id_str"],
                xlabel=f"{rt_diff_label}", ylabel=f"{comp_diff_label}",
                title="Diff (Original − Simplified)",
                title_bold=True,
            )
            add_quadrant_labels(
                axes[0, 1],
                top_left=f"Small {rt_label} diff\nLarge {comp_label} diff",
                top_right=f"Large {rt_label} diff\nLarge {comp_label} diff",
                bottom_left=f"Small {rt_label} diff\nSmall {comp_label} diff",
                bottom_right=f"Large {rt_label} diff\nSmall {comp_label} diff",
            )

            # (0,2) Absolute Diff
            r_abs, p_abs = scatter_with_labels(
                axes[0, 2],
                abs_diff_rt, abs_diff_comp, clean["text_id_str"],
                xlabel=f"|{rt_diff_label}|", ylabel=f"|{comp_diff_label}|",
                title="Absolute Diff",
                show_zero_lines=False, title_bold=True,
            )
            add_quadrant_labels(
                axes[0, 2],
                top_left=f"Small {rt_label} diff\nLarge {comp_label} diff",
                top_right=f"Large {rt_label} diff\nLarge {comp_label} diff",
                bottom_left=f"Small {rt_label} diff\nSmall {comp_label} diff",
                bottom_right=f"Large {rt_label} diff\nSmall {comp_label} diff",
            )

            # --- Row 2: Z-scored ---
            # (1,0) Z-scored Adv measures
            scatter_with_labels(
                axes[1, 0],
                adv_rt_z, adv_comp_z, clean["text_id_str"],
                xlabel=f"{rt_label} (z)", ylabel=f"{comp_label} (z)",
                title="Original level (z-scored)",
                show_zero_lines=True, title_bold=True,
            )
            add_quadrant_labels(
                axes[1, 0],
                top_right=f"{rt_high_label}\n{comp_high_label}",
                bottom_left=f"{rt_low_label}\n{comp_low_label}",
            )

            # (1,1) Z-scored Diff
            scatter_with_labels(
                axes[1, 1],
                diff_rt_z, diff_comp_z, clean["text_id_str"],
                xlabel=f"{rt_diff_label} (z)", ylabel=f"{comp_diff_label} (z)",
                title="Diff (z-scored)",
                title_bold=True,
            )
            add_quadrant_labels(
                axes[1, 1],
                top_left=f"Small {rt_label} diff\nLarge {comp_label} diff",
                top_right=f"Large {rt_label} diff\nLarge {comp_label} diff",
                bottom_left=f"Small {rt_label} diff\nSmall {comp_label} diff",
                bottom_right=f"Large {rt_label} diff\nSmall {comp_label} diff",
            )

            # (1,2) Z-scored Absolute Diff
            abs_diff_rt_z = pd.Series(zscore(abs_diff_rt, nan_policy='omit'), index=clean.index)
            abs_diff_comp_z = pd.Series(zscore(abs_diff_comp, nan_policy='omit'), index=clean.index)
            scatter_with_labels(
                axes[1, 2],
                abs_diff_rt_z, abs_diff_comp_z, clean["text_id_str"],
                xlabel=f"|{rt_diff_label}| (z)", ylabel=f"|{comp_diff_label}| (z)",
                title="Absolute Diff (z-scored)",
                show_zero_lines=True, title_bold=True,
            )
            add_quadrant_labels(
                axes[1, 2],
                top_left=f"Small {rt_label} diff\nLarge {comp_label} diff",
                top_right=f"Large {rt_label} diff\nLarge {comp_label} diff",
                bottom_left=f"Small {rt_label} diff\nSmall {comp_label} diff",
                bottom_right=f"Large {rt_label} diff\nSmall {comp_label} diff",
            )

            fig.tight_layout()
            fname = f"{rt_col}_vs_{comp_col}.pdf"
            fig.savefig(out_dir / fname, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved {out_dir / fname}")

            out_key = str(out_dir)
            if out_key not in report_map:
                report_map[out_key] = []
            report_map[out_key].append("-" * 70)

            # Diff (row 1, col 2)
            report_map[out_key] += report_extremes(
                clean, diff_rt, diff_comp, rt_label, comp_label,
                r_diff, p_diff,
                subplot_label="Diff (row 1, col 2)"
            )
            # Absolute Diff (row 1, col 3)
            clean_abs = clean[["text_id_str"]].copy()
            clean_abs["_abs_rt"] = abs_diff_rt.values
            clean_abs["_abs_comp"] = abs_diff_comp.values
            report_map[out_key] += report_extremes(
                clean_abs, "_abs_rt", "_abs_comp",
                f"|{rt_label}|", f"|{comp_label}|",
                r_abs, p_abs,
                subplot_label="Absolute Diff (row 1, col 3)"
            )
            # Z-scored Diff (row 2, col 2)
            clean_z = clean[["text_id_str"]].copy()
            clean_z["_z_rt"] = diff_rt_z.values
            clean_z["_z_comp"] = diff_comp_z.values
            report_map[out_key] += report_extremes(
                clean_z, "_z_rt", "_z_comp",
                f"{rt_label} (z)", f"{comp_label} (z)",
                r_diff, p_diff,
                subplot_label="Diff z-scored (row 2, col 2)"
            )

    # ---- RT vs Formula composite ----
    rt_vs_formula_dir = RESULTS_DIR / "RT_vs_formula"
    rt_vs_formula_dir.mkdir(parents=True, exist_ok=True)

    for rt_col in et_cols:
        diff_rt = f"diff_{rt_col}"
        if diff_rt not in df.columns:
            continue
        rt_label = RT_DISPLAY.get(rt_col, rt_col)
        rt_sign, _, _ = rt_difficulty_direction(rt_col)

        needed = [diff_rt, "formula_composite_z", "text_id_str"]
        clean = df[needed].dropna()
        if len(clean) < 10:
            continue

        plot_diff_rt = clean[diff_rt] * rt_sign
        rt_diff_label = f"{rt_label} diff"
        if rt_sign == -1:
            rt_diff_label += " (flipped)"
        formula_label = "Formula Composite (z)"

        abs_diff_rt = plot_diff_rt.abs()
        abs_formula = clean["formula_composite_z"].abs()
        diff_rt_z = pd.Series(zscore(plot_diff_rt, nan_policy='omit'), index=clean.index)
        formula_z = pd.Series(zscore(clean["formula_composite_z"], nan_policy='omit'), index=clean.index)

        pair_title = f"{rt_label} vs Formula Composite"
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(pair_title, fontsize=14, fontweight='bold', y=1.01)

        # (0) Diff
        r_diff, p_diff = scatter_with_labels(
            axes[0], plot_diff_rt, clean["formula_composite_z"],
            clean["text_id_str"],
            xlabel=f"{rt_diff_label}", ylabel=formula_label,
            title="Diff (Original − Simplified)",
            title_bold=True,
        )
        add_quadrant_labels(
            axes[0],
            top_left=f"Small {rt_label} diff\nLarge Formula diff",
            top_right=f"Large {rt_label} diff\nLarge Formula diff",
            bottom_left=f"Small {rt_label} diff\nSmall Formula diff",
            bottom_right=f"Large {rt_label} diff\nSmall Formula diff",
        )

        # (1) Absolute Diff
        r_abs, p_abs = scatter_with_labels(
            axes[1], abs_diff_rt, abs_formula,
            clean["text_id_str"],
            xlabel=f"|{rt_diff_label}|", ylabel=f"|{formula_label}|",
            title="Absolute Diff",
            show_zero_lines=False, title_bold=True,
        )
        add_quadrant_labels(
            axes[1],
            top_left=f"Small {rt_label} diff\nLarge Formula diff",
            top_right=f"Large {rt_label} diff\nLarge Formula diff",
            bottom_left=f"Small {rt_label} diff\nSmall Formula diff",
            bottom_right=f"Large {rt_label} diff\nSmall Formula diff",
        )

        # (2) Z-scored Diff
        scatter_with_labels(
            axes[2], diff_rt_z, formula_z,
            clean["text_id_str"],
            xlabel=f"{rt_diff_label} (z)", ylabel=f"{formula_label} (z)",
            title="Diff (z-scored)",
            title_bold=True,
        )
        add_quadrant_labels(
            axes[2],
            top_left=f"Small {rt_label} diff\nLarge Formula diff",
            top_right=f"Large {rt_label} diff\nLarge Formula diff",
            bottom_left=f"Small {rt_label} diff\nSmall Formula diff",
            bottom_right=f"Large {rt_label} diff\nSmall Formula diff",
        )

        fig.tight_layout()
        fname = f"{rt_col}_vs_formula_composite.pdf"
        fig.savefig(rt_vs_formula_dir / fname, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {rt_vs_formula_dir / fname}")

        formula_key = str(rt_vs_formula_dir)
        if formula_key not in report_map:
            report_map[formula_key] = []
        report_map[formula_key].append("-" * 70)
        report_map[formula_key] += report_extremes(
            clean, diff_rt, "formula_composite_z",
            rt_label, formula_label,
            r_diff, p_diff,
            subplot_label="Diff (col 1)"
        )
        clean_abs = clean[["text_id_str"]].copy()
        clean_abs["_abs_rt"] = abs_diff_rt.values
        clean_abs["_abs_f"] = abs_formula.values
        report_map[formula_key] += report_extremes(
            clean_abs, "_abs_rt", "_abs_f",
            f"|{rt_label}|", f"|{formula_label}|",
            r_abs, p_abs,
            subplot_label="Absolute Diff (col 2)"
        )

    # ---- Save reports per subfolder ----
    for dir_path_str, lines in report_map.items():
        dir_path = Path(dir_path_str)
        header = [
            "=" * 70,
            f"Paradigm Divergence: {dir_path.name}",
            "paragraph resolution, L1_and_L2, FirstReading",
            "=" * 70,
            "",
        ]
        report_path = dir_path / "report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(header + lines))
        logger.info(f"Saved report to {report_path}")


if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    plot_and_report(src_path)
