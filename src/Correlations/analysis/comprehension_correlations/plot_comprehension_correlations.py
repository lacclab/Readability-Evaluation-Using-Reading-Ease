"""
Comprehension Correlations plot — for CL rebuttal only.

Generates the horizontal bar chart of Comprehension Accuracy / QA RT
correlations with readability metrics, showing both "Not Controlled"
and "Controlled (Δ: Original − Simplified)" sections.

Also generates the reviewer-requested version: Controlled only + legend.
"""

from pathlib import Path
import shutil
from loguru import logger

from src.Correlations.calc_correlations import agg_folds_correlations, calc_correlations
from src.Correlations.plots_code.grid_corr_bar_RTx2_RTxSenPar_diff_only import plot_corr_grid_RTx2_RTxSenPar
from src.Correlations.define_cols import (
    READING_COMPREHENSION_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS, SLOR_COLS, UID_COLS,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Cols that are newly added (SLOR, UID, surp) and need recomputation in calc step.
# Matches the set used in src/Correlations/run.py.
CALC_FOR_SPECIFIC_TEXT_COLS = ["PPL Pythia 70M", "mean_entropy_pythia"] + SLOR_COLS + UID_COLS


def run_calc_comprehension(src_path: Path):
    """Recompute per-fold correlations for comprehension on the specific newly-added
    text cols (SLOR, UID, surp). Slow — only run when those cols change."""
    calc_correlations(
        src_path,
        resolution="paragraph",
        reader_type="L1_and_L2",
        reading_regime="FirstReading",
        pred_type="comprehension",
        include_bootstrap=True,
        run_for_specific_text_cols=CALC_FOR_SPECIFIC_TEXT_COLS,
    )


def ensure_comprehension_in_agg(src_path: Path):
    """Re-aggregate comprehension per-fold data into the agg file (fast, no recomputation)."""
    agg_folds_correlations(
        src_path,
        resolution="paragraph",
        L1_or_L2="L1_and_L2",
        reading_regime="FirstReading",
        include_bootstrap=True,
        pred_cols=READING_COMPREHENSION_COLS,
    )


def _copy_to_results(src_path: Path, output_file: str):
    """Copy the plot from the standard output path to the local results dir."""
    saved_name = output_file.replace(".pdf", "_boot.pdf")
    standard_path = src_path / f"Correlations/L1_and_L2/FirstReading/{saved_name}"
    if standard_path.exists():
        dest = RESULTS_DIR / saved_name
        shutil.copy2(standard_path, dest)
        logger.info(f"Copied plot to {dest}")
    else:
        logger.warning(f"Expected plot not found at {standard_path}")


def plot_comprehension_correlations(src_path: Path):
    """Full 2×1 grid: Not Controlled + Controlled, both pred cols."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = "comprehension_all_levels_pearson_corr_RE_next_to_delta_RE.pdf"

    plot_corr_grid_RTx2_RTxSenPar(
        src_path=src_path,
        reader_type="L1_and_L2",
        reading_regime="FirstReading",
        pred_cols=READING_COMPREHENSION_COLS,
        text_cols=(MAIN_TEXT_COLS + MAIN_SURP_COLS),
        corr_to_plot=["pearson_corr"],
        output_file=output_file,
        est_strategy="Bootstrap",
        orientation="horizontal",
        diff_only=False,
        pred_type="comprehension",
    )
    _copy_to_results(src_path, output_file)


def plot_comprehension_controlled_only(src_path: Path):
    """Reviewer request: Controlled (main analysis) only + legend."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = "comprehension_controlled_only_corr.pdf"

    plot_corr_grid_RTx2_RTxSenPar(
        src_path=src_path,
        reader_type="L1_and_L2",
        reading_regime="FirstReading",
        pred_cols=READING_COMPREHENSION_COLS,
        text_cols=(MAIN_TEXT_COLS + MAIN_SURP_COLS),
        corr_to_plot=["pearson_corr"],
        output_file=output_file,
        est_strategy="Bootstrap",
        orientation="horizontal",
        diff_only=True,
        pred_type="comprehension",
    )
    _copy_to_results(src_path, output_file)


def plot_comprehension_accuracy_controlled_only(src_path: Path):
    """Reviewer request: Controlled + Comprehension Accuracy only (single panel) + legend."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = "comprehension_accuracy_controlled_only_corr.pdf"

    plot_corr_grid_RTx2_RTxSenPar(
        src_path=src_path,
        reader_type="L1_and_L2",
        reading_regime="FirstReading",
        pred_cols=["comprehension_score"],
        text_cols=(MAIN_TEXT_COLS + MAIN_SURP_COLS),
        corr_to_plot=["pearson_corr"],
        output_file=output_file,
        est_strategy="Bootstrap",
        orientation="vertical",
        diff_only=True,
        pred_type="comprehension",
        title_override="Correlations with Reading Comprehension (Passages)",
    )
    _copy_to_results(src_path, output_file)


if __name__ == "__main__":
    src_path = Path.cwd() / "src"

    run_calc = False  # set True after adding new text cols (SLOR / UID / surp)
    if run_calc:
        run_calc_comprehension(src_path)

    ensure_comprehension_in_agg(src_path)
    plot_comprehension_correlations(src_path)
    plot_comprehension_controlled_only(src_path)
    plot_comprehension_accuracy_controlled_only(src_path)
