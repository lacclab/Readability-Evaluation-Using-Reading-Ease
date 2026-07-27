"""Entry point: run split-half analysis for L1_and_L2 / FirstReading.

For each iteration (1..N):
    - Randomly split subjects into halves A/B, stratified by (batch, L1_or_L2).
    - For each resolution (paragraph, sentence), for each RT metric:
        - Aggregate metric per (text, level) by averaging subjects within each half.
        - Per level_type in {Adv+Ele, diff}, correlate half A vs half B across texts.

Outputs (under src/Correlations/analysis/split_half/results/):
    - split_half_raw_iterations.csv  : one row per (iter, metric, resolution, level_type)
    - split_half_summary.csv         : aggregated mean, 95% CI (percentile + t-based), SB-corrected
    - split_half_scatter_data.csv    : per-text per-half values, first 10 iters (for scatters)
    - plots/no_ci/, plots/percentile_ci/, plots/tstd_ci/ : bar plots per CI method
    - plots/scatters/                : scatter plots of half A vs half B (10 iters each)
    - plots/r_distributions/         : histograms of r across iterations

Usage:
    python -m src.Correlations.analysis.split_half.run
        Run everything: iterations + summary + plots.
    python -m src.Correlations.analysis.split_half.run --skip-iterations
        Skip the (expensive) iterations step; reload the existing raw CSV and
        only re-build the summary and plots.
"""
import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.Correlations.analysis.split_half.analysis import (
    run_split_half_iterations, summarize,
)
from src.Correlations.analysis.split_half.data import (
    ALL_RT_METRICS, READING_SPEED, load_all_rt_metric_dfs, load_participants_metadata,
)
from src.Correlations.analysis.split_half.plots import (
    plot_all_bars, plot_all_r_distributions, plot_all_scatters,
)

N_ITER = 1000
SCATTER_ITERS = 10
READER_TYPE = 'L1_and_L2'
READING_REGIME = 'FirstReading'


def compute_raw(
    src_path: Path, out_dir: Path, n_iter: int, scatter_iters: int,
) -> tuple:
    """Run the split-half iterations and save raw + scatter CSVs. Returns (raw_df, scatter_df)."""
    metadata = load_participants_metadata(src_path, READER_TYPE)
    logger.info(f"Loaded metadata for {len(metadata)} participants")

    all_raw, all_scatter = [], []
    for resolution in ['paragraph', 'sentence']:
        metrics = [m for m in ALL_RT_METRICS
                   if not (resolution == 'sentence' and m == READING_SPEED)]
        logger.info(f"Loading per-subject data for {resolution} ({len(metrics)} metrics)")
        subject_dfs = load_all_rt_metric_dfs(
            src_path, resolution, READER_TYPE, READING_REGIME, metrics
        )
        raw, scatter = run_split_half_iterations(
            subject_dfs, metadata, resolution,
            n_iter=n_iter, scatter_iters=scatter_iters,
        )
        all_raw.append(raw)
        all_scatter.append(scatter)

    raw_df = pd.concat(all_raw, ignore_index=True)
    raw_df.to_csv(out_dir / 'split_half_raw_iterations.csv', index=False)
    logger.info(f"Saved raw iterations to {out_dir / 'split_half_raw_iterations.csv'}")

    scatter_df = pd.concat(all_scatter, ignore_index=True)
    scatter_df.to_csv(out_dir / 'split_half_scatter_data.csv', index=False)
    logger.info(f"Saved scatter data to {out_dir / 'split_half_scatter_data.csv'}")

    return raw_df, scatter_df


def build_summary_and_plots(out_dir: Path, plots_dir: Path, raw_df: pd.DataFrame, scatter_df: pd.DataFrame) -> None:
    """Compute summary CSV and generate all plots from already-computed raw/scatter dfs."""
    summary_df = summarize(raw_df)
    summary_df.to_csv(out_dir / 'split_half_summary.csv', index=False)
    logger.info(f"Saved summary to {out_dir / 'split_half_summary.csv'}")

    plot_all_bars(summary_df, plots_dir)
    plot_all_scatters(scatter_df, plots_dir / 'scatters')
    plot_all_r_distributions(raw_df, plots_dir / 'r_distributions')
    logger.info(f"Saved plots under {plots_dir}")


def run(
    src_path: Path,
    n_iter: int = N_ITER,
    scatter_iters: int = SCATTER_ITERS,
    skip_iterations: bool = False,
) -> None:
    out_dir = src_path / 'Correlations' / 'analysis' / 'split_half' / 'results'
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if skip_iterations:
        raw_csv = out_dir / 'split_half_raw_iterations.csv'
        scatter_csv = out_dir / 'split_half_scatter_data.csv'
        if not raw_csv.exists() or not scatter_csv.exists():
            raise FileNotFoundError(
                f"--skip-iterations requires existing {raw_csv.name} and {scatter_csv.name}"
            )
        logger.info(f"Loading existing raw iterations from {raw_csv}")
        raw_df = pd.read_csv(raw_csv)
        scatter_df = pd.read_csv(scatter_csv)
    else:
        raw_df, scatter_df = compute_raw(src_path, out_dir, n_iter, scatter_iters)

    build_summary_and_plots(out_dir, plots_dir, raw_df, scatter_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skip-iterations', action='store_true',
        help='Skip the iteration step and reload the existing raw CSVs; only rebuild summary + plots.',
    )
    parser.add_argument('--n-iter', type=int, default=N_ITER)
    args = parser.parse_args()

    src_path = Path.cwd() / 'src'
    run(src_path, n_iter=args.n_iter, skip_iterations=args.skip_iterations)
