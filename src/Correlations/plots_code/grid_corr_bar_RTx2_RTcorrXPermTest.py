import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from typing import List, Literal
from loguru import logger

from src.constants import PRED_COLS_FULL_LABELS
from src.utils.plot_utils import (
    add_significance_legend,
    SIGNIFICANCE_COLORS, SIGNIFICANCE_LABELS, SIGNIFICANCE_SIGN_DIFF_COLORS, SIGNIFICANCE_SIGN_DIFF_LABELS
)
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlations_bar_plot import _single_corr_plot
from src.Correlations.plots_code.single_permutation_test_plot import _single_plot_perm_test, _load_perm_test_res_df

def plot_corr_4x2_grid(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    est_strategy: Literal["CV", "Bootstrap"],
    pred_type: Literal["RT", "comprehension"],
    corr_to_plot: List[str],
    text_cols: List[str],
    ):
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    logger.info(f"{resolution=} | {reading_regime=}")
    
    # define level type
    level_type = 'diff'
    rows = ['reading_speed', 'mean_nonzero_TF', 'SkipRateTotal', 'RegRateTotal']
    rows_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[row]}\n\n$Pearson$ $r$" for row in rows]
    cols = ['Correlations', 'Permutation Test']
    
    perm_test_df = _load_perm_test_res_df(results_dir, est_strategy, resolution)
    if perm_test_df is None:
        return
    corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
    
    # filter text_cols
    corr_df = corr_df[corr_df['text_col'].isin(text_cols)]
    
    # create fig with subplots
    n_rows = 4
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*6.5), sharey=True)
    
    # Set y-label on the left column, set column titles on top row
    for j in range(n_cols):
        axs[0, j].set_title(rows[j], fontsize=12, fontweight='bold')
    
    # Loop over pred_cols, level_types
    for i in range(n_rows):
        pred_col = cols[i]
        for j in range(n_cols):
            ax = axs[i, j]
            if j == 0:
                row_label = rows_labels[i]
                sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'] == level_type)].reset_index(drop=True)
                _single_corr_plot(
                    resolution, ax, i, j, sub_corr_df, 
                    corr_to_plot, pred_col, text_cols, 
                    all_levels=False, est_strategy=est_strategy, y_label=row_label)
            if j == 1:
                sub_perm_test_df = perm_test_df[(perm_test_df['pred_col'] == pred_col) & (perm_test_df['level_type'] == level_type)]
                _single_plot_perm_test(
                    ax, sub_perm_test_df, pred_col, i, 
                    level_type, all_levels=False, resolution=resolution)
        
    # add significance legend
    fig = add_significance_legend(
        fig, 
        significance_colors=SIGNIFICANCE_SIGN_DIFF_COLORS,
        significance_labels=SIGNIFICANCE_SIGN_DIFF_LABELS
        )
        
    # Build a single legend for significance colors
    handles = []
    for sig_symbol, color in SIGNIFICANCE_COLORS.items():
        label = f"{sig_symbol} {SIGNIFICANCE_LABELS[sig_symbol]}"
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)

    fig.legend(
        handles=handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.0), 
        ncol=4, 
        fontsize=9, 
        frameon=False
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # add space between subplots
    plt.subplots_adjust(hspace=1, wspace=0.05)    
    
    output_file = f"grid_4X2_{pred_type}_{resolution}_{est_strategy}.pdf" 
       
    _save_file_to_all_paths(
        resolution=resolution, 
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=None, 
        text_cols=text_cols, 
        corr_to_plot=corr_to_plot, src_path=src_path, est_strategy=est_strategy
        )