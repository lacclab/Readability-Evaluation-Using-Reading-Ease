import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import pandas as pd
from typing import List, Literal
from loguru import logger

from src.constants import PRED_COLS_FULL_LABELS
from src.utils.plot_utils import SIGNIFICANCE_COLORS, SIGNIFICANCE_LABELS
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlations_bar_plot import _single_corr_plot


def plot_correlations_diff_levels(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"],
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    pred_cols: List[str],
    text_cols: List[str],
    corr_to_plot: List[str],
    output_file: str,
    est_strategy: Literal["Regular", "CV", "Bootstrap"] = 'Regular',
    fontsize_title = 14,
    legend_text_fontsize=12,
    ):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr,
    #   pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {resolution} | {reading_regime} | {reader_type} | {pred_cols}")
    corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
    
    # filter text_cols
    corr_df = corr_df[corr_df['text_col'].isin(text_cols)]
    
    # define level type
    level_type = 'diff'
    
    first_row_cols = ['reading_speed', 'mean_nonzero_TF']
    first_row_cols_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[col]}\n\n" for col in first_row_cols]
    second_row_cols = ['SkipRateTotal', 'RegRateTotal']
    second_row_cols_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[col]}\n\n" for col in second_row_cols]
    
    # create fig with subplots
    n_rows = 2
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5.5), sharey=True)
    # fig.suptitle(f'Alignment Resolution: {resolution}')

    # Set y-label on the left column, set column titles on top row
    for j in range(n_cols):
        axs[0, j].set_title(first_row_cols_labels[j], fontsize=fontsize_title, fontweight='bold')
        axs[1, j].set_title(second_row_cols_labels[j], fontsize=fontsize_title, fontweight='bold')
    
    # Loop over pred_cols, level_types
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            pred_col = first_row_cols[j] if i == 0 else second_row_cols[j]
            sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'] == level_type)].reset_index(drop=True)
            _single_corr_plot(
                resolution, ax, i, j, sub_corr_df, 
                corr_to_plot, pred_col, text_cols, 
                all_levels=False, est_strategy=est_strategy)

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
        fontsize=legend_text_fontsize, 
        frameon=False
    )
    
    if len(corr_to_plot) > 1:
        # 5) A small legend in the top-right
        #  We'll define two patches for Pearson vs. Spearman
        pear_patch = Patch(facecolor='white', edgecolor='black', hatch='',    label='pearson_corr')
        spear_patch= Patch(facecolor='white', edgecolor='black', hatch='///', label='spearman_corr')
        if corr_to_plot == ["pearson_corr"]:
            handels = [pear_patch]
        elif corr_to_plot == ["spearman_corr"]:
            handels = [spear_patch]
        else:
            handels = [pear_patch, spear_patch]
            
        fig.legend(
            handles=handels,
            loc='lower right',
            fontsize=8,
            # title="Correlation Type"
        )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # add space between subplots
    plt.subplots_adjust(hspace=1.5, wspace=0.05)    
        
    _save_file_to_all_paths(resolution, reader_type, reading_regime, output_file, pred_cols, text_cols, corr_to_plot, src_path, est_strategy)
