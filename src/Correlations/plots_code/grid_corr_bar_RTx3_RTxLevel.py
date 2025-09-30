import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal
from loguru import logger

from src.utils.plot_utils import DELTA
from src.constants import PRED_COLS_FULL_LABELS
from src.Correlations.define_cols import TEXT_COLS_FULL_LABELS
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlations_bar_plot import _single_corr_plot
from src.Correlations.plots_code.corr_plot_utils import ( 
    _add_signficance_legend, _filter_corr_df_by_text_cols
    )

def plot_correlations_all_levels(
    src_path: str, 
    resolution: Literal["sentence", "paragraph", "article"],
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str, 
    pred_cols: List[str], 
    text_cols: List[str], 
    corr_to_plot: List[str], 
    output_file: str,
    est_strategy: Literal["Regular", "CV", "Bootstrap"],
    fontsize_title = 14,
    legend_text_fontsize=12,
    text_cols_labels=TEXT_COLS_FULL_LABELS,
    SM_prompts_plot=False,
    orientation: Literal["vertical","horizontal"] = "vertical",
    ):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {resolution} | {reading_regime} | {reader_type} | {pred_cols}")
    corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
    corr_df = _filter_corr_df_by_text_cols(corr_df, text_cols, resolution)
    corr_df['reader_type'] = reader_type
    corr_df['reading_regime'] = reading_regime
    
    level_types = ['Adv', 'Ele', 'diff']
    
    if orientation == "vertical":
        # create fig with subplots
        n_rows = len(pred_cols)
        n_cols = len(level_types)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5.5), sharey=True)

        # Set y-label on the left column, set column titles on top row
        for j, level_type in enumerate(level_types):
            level_type_labels = {'Adv': 'Original\n\n\n', 'Ele': 'Simplified\n\n\n', 'diff': f'{DELTA}: Original - Simplified\n\n\n'}
            axs[0, j].set_title(level_type_labels[level_type], fontsize=fontsize_title, fontweight='bold')
    
    else:
        # create fig with subplots
        n_rows = len(level_types)
        n_cols = len(pred_cols)
        col_length = 6.3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5.5, n_rows*col_length), sharex=True, sharey='row')
        
        # Set y-label on the left column, set row titles on top row
        for j, y_type in enumerate(pred_cols):
            y_labels = {pred_col: f'{PRED_COLS_FULL_LABELS[pred_col]}\n' for pred_col in pred_cols}
            axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')

    
    # Loop over pred_cols, level_types
    for i, pred_col in enumerate(pred_cols):
        for j, level_type in enumerate(level_types):
            ax = axs[i, j] if orientation == "vertical" else axs[j, i]
            row_index = i if orientation == "vertical" else j
            col_index = j if orientation == "vertical" else i
            sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'] == level_type)].reset_index(drop=True)
            if sub_corr_df.empty:
                logger.warning(f"Empty sub_corr_df for {pred_col} | {level_type}")
                continue
            _single_corr_plot(
                resolution, ax, row_index, col_index,
                sub_corr_df, corr_to_plot, pred_col, text_cols, 
                all_levels=True, est_strategy=est_strategy, text_cols_labels=text_cols_labels, 
                SM_prompts_plot=SM_prompts_plot,
                orientation=orientation
                )


    fig = _add_signficance_legend(fig, legend_text_fontsize)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save_file_to_all_paths(resolution, reader_type, reading_regime, output_file, pred_cols, text_cols, corr_to_plot, src_path, est_strategy)
