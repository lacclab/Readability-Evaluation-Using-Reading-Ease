import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal
from loguru import logger

from src.utils.plot_utils import DELTA
from src.constants import PRED_COLS_FULL_LABELS
from src.Correlations.define_cols import READING_COMPREHENSION_COLS, TEXT_COLS_FULL_LABELS
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
    all_and_diff=False,
    pred_type="RT",
    level_labels_override: dict = None,
    ):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {resolution} | {reading_regime} | {reader_type} | {pred_cols}")
    corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
    corr_df = _filter_corr_df_by_text_cols(corr_df, text_cols, resolution)
    corr_df['reader_type'] = reader_type
    corr_df['reading_regime'] = reading_regime
    
    if all_and_diff:
        level_types = ['all', 'diff']
    else:
        level_types = ['Adv', 'Ele', 'diff']
    
    # Set default level labels override for comprehension
    if level_labels_override is None and pred_type == "comprehension":
        level_labels_override = {'all': 'Not Controlled\n\n', 'diff': 'Controlled\n\n'}
    
    if orientation == "vertical":
        # create fig with subplots
        n_rows = len(pred_cols)
        n_cols = len(level_types)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5.5), sharey=True)

        # Set y-label on the left column, set column titles on top row
        for j, level_type in enumerate(level_types):
            if all_and_diff:
                if level_labels_override is not None:
                    level_type_labels = {k: v.rstrip('\n') + '\n\n\n' for k, v in level_labels_override.items()}
                else:
                    level_type_labels = {'all': 'Original and Simplified\n\n\n', 'diff': f'{DELTA}: Original - Simplified\n\n\n'}
            else:
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
                resolution=resolution, ax=ax, row_index=row_index, col_index=col_index,
                sub_corr_df=sub_corr_df, sub_corr_boot_df=None, corr_to_plot=corr_to_plot, pred_col=pred_col, text_cols=text_cols,
                all_levels=True, est_strategy=est_strategy, text_cols_labels=text_cols_labels,
                SM_prompts_plot=SM_prompts_plot,
                orientation=orientation,
                all_and_diff=all_and_diff,
                level_labels_override=level_labels_override
                )


    fig = _add_signficance_legend(fig, legend_text_fontsize)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save_file_to_all_paths(resolution, reader_type, reading_regime, output_file, pred_cols, text_cols, corr_to_plot, src_path, est_strategy)

if __name__ == "__main__":
    from pathlib import Path
    from src.Correlations.define_cols import (
        MAIN_RT_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS
    )

    src_path = Path.cwd() / "src"
    

    # try_main_RT_cols = ['reading_speed', 'mean_nonzero_TF', 'SkipRateTotal', 'RegRateTotal']
    # resolution = "article"  # "sentence", "paragraph", "article"
    # plot_correlations_all_levels(
    #         src_path=src_path,
    #         resolution=resolution, 
    #         reader_type="L1", 
    #         reading_regime="Gathering0", 
    #         pred_cols=MAIN_RT_COLS, 
    #         text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
    #         corr_to_plot=["pearson_corr"],
    #         output_file=f"SM_all_levels_pearson_corr_{resolution}.pdf",
    #         est_strategy="Bootstrap"
    #     )
    
    plot_correlations_all_levels(
        src_path,
        resolution="paragraph", 
        reader_type="L1_and_L2", 
        reading_regime="FirstReading", 
        pred_cols=READING_COMPREHENSION_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file="comprehension_corr_paragraph.pdf",
        est_strategy="Bootstrap",
        orientation="horizontal"
    )