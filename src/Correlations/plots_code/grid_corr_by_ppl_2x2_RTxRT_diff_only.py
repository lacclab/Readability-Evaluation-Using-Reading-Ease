import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal
from loguru import logger

from src.constants import PRED_COLS_FULL_LABELS
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlation_by_ppl_plot import _single_corr_by_perplexity_plot, _build_legend_ppl_plot
from src.Correlations.analysis.models_ppl import _get_models_data


def plot_corr_by_perplexity_diff_levels(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"],
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    pred_cols: List[str], 
    surp_cols: List[str], 
    corr_to_plot: List[str], 
    output_file: str,
    est_strategy: Literal["Regular", "CV", "Bootstrap"] = 'Regular',
    fontsize_title=20,
    fontsize_legend_text=16,
    markzise_legend=12
):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr,
    #   pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {resolution} | {reading_regime} | {reader_type} | {pred_cols}")
    corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
    # get models data
    surp_to_ppl, surp_to_family, surp_to_model_name_with_size = _get_models_data(src_path)
    
    # filter text_cols
    corr_df = corr_df[corr_df['text_col'].isin(surp_cols)]
    
    # define level type
    level_type = 'diff'
    
    first_row_cols = ['reading_speed', 'mean_nonzero_TF']
    first_row_cols_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[col]}" for col in first_row_cols]
    second_row_cols = ['SkipRateTotal', 'RegRateTotal']
    second_row_cols_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[col]}" for col in second_row_cols]
    
    
    n_rows = 2
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*6.5, n_rows*5), sharey=True)

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
            _ = _single_corr_by_perplexity_plot(
                ax, j, sub_corr_df, corr_to_plot, pred_col, 
                surp_to_model_name_with_size, surp_to_family, surp_to_ppl, 
                all_levels=False, est_strategy=est_strategy)
            
    fig = _build_legend_ppl_plot(fig, surp_cols, corr_df, surp_to_family,
                                 fontsize_legend_text, markzise_legend)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    
    # make space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.05)
        
    _save_file_to_all_paths(
        resolution=resolution, 
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=pred_cols, 
        text_cols=surp_cols, 
        corr_to_plot=corr_to_plot, src_path=src_path, est_strategy=est_strategy
        )
   