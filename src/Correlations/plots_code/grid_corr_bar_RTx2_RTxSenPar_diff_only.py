import matplotlib.pyplot as plt
from typing import List, Literal
from loguru import logger
import pandas as pd

from src.constants import PRED_COLS_FULL_LABELS
from src.utils.plot_utils import add_significance_legend, SIGNIFICANCE_SIGN_DIFF_COLORS, SIGNIFICANCE_SIGN_SLOPE_LABELS
from src.Correlations.define_cols import TEXT_COLS_FULL_LABELS
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlations_bar_plot import (
    _single_corr_plot, _pair_bars_or_not
)
from src.Correlations.plots_code.corr_plot_utils import ( 
    _add_legend_for_hatch, _add_signficance_legend, _load_corr_df, HATCH_STR_DICT_LABELS, _add_bin_names_legend
    )
from src.utils.files_utils import replace_results_in_file

def plot_corr_grid_RTx2_RTxSenPar_diff_only(
    src_path: str,
    reader_type: str,
    reading_regime: str,
    pred_cols: List[str],
    text_cols: List[str],
    corr_to_plot: List[str],
    output_file: str,
    est_strategy: Literal["Regular", "CV", "Bootstrap"],
    fontsize_title = 16,
    legend_text_fontsize=12,
    text_cols_labels=TEXT_COLS_FULL_LABELS,
    SM_prompts_plot=False,
    main_plot=False,
    orientation: Literal["vertical","horizontal"] = "vertical",
    bins_on_x: bool = False,
    debug_mode: bool = False
):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {reading_regime} | {reader_type} | {pred_cols}")
    sentences_corr_df = _load_corr_df(reader_type, reading_regime, resolution='sentence', src_path=src_path, text_cols=text_cols)
    paragraphs_corr_df = _load_corr_df(reader_type, reading_regime, resolution='paragraph', src_path=src_path, text_cols=text_cols)
    
    L1_next_to_L2=True if reader_type == "L1_next_to_L2" else False
    Gathering0_next_to_Hunting0=True if reading_regime == "Gathering0_next_to_Hunting0" else False
    FirstReading_next_to_Gathering0=True if reading_regime == "FirstReading_next_to_Gathering0" else False

    resolution_types = ['sentence', 'paragraph']
    level_type = 'diff'
    
    if orientation == "vertical":
        # create fig with subplots
        n_rows = len(pred_cols)
        n_cols = len(resolution_types)
        
        if bins_on_x:
            row_length = 8
        else:
            row_length = 7
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*row_length, n_rows*5.5), sharey=True)
        
        # Set y-label on the left column, set column titles on top row
        for j, y_type in enumerate(resolution_types):
            y_labels = {'sentence': 'Sentences\n\n', 'paragraph': 'Passages\n\n'}
            axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')
    else:
        # create fig with subplots
        n_rows = len(resolution_types)
        n_cols = len(pred_cols)
        if main_plot:
            col_length = 7
        elif SM_prompts_plot:
            col_length = 9
        else:
            col_length = 8
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5.5, n_rows*col_length), sharex=True, sharey='row')
        
        # Set y-label on the left column, set row titles on top row
        for j, y_type in enumerate(pred_cols):
            y_labels = {pred_col: f'{PRED_COLS_FULL_LABELS[pred_col]}\n' for pred_col in pred_cols}
            axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')

    # Loop over pred_cols, level_types
    for i, pred_col in enumerate(pred_cols):
        for j, resolution in enumerate(resolution_types):
            ax = axs[i, j] if orientation == "vertical" else axs[j, i]
            row_index = i if orientation == "vertical" else j
            col_index = j if orientation == "vertical" else i
            corr_df = sentences_corr_df if resolution == 'sentence' else paragraphs_corr_df
            sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'] == level_type)].reset_index(drop=True)
            if sub_corr_df.empty:
                logger.warning(f"Empty sub_corr_df for {pred_col} at {resolution} level. Skipping...")
                continue
            

            pair_bars = _pair_bars_or_not(    
                L1_next_to_L2,  
                len(corr_to_plot),  
                Gathering0_next_to_Hunting0, 
                FirstReading_next_to_Gathering0,
                reading_regime,
                )
            _single_corr_plot(
                resolution, ax, row_index, col_index, 
                sub_corr_df, corr_to_plot, pred_col, text_cols, 
                all_levels=True, est_strategy=est_strategy, text_cols_labels=text_cols_labels, 
                SM_prompts_plot=SM_prompts_plot, main_plot=main_plot,
                L1_next_to_L2=L1_next_to_L2,
                Gathering0_next_to_Hunting0=Gathering0_next_to_Hunting0,
                FirstReading_next_to_Gathering0=FirstReading_next_to_Gathering0,
                orientation=orientation,
                pair_bars=pair_bars
            )

    if L1_next_to_L2:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['L1_next_to_L2'])
    if Gathering0_next_to_Hunting0:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['Gathering0_next_to_Hunting0'])
    if FirstReading_next_to_Gathering0:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['FirstReading_next_to_Gathering0'])
    if len(corr_to_plot) > 1 and 'pearson' in corr_to_plot[0].lower() and 'spearman' in corr_to_plot[1].lower():
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['pearson_spearman'])
        
        
    _save_file_to_all_paths(resolution, reader_type, reading_regime, output_file, pred_cols, text_cols, corr_to_plot, src_path, est_strategy)



