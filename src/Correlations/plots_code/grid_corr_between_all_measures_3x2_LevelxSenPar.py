import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal
from loguru import logger
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlation_between_readability_measures import _single_corr_between_readability_measures

def plot_all_within_readability_measures_correlation(
        src_path: str,
        reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
        reading_regime: str,
        fontsize_title=16,
        fontsize_legend_text=12,
    ):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info("Plotting correlations_between_all_measures")
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    all_sentences_metrics_df = pd.read_csv(results_dir / "RT_all_metrics_df_sentence.csv")
    all_paragraphs_metrics_df = pd.read_csv(results_dir / "RT_all_metrics_df_paragraph.csv")
    
    dfs = {
        'sentence': all_sentences_metrics_df,
        'paragraph': all_paragraphs_metrics_df,
    }
    
    resolution_types = ['sentence', 'paragraph']
    level_types = ['Adv', 'Ele', 'diff']
    
    # create fig with subplots
    n_rows = len(level_types)
    n_cols = len(resolution_types)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7.5, n_rows*7.5), sharey=True)
    
    # Set y-label on the left column, set column titles on top row
    for j, y_type in enumerate(resolution_types):
        y_labels = {'sentence': 'Sentences\n\n', 'paragraph': 'Paragraphs\n\n'}
        axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')
    
    # Loop
    for i, level_type in enumerate(level_types):
        for j, resolution in enumerate(resolution_types):
            ax = axs[i, j]
            metrics_df = dfs[resolution]
            logger.info(f"Plotting {level_type} x {resolution}")
            _single_corr_between_readability_measures(ax, metrics_df, j, level_type, LevelxSenPar=True)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    output_file = "all_readability_measures_correlations.pdf"
        
    _save_file_to_all_paths(
        resolution=resolution,
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=None, 
        text_cols=None, 
        corr_to_plot=None, src_path=src_path, est_strategy=""
    )
