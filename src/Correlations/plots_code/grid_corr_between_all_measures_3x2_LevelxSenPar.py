import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal
from loguru import logger
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlation_between_readability_measures import _single_corr_between_readability_measures
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

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
    level_types = ['diff']
    
    # create fig with subplots
    n_cols = len(resolution_types)
    n_rows = len(level_types)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10.5)) # sharey=True
    
    # Set y-label on the left column, set column titles on top row
    for j, y_type in enumerate(resolution_types):
        y_labels = {'sentence': 'Sentences\n\n', 'paragraph': 'Passages\n\n'}
        axs[j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')
        # axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')

    # # Set y-label on the left column, set column titles on top row
    # for i, y_type in enumerate(level_types):
    #     y_labels = {'diff': f'{DELTA}: Original - Simplified\n'}
    #     axs[i].set_title(y_labels['diff'], fontsize=fontsize_title, fontweight='bold')   
    
    # Loop
    for i, level_type in enumerate(level_types):
        for j, resolution in enumerate(resolution_types):
            # ax = axs[i, j]
            ax = axs[j]
            metrics_df = dfs[resolution]
            logger.info(f"Plotting {level_type} x {resolution}")
            _single_corr_between_readability_measures(ax, metrics_df, i, j, level_type, resolution, LevelxSenPar=True)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # define full colormap (-1 to 1)
    full_cmap = cm.get_cmap('RdYlBu_r')

    # truncate to the upper half (corresponding to 0→1 correlations)
    def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap

    cmap = truncate_colormap(full_cmap, 0.5, 1.0)
    norm = plt.Normalize(0, 1)
    
    # create scalar mappable for colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # place colorbar on the right side (vertical)
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.035, pad=0.1)
    cbar.set_label('Pearson Correlation (0–1)', fontsize=fontsize_legend_text, fontweight='bold')
    cbar.ax.tick_params(labelsize=fontsize_legend_text)
    
    # # place colorbar at the bottom (horizontal)
    # cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.035, pad=-0.1)
    # # show only ticks 0–1, even though colors are from -1–1
    # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # cbar.set_label('Pearson Correlation (0–1)', fontsize=fontsize_legend_text, fontweight='bold')
    # cbar.ax.tick_params(labelsize=fontsize_legend_text)

    # adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
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

if __name__ == "__main__":
    from pathlib import Path

    src_path = Path.cwd() / "src"
    L1_or_L2 = "L1_and_L2" # "L1" or "L2" or "L1_and_L2"
    
    plot_all_within_readability_measures_correlation(
        src_path=src_path,
        reader_type=L1_or_L2,
        reading_regime="FirstReading",
    )
