import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from src.utils.plot_utils import DELTA
from src.Correlations.define_cols import (
    MAIN_TEXT_COLS, MAIN_SURP_COLS, TEXT_COLS_FULL_LABELS,
)
from src.utils.stat_analysis.stat_utils import p_to_star

def _single_corr_between_readability_measures(
    ax, 
    metrics_df, 
    row_index,
    col_index,
    level_type,
    resolution,
    LevelxSenPar=False,
    axes_fontsize=14,
    ):

    if metrics_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    text_cols = (MAIN_TEXT_COLS+MAIN_SURP_COLS).copy()
    
    n_rows = len(text_cols)
    n_cols = len(text_cols)

    # 2) We'll draw a manual grid of squares for each (row,col)
    #   We'll interpret the "matrix" so x => col, y => row, but by default
    #   axis coordinate => (col, row).
    #   We'll define each square as (col, row) in coordinate, size=1x1.

    # define the axis extent: we want 0..n_cols in x, 0..n_rows in y
    # so each cell is size 1
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # We'll invert_yaxis if we want row 0 at top, or keep as is. 
    ax.invert_yaxis()
    
    done = []

    for r_i, r_name in enumerate(text_cols):
        for c_i, c_name in enumerate(text_cols):
            if r_name == c_name:
                continue
            if (c_name, r_name) in done or (r_name, c_name) in done:
                continue
                
            # calc corr between cols
            col_a_name = f"{level_type}_{r_name}"
            col_b_name = f"{level_type}_{c_name}"
            clean_df = metrics_df[[col_a_name, col_b_name]].dropna()
            # log how many rows where dropped
            dropped = len(metrics_df) - len(clean_df)
            N_SENTENCES_NO_MATCH = 94
            if dropped > N_SENTENCES_NO_MATCH:
                print(f"Dropping {dropped} out of {len(metrics_df)} rows for correlation between {col_a_name} and {col_b_name}")
            pearson_corr, pearson_p = pearsonr(clean_df[col_a_name], clean_df[col_b_name])
            done.append((r_name, c_name))
            
            symbol = p_to_star(pearson_p)
            # if symbol is not string - continue
            if not isinstance(symbol, str):
                continue
            
            # highlight
            lw = 1.0
            # if (c_name == 'Pythia 70M Mean'):
            #     lw = 3.0  # thicker edge

            # color should be heat mapped from 0 to 1 using abs(pearson_corr)
            color_ = plt.cm.RdYlBu_r((abs(pearson_corr) + 1) / 2)
            rect = mpatches.Rectangle(
                (r_i, c_i), 1, 1,  # x,y => bottom-left corner, width=1, height=1
                facecolor=color_,
                edgecolor='black',
                linewidth=lw
            )
            ax.add_patch(rect)
            
            # If we want numeric text from pivot_value
            if pd.notna(pearson_corr):
                # place text in center of cell
                x_center = r_i + 0.5
                y_center = c_i + 0.5
                ax.text(
                    x_center, y_center,
                    round(pearson_corr,1),
                    ha='center', va='center', fontsize=9
                )

    # 3) Set tick labels using short-labeled version
    #   We'll put col ticks at [0.5, 1.5, 2.5,...], row ticks likewise
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    # map row_labels -> short label if present
    row_short = [TEXT_COLS_FULL_LABELS.get(r, r) for r in text_cols]
    col_short = [TEXT_COLS_FULL_LABELS.get(c, c) for c in text_cols]

    ax.set_xticklabels(col_short, rotation=90, fontsize=axes_fontsize)
    ax.set_yticklabels(row_short, fontsize=axes_fontsize)
    
    # if LevelxSenPar:
    #     # if col_index == 0:
    #     #     level_type_labels = {'Adv': 'Original\n', 'Ele': 'Simplified\n', 'diff': f'{DELTA}: Original - Simplified\n'}
    #     #     y_axis_str = f"{level_type_labels[level_type]}\n\n"
    #     #     ax.set_ylabel(y_axis_str, fontsize=axes_fontsize+1, fontweight='bold')    
    #     if row_index == 0:
    #         y_labels = {'sentence': 'Sentences\n\n', 'paragraph': 'Passages\n\n'}
    #         y_axis_str = f"{y_labels[resolution]}\n\n"
    #         ax.set_ylabel(y_axis_str, fontsize=axes_fontsize+1, fontweight='bold')    
    