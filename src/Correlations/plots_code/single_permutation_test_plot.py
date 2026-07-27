import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger

from src.constants import PRED_COLS_SHORT_LABELS
from src.utils.plot_utils import SIGNIFICANCE_SIGN_DIFF_COLORS, DELTA
from src.Correlations.define_cols import (
    MAIN_TEXT_COLS, MAIN_SURP_COLS, TEXT_COLS_FULL_LABELS,
)
from src.constants import PRED_COLS_FULL_LABELS

def _single_plot_perm_test(
    ax, sub_perm_test_df, steiger_res,
    pred_col, 
    row_index, col_index,
    level_type, all_levels, resolution,
    RTxSenPar=False,
    ax_title_fontsize=14,
    axes_fontsize=14,
):
    """
    sub_perm_test_df columns (one row per pair):
      text_col_1, mean_corr_1, std_corr_1, n_noNan_1
      text_col_2, mean_corr_2, std_corr_2, n_noNan_2
      perm_p, perm_stat, perm_p_symbol
      2_is_bigger
    """

    if sub_perm_test_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return

    df_with_corr_compare_test = steiger_res # instead of sub_perm_test_df
    df_with_corr_diff = sub_perm_test_df
    
    # 1) pivots
    pivot_symbol = df_with_corr_compare_test.pivot(
        index='text_col_1', columns='text_col_2', values='p_val_symbol'
    )
    pivot_value = df_with_corr_diff.pivot(
        index='text_col_1', columns='text_col_2', values='2_is_bigger'
    )

    text_cols = (MAIN_TEXT_COLS + MAIN_SURP_COLS).copy()

    n_rows = len(text_cols)
    n_cols = len(text_cols)

    # 2) grid setup
    ax.set_aspect("equal")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    
    def swap_val(val):
        if val == '+':
            return '-'
        elif val == '-':
            return '+'
        else:
            return val

    def lookup(rc, cc, table):
        """Return (symbol, val) from (r,c) or (c,r); else (None, None)."""
        if (rc in table.index) and (cc in table.columns):
            val = table.loc[rc, cc]
            if isinstance(val, str) and not pd.isna(val):
                val = swap_val(val)
                return val
        if (cc in table.index) and (rc in table.columns):
            val = table.loc[cc, rc]
            if isinstance(val, str) and not pd.isna(val):
                return val
        return None, None

    for r_i, r_name in enumerate(text_cols):
        for c_i, c_name in enumerate(text_cols):
            # ---- draw ONLY lower triangle ----
            if r_i <= c_i:
                continue

            # fetch symbol/val (try symmetric if missing)
            val = lookup(r_name, c_name, pivot_value)
            if not isinstance(val, str) or pd.isna(val):
                val = lookup(r_name, c_name, pivot_value)
                continue
            
            symbol = lookup(r_name, c_name, pivot_symbol)
            if not isinstance(symbol, str) or pd.isna(symbol):
                symbol = lookup(r_name, c_name, pivot_symbol)
                continue


            key = f"{symbol} {val}"
            color_ = SIGNIFICANCE_SIGN_DIFF_COLORS.get(key)
            if color_ is None:
                continue

            lw = 3.0 if (r_name == 'Pythia 70M Mean' or c_name == 'Pythia 70M Mean') else 1.0

            # draw rectangle
            rect = mpatches.Rectangle(
                (c_i, r_i), 1, 1, facecolor=color_, edgecolor='black', linewidth=lw
            )
            ax.add_patch(rect)

    # 3) ticks & labels
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    row_short = [TEXT_COLS_FULL_LABELS.get(r, r) for r in text_cols]
    col_short = [TEXT_COLS_FULL_LABELS.get(c, c) for c in text_cols]

    ax.set_xticklabels(col_short, rotation=90, fontsize=axes_fontsize)
    ax.set_yticklabels(row_short, fontsize=axes_fontsize)

    if RTxSenPar and col_index == 0:
        pred_col_str = f"{PRED_COLS_FULL_LABELS[pred_col]}\n\n"
        ax.set_ylabel(pred_col_str, fontsize=axes_fontsize+1, fontweight='bold')

    if all_levels:
        level_type_labels = {'Adv': 'Original\n', 'Ele': 'Simplified\n', 'diff': f'{DELTA}: Original - Simplified\n'}
        title = f"{level_type_labels[level_type]}\n\n{PRED_COLS_SHORT_LABELS[pred_col]}" if row_index == 0 \
                else f"{PRED_COLS_SHORT_LABELS[pred_col]}"
        ax.set_title(title, fontsize=ax_title_fontsize, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.2)



def _load_perm_test_res_df(results_dir, est_strategy, resolution):
    # load perm_test_df
    try:
        perm_path = results_dir / f"perm_test_{resolution}_{est_strategy}.csv"
        perm_test_df = pd.read_csv(perm_path)
    except FileNotFoundError:
        logger.error(f"File not found: {perm_path}")
        return
    
    # columns: pred_col, level_type, text_col_1, mean_corr_1, std_corr_1, n_noNan_1, text_col_2, mean_corr_2, std_corr_2, n_noNan_2, perm_p, perm_stat
    # filter only perm_p != None
    perm_test_df = perm_test_df[perm_test_df['perm_p'].notna()]
    # 1 bigger then 2
    perm_test_df['2_is_bigger'] = perm_test_df['mean_corr_1'] < perm_test_df['mean_corr_2']
    # convert True -> + and False -> -
    perm_test_df['2_is_bigger'] = perm_test_df['2_is_bigger'].apply(lambda x: '+' if x else '-')
    return perm_test_df

    