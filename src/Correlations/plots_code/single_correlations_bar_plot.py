import pandas as pd
import numpy as np
from typing import Literal
from src.constants import PRED_COLS_FULL_LABELS
from src.utils.plot_utils import SIGNIFICANCE_COLORS
from src.Correlations.calc_correlations import N_BOOTSTRAP
from src.Correlations.plots_code.corr_plot_utils import HATCH_STR_DICT, OFFSET_DICT
from src.Correlations.define_cols import (
    TEXT_COLS_FULL_LABELS, TEXT_COL_TO_YEAR, 
    TRADITIONAL_MEASURES, MODERN_MEASURES, SYSTEMS_MEASURES, 
    MAIN_PROMPT_COLS, PSYCHOLINGUISTIC_MEASURES,
    SCORE_PROMPT_LABEL, GRADE_PROMPT_LABEL, SCORE_SPECIFIC_PROMPT_LABEL, GRADE_SPECIFIC_PROMPT_LABEL, PROMPT_LABEL_ORDER, PROMPT_LABEL_ORDER_REVERSED
)
from loguru import logger
from src.utils.plot_utils import DELTA

LOCATION_PROMPT_CATEGORIES = -0.62
LOCATION_READABILITY_CATEGORIES = - 0.62
LOCATION_READABILITY_CATEGORIES_MAIN_PLOT = - 0.75
LOCATION_YEARS__MAIN_PLOT = -0.55
WIDTH_PAIR_BARS = 0.4
WIDTH_BARS = 0.5

# =========================
# ENTRY POINTS
# =========================

def _single_corr_plot(
    resolution, 
    ax, row_index, col_index, 
    sub_corr_df, 
    sub_corr_boot_df,
    corr_to_plot, 
    pred_col, 
    text_cols, 
    all_levels, 
    est_strategy: Literal["Regular", "CV", "Bootstrap"] = 'Regular',
    y_label = None,
    axes_fontsize=11,
    subtitle_fontsize=11,
    text_cols_labels=TEXT_COLS_FULL_LABELS,
    SM_prompts_plot=False,
    main_plot=False,
    L1_next_to_L2=False,
    Gathering0_next_to_Hunting0=False,
    FirstReading_next_to_RepeatedReading=False,
    FirstReading_next_to_Gathering0=False,
    Pearson_next_to_Spearman=False,
    RE_next_to_delta_RE=False,
    orientation: Literal["vertical","horizontal"]="vertical",
    pair_bars=False,
    perm_test_res=None,
    steiger_res=None,
    need_to_calc_perm_test=False,
    all_and_diff=False,
    SM_TEXT_plot=False,
    ): 
    corr_cols = _get_corr_cols_by_est_strategy(est_strategy)
    corr_metas = _get_corr_metas(corr_to_plot, corr_cols)
    n_corrs = len(corr_metas)
    sub_corr_df = _preprocess_sub_corr_df(sub_corr_df, text_cols, text_cols_labels, SM_prompts_plot, orientation, RE_next_to_delta_RE)
    if pair_bars:
        corr_diffs, _ = _get_corr_diffs(
            sub_corr_df, sub_corr_boot_df, corr_cols['pearson_col'], corr_cols['spearman_col'], 
            L1_next_to_L2, 
            Gathering0_next_to_Hunting0, 
            FirstReading_next_to_RepeatedReading, 
            FirstReading_next_to_Gathering0, 
            RE_next_to_delta_RE,
            n_corrs, need_to_calc_perm_test)
        axes_fontsize = 13
        subtitle_fontsize = 13

    # 1) positions for each text_col (become x for vertical, y for horizontal)
    existing_text_cols = sub_corr_df['text_col'].unique()
    exisiting_text_cols_labels = sub_corr_df['text_col_label'].unique()
    n = len(existing_text_cols)
    cat_positions = np.arange(n)
    position_by_text_col = {col: i for i, col in enumerate(existing_text_cols)}

    # value axis & grid
    _set_value_axis_lim_and_ticks(
        ax, col_index, axes_fontsize, 
        all_levels, orientation, main_plot, 
        pair_bars=pair_bars, 
        Pearson_next_to_Spearman=Pearson_next_to_Spearman, 
        RE_next_to_delta_RE=RE_next_to_delta_RE
    )
    _add_value_axis_lines(ax, col_index, zorder=1, orientation=orientation, main_plot=main_plot)
    _set_value_axis_label(ax, y_label, row_index, col_index, all_levels, all_and_diff, corr_to_plot, pred_col, axes_fontsize, orientation, main_plot)

    for L1_L2_val, group_df in sub_corr_df.groupby('reader_type'):
        for reading_regime, group_df in group_df.groupby('reading_regime'):
            for level, group_df in group_df.groupby('level_type'):
                for _, row in group_df.iterrows():
                    for corr_col, symbol_col in corr_metas:
                        if corr_col not in row or symbol_col not in row:
                            continue
                        if row['n_bootstraps'] < N_BOOTSTRAP and est_strategy == "Bootstrap":
                            logger.warning(f"Low N bootstraps ({row['n_bootstraps']}) for {row['text_col']} - {pred_col} - {resolution} - {row['reader_type']}.")
                        
                        offset, hatch, width, capsize, plotwithout_CI = _get_offset_and_hatch(
                            L1_next_to_L2, L1_L2_val, level,
                            n_corrs, corr_col,
                            Gathering0_next_to_Hunting0, FirstReading_next_to_RepeatedReading,
                            FirstReading_next_to_Gathering0,
                            RE_next_to_delta_RE,
                            reading_regime, pair_bars
                        )
                        text_col = row['text_col']
                        if pair_bars:
                            corr_diff = corr_diffs[row['text_col']] if corr_diffs else 0
                        else:
                            corr_diff = None

                        corr_val = abs(row[corr_col])
                        signif   = row[symbol_col]
                        color_   = SIGNIFICANCE_COLORS.get(signif, SIGNIFICANCE_COLORS['ns'])
                        cat_pos  = position_by_text_col[row['text_col']]
                        cat_pos_with_offset = cat_pos + offset
                        year = TEXT_COL_TO_YEAR.get(row['text_col'], "")

                        _add_bar(
                            ax, cat_pos_with_offset, width, capsize, row, 
                            corr_cols, corr_col, corr_val, 
                            color_, hatch, est_strategy, zorder=10, plotwithout_CI=plotwithout_CI,
                            orientation=orientation
                        )
                        _add_value_text(
                            ax, col_index, corr_val, all_levels,
                            cat_pos, cat_pos_with_offset,
                            year, main_plot, corr_diff,
                            orientation=orientation,
                            pair_bars=pair_bars,
                            perm_test_res=perm_test_res,
                            steiger_res=steiger_res,
                            text_col=text_col,
                            Pearson_next_to_Spearman=Pearson_next_to_Spearman,
                            RE_next_to_delta_RE=RE_next_to_delta_RE
                        )

    _add_category_separators(ax, col_index, resolution, SM_prompts_plot, subtitle_fontsize, orientation, main_plot, pair_bars, SM_TEXT_plot)
    _set_text_cols_ticks(ax, exisiting_text_cols_labels, cat_positions, axes_fontsize, SM_prompts_plot, orientation)

    return None


# ---------
# Helpers
# ---------

def _get_corr_diffs(
    sub_corr_df, sub_corr_boot_df, 
    pearson_col, spearman_col, 
    L1_next_to_L2, 
    Gathering0_next_to_Hunting0, 
    FirstReading_next_to_RepeatedReading, 
    FirstReading_next_to_Gathering0,
    RE_next_to_delta_RE,
    n_corrs, need_to_calc_perm_test
    ):
    if Gathering0_next_to_Hunting0:
        # calc using pivot
        pivot_df = sub_corr_df.pivot(index='text_col', columns='reading_regime', values=pearson_col)
        diffs_series = pivot_df['Gathering0'] - pivot_df['Hunting0']
        return diffs_series.to_dict(), None
    
    elif FirstReading_next_to_RepeatedReading:
        # calc using pivot
        pivot_df = sub_corr_df.pivot(index='text_col', columns='reading_regime', values=pearson_col)
        diffs_series = pivot_df['FirstReading'] - pivot_df['RepeatedReading']
        return diffs_series.to_dict(), None
    elif FirstReading_next_to_Gathering0:
        # calc using pivot
        pivot_df = sub_corr_df.pivot(index='text_col', columns='reading_regime', values=pearson_col)
        diffs_series = pivot_df['FirstReading'] - pivot_df['Gathering0']
        return diffs_series.to_dict(), None
    elif L1_next_to_L2:      
        # calc using pivot
        pivot_df = sub_corr_df.pivot(index='text_col', columns='reader_type', values=pearson_col)
        diffs_series = pivot_df['L2'] - pivot_df['L1']
        return diffs_series.to_dict(), None
    elif n_corrs == 2:
        diffs_series = sub_corr_df.set_index('text_col')[pearson_col] - sub_corr_df.set_index('text_col')[spearman_col]
        return diffs_series.to_dict(), None
    elif RE_next_to_delta_RE:
        # calc using pivot
        pivot_df = sub_corr_df.pivot(index='text_col', columns='level_type', values=pearson_col)
        diffs_series = pivot_df['all'] - pivot_df['diff']
        return diffs_series.to_dict(), None
    else:
        return False, None

def _pair_bars_or_not(    
    L1_next_to_L2,  
    Pearson_next_to_Spearman,  
    Gathering0_next_to_Hunting0, 
    FirstReading_next_to_RepeatedReading,
    FirstReading_next_to_Gathering0,
    RE_next_to_delta_RE
    ):
    if (L1_next_to_L2 or 
        Gathering0_next_to_Hunting0 or 
        FirstReading_next_to_RepeatedReading or 
        FirstReading_next_to_Gathering0 or
        Pearson_next_to_Spearman or
        RE_next_to_delta_RE
        ):
        return True
    else:
        return False

def _get_offset_and_hatch(
    L1_next_to_L2, L1_L2_val, level,
    n_corrs, corr_col, 
    Gathering0_next_to_Hunting0, 
    FirstReading_next_to_RepeatedReading,
    FirstReading_next_to_Gathering0,
    RE_next_to_delta_RE,
    reading_regime, pair_bars
    ):
    if pair_bars:
        plotwithout_CI = False
        capsize = 1
    else:
        plotwithout_CI = False
        capsize = 2
    
    if L1_next_to_L2:
        hatch = HATCH_STR_DICT['L1_next_to_L2'][L1_L2_val]
        offset = OFFSET_DICT['L1_next_to_L2'][L1_L2_val]
    elif Gathering0_next_to_Hunting0:
        hatch = HATCH_STR_DICT['Gathering0_next_to_Hunting0'][reading_regime]
        offset = OFFSET_DICT['Gathering0_next_to_Hunting0'][reading_regime]
    elif FirstReading_next_to_RepeatedReading:
        hatch = HATCH_STR_DICT['FirstReading_next_to_RepeatedReading'][reading_regime]
        offset = OFFSET_DICT['FirstReading_next_to_RepeatedReading'][reading_regime]
    elif FirstReading_next_to_Gathering0:
        hatch = HATCH_STR_DICT['FirstReading_next_to_Gathering0'][reading_regime]
        offset = OFFSET_DICT['FirstReading_next_to_Gathering0'][reading_regime]
    elif n_corrs == 2:
        hatch = HATCH_STR_DICT['pearson_spearman']['Pearson'] if 'pearson' in corr_col else HATCH_STR_DICT['pearson_spearman']['Spearman']
        offset = OFFSET_DICT['pearson_spearman']['Pearson'] if 'pearson' in corr_col else OFFSET_DICT['pearson_spearman']['Spearman']
    elif RE_next_to_delta_RE:
        hatch = HATCH_STR_DICT['RE_next_to_delta_RE'][level]
        offset = OFFSET_DICT['RE_next_to_delta_RE'][level]
    else:
        offset = 0
        hatch = ''
    
    if pair_bars:
        width = WIDTH_PAIR_BARS
        
    else:
        width = WIDTH_BARS

    return offset, hatch, width, capsize, plotwithout_CI

def _preprocess_sub_corr_df(sub_corr_df, text_cols, text_cols_labels, SM_prompts_plot, orientation, RE_next_to_delta_RE):
    # assign id to each col in text_cols
    text_cols_ids = {col: i for i, col in enumerate(text_cols)}
    sub_corr_df['text_col_id'] = sub_corr_df['text_col'].map(text_cols_ids)
    # order df by text_col_id
    sub_corr_df = sub_corr_df.sort_values(by='text_col_id').reset_index(drop=True)
    # if orientation is horizontal then reverse the order
    if orientation == "horizontal":
        sub_corr_df = sub_corr_df.iloc[::-1].reset_index(drop=True)
    
    # replace text_col with text_col_label 
    sub_corr_df['text_col_label'] = sub_corr_df['text_col'].map(text_cols_labels)
    
    if SM_prompts_plot:
        sub_corr_df = _order_by_prompt_category(sub_corr_df, orientation)
        
    if RE_next_to_delta_RE:
        # keep only level 'all' and 'diff'
        sub_corr_df = sub_corr_df[sub_corr_df['level_type'].isin(['all', 'diff'])].reset_index(drop=True)
        
    return sub_corr_df

def _add_bar(
    ax, cat_pos_with_offset, width, capsize, row, 
    corr_cols, corr_col, corr_val, 
    color_, hatch_str, est_strategy, zorder, plotwithout_CI=False,
    orientation: Literal["vertical","horizontal"]="vertical",
    ):
    if est_strategy == "Regular" or plotwithout_CI:
        if orientation == "vertical":
            ax.bar(
                cat_pos_with_offset, corr_val,
                width=width, color=color_, hatch=hatch_str,
                edgecolor='black', zorder=zorder
            )
        else:
            ax.barh(
                cat_pos_with_offset, corr_val,
                height=width, color=color_, hatch=hatch_str,
                edgecolor='black', zorder=zorder
            )
    else:
        CI = row[corr_cols['pearson_CI_yerr_col']] if corr_col == "pearson_corr" else row[corr_cols['spearman_CI_yerr_col']]
        if orientation == "vertical":
            ax.bar(
                cat_pos_with_offset, corr_val,
                width=width, color=color_, hatch=hatch_str,
                edgecolor='black', yerr=CI, capsize=capsize
            )
        else:
            # barh uses xerr for horizontal error bars
            ax.barh(
                cat_pos_with_offset, corr_val,
                height=width, color=color_, hatch=hatch_str,
                edgecolor='black', xerr=CI, error_kw={'capsize': capsize}
            )
    return ax

def _add_value_text(
    ax, col_index, corr_val, 
    all_levels, 
    cat_pos, cat_pos_with_offset, 
    year, main_plot, corr_diff, orientation,
    pair_bars, perm_test_res, steiger_res,
    text_col,
    Pearson_next_to_Spearman, RE_next_to_delta_RE
    ):
    fontsize = 9 if all_levels else 8
    if pair_bars:
        fontsize = 10
    color = 'grey'
    rotation = 90 if orientation == "vertical" else 0

    if main_plot:
        text = year
        if orientation == "vertical":
            ax.text(cat_pos, 1.03, text, ha='center', va='bottom', fontsize=fontsize, rotation=rotation, color='#57534D')
        else:
            if col_index == 0:
                ax.text(LOCATION_YEARS__MAIN_PLOT, cat_pos, f"{year}", ha='left', va='center', fontsize=fontsize)
        return ax

    # show diffs when offset groups are present
    elif pair_bars:
        
        if Pearson_next_to_Spearman or RE_next_to_delta_RE:
            symbol_star = 'ns'
        else:
            curr_steiger_res = steiger_res[steiger_res['text_col'] == text_col]
            symbol_star = curr_steiger_res['p_val_symbol'].item()
        
        
        if symbol_star == 'ns':
            symbol_star_print = ''
        else:
            symbol_star_print = f"{symbol_star}"
            
        text = f"{corr_diff:.2f} {symbol_star_print}"
        fontsize = fontsize + 1
        
        # if abs(corr_diff) >= 0.1:
        if symbol_star != 'ns': # highlight significant differences
            color = '#39843B' if corr_diff > 0 else '#2652B0'
            fontweight='bold'
            # fontsize += 0.5
        else:
            color = 'grey'
            fontweight='normal'
            # fontsize -= 0.5
        if Pearson_next_to_Spearman or RE_next_to_delta_RE:
            color = 'black'
        if orientation == "vertical":
            ax.text(cat_pos, 1.03, text, ha='center', va='bottom', fontsize=fontsize, rotation=rotation, color=color, fontweight=fontweight)
        else:
            ax.text(1.03, cat_pos, text, ha='left', va='center', fontsize=fontsize, rotation=rotation, color=color, fontweight=fontweight)
        return ax

    else:
        # otherwise show the corr value next to the bar end
        text = f"{corr_val:.2f}"
        
        if orientation == "vertical":
            ax.text(cat_pos_with_offset, 1.03, text, ha='center', va='bottom', fontsize=fontsize, rotation=rotation, color=color)
        else:
            ax.text(1.03, cat_pos_with_offset, text, ha='left', va='center', fontsize=fontsize, rotation=rotation, color=color)
    return ax


def _set_value_axis_lim_and_ticks(
    ax, col_index, axes_fontsize, 
    all_levels, orientation, main_plot, 
    bins_on_x=False, pair_bars=False,
    Pearson_next_to_Spearman=False, RE_next_to_delta_RE=False
    ):  
    if main_plot:
        lim_max = 1
        ticks = np.arange(0, 1, 0.2)
    elif bins_on_x:
        lim_max = 0.5
        ticks = np.arange(0, lim_max+0.1, 0.1)
    elif pair_bars:
        lim_max = 1.24
        ticks = np.arange(0, 1.2, 0.2)
    else:
        lim_max = 1.2 if all_levels else 1.22
        ticks = np.arange(0, 1.2, 0.2)
        
    if orientation == "vertical":
        ax.set_ylim([0, lim_max])
        ax.set_yticks(ticks)
    else:
        ax.set_xlim([0, lim_max])
        ax.set_xticks(ticks)
        # set fontsize of x ticks
        ax.tick_params(axis='x', labelsize=axes_fontsize)
    return ax

def _set_value_axis_label(ax, set_label, row_index, col_index, all_levels, all_and_diff, corr_to_plot, pred_col, axes_fontsize, orientation, main_plot, bins_on_x_plot=False):
    if corr_to_plot == ['pearson_corr']:
        corr_str = "$Pearson$ $r$"
    elif corr_to_plot == ['spearman_corr']:
        corr_str = "$Spearman$ $ρ$"
    else:
        corr_str = "$r$"
    if bins_on_x_plot:
        corr_str = "Measure Quality"
    
    
    if orientation == "vertical":
        pred_label = f"{PRED_COLS_FULL_LABELS[pred_col]}\n\n" if all_levels else ""
            
        if col_index == 0:
            ax.set_ylabel(f"{pred_label}{corr_str}", fontsize=axes_fontsize+1, fontweight='bold')
        if bins_on_x_plot and row_index == 2:
            ax.set_xlabel("Proficiency", fontsize=axes_fontsize+1)
    else:
        if all_levels:
            if all_and_diff:
                level_labels = {'all': 'Reading Ease\n\n', 'diff': f'{DELTA}: Reading Ease\n(Original - Simplified)\n'}
                level_by_index = {0: 'all', 1: 'diff'}
            else:
                level_labels = {'Adv': 'Original\n\n\n', 'Ele': 'Simplified\n\n\n', 'diff': f'{DELTA}: Original - Simplified\n\n\n'}
                level_by_index = {0: 'Adv', 1: 'Ele', 2: 'diff'}
            # get level by row_index
            level_label = level_labels[level_by_index[row_index]]
            level_fontsize = axes_fontsize + 9
        elif main_plot:
            resolution_label = 'Sentences\n\n\n\n\n\n' if row_index == 0 else 'Passages\n\n\n\n\n\n'
            resolution_fontsize = axes_fontsize + 9
        else:
            resolution_label = 'Sentences\n\n\n' if row_index == 0 else 'Passages\n\n\n'
            resolution_fontsize = axes_fontsize + 9
    
        if row_index == 1 and col_index == 1:
            ax.set_xlabel(corr_str, fontsize=axes_fontsize+1, fontweight='bold')
        elif col_index == 0:
            if all_levels:
                ax.set_ylabel(level_label, fontsize=level_fontsize, fontweight='bold')
            else:
                ax.set_ylabel(resolution_label, fontsize=resolution_fontsize, fontweight='bold')

    if set_label:
        # keep user override on the proper axis
        ax.set_ylabel(set_label, fontsize=axes_fontsize+1, fontweight='bold')
    return ax


def _set_text_cols_ticks(ax, text_cols_labels, cat_positions, axes_fontsize, SM_prompts_plot, orientation):
    if SM_prompts_plot:
        for prompt_label in [SCORE_SPECIFIC_PROMPT_LABEL, GRADE_SPECIFIC_PROMPT_LABEL, SCORE_PROMPT_LABEL, GRADE_PROMPT_LABEL]:
            text_cols_labels = [label.replace(f" {prompt_label}", "") for label in text_cols_labels]

    if orientation == "vertical":
        ax.set_xticks(cat_positions)
        ax.set_xticklabels(text_cols_labels, rotation=90, fontsize=axes_fontsize)
    else:
        ax.set_yticks(cat_positions)
        ax.set_yticklabels(text_cols_labels, fontsize=axes_fontsize)
        ax.set_ylim(-1, len(text_cols_labels))
    return ax

def _add_value_axis_lines(ax, col_index, zorder, orientation, main_plot, bins_on_x=False):
    if orientation == "vertical":
        for y in np.arange(0, 1, 0.2):
            ax.axhline(y, color='#D6D4D4', linestyle='--', linewidth=0.3, zorder=zorder)
        
        if bins_on_x:
            for y in np.arange(0, 1, 0.1):
                ax.axhline(y, color='#D6D4D4', linestyle='--', linewidth=0.3, zorder=zorder)
    else:
        for x in np.arange(0, 1, 0.2):
            ax.axvline(x, color='#D6D4D4', linestyle='--', linewidth=0.3, zorder=zorder)
        
        if bins_on_x:
            for x in np.arange(0, 1, 0.1):
                ax.axvline(x, color='#D6D4D4', linestyle='--', linewidth=0.3, zorder=zorder)
        
    
    if not main_plot or col_index == 2:
        if orientation == "vertical":
            ax.axhline(1, color='grey', linewidth=0.5)
        else:
            ax.axvline(1, color='grey', linewidth=0.5)
    return ax

def _add_category_separators(ax, col_index, resolution, SM_prompts_plot, subtitle_fontsize, orientation, main_plot, pair_plot, SM_TEXT_plot):
    linestyle = '--'
    linewidth = 0.5
    
    if SM_TEXT_plot:
        return ax
    if SM_prompts_plot:
        _add_lines_prompt_categories(ax, col_index, subtitle_fontsize, linestyle, linewidth, orientation)
    else:
        _add_lines_readability_measures_categories(ax, col_index, resolution, main_plot, subtitle_fontsize, pair_plot, linestyle, linewidth, orientation)
    return ax

def _get_corr_metas(corr_to_plot, corr_cols):
    # Each item = (corr_col,  p_symbol_col, offset, hatch)
    if corr_to_plot == ['pearson_corr']:
        corr_metas = [
            (corr_cols['pearson_col'],   corr_cols['pearson_symbol_col'])
        ]
    elif corr_to_plot == ['spearman_corr']:
        corr_metas = [
            (corr_cols['spearman_col'], corr_cols['spearman_symbol_col'])
        ]
    else:
        corr_metas = [
            (corr_cols['pearson_col'],   corr_cols['pearson_symbol_col']),
            (corr_cols['spearman_col'], corr_cols['spearman_symbol_col'])
        ]
    return corr_metas

def _get_metas_by_L1_L2():
    metas = {
        'L1': {
            'offset': -0.2,
            'hatch_str': ''
        },
        'L2': {
            'offset': 0.2,
            'hatch_str': '///'
        }
    }
    return metas

def _get_corr_cols_by_est_strategy(est_strategy):
    if est_strategy == "Regular":
        pearson_col = 'pearson_corr_all'
        spearman_col = 'spearman_corr_all'
        pearson_symbol_col = 'pearson_p_all_symbol'
        spearman_symbol_col = 'spearman_p_all_symbol'
        pearson_CI_yerr_col = None
        spearman_CI_yerr_col = None
    elif est_strategy == "CV":
        pearson_col = 'pearson_corr_CV'
        spearman_col = 'spearman_corr_CV'
        pearson_symbol_col = 'pearson_p_CV_symbol'
        spearman_symbol_col = 'spearman_p_CV_symbol'
        pearson_CI_yerr_col = 'CI_yerr_pearson_CV'
        spearman_CI_yerr_col = 'CI_yerr_spearman_CV'
    elif est_strategy == "Bootstrap":
        pearson_col = 'pearson_corr_boot'
        spearman_col = 'spearman_corr_boot'
        pearson_symbol_col = 'pearson_p_boot_symbol'
        spearman_symbol_col = 'spearman_p_boot_symbol'
        pearson_CI_yerr_col = 'CI_yerr_pearson_boot'
        spearman_CI_yerr_col = 'CI_yerr_spearman_boot'
    else:
        raise ValueError("est_strategy must be 'Regular' or 'CV' or 'Bootstrap'")
    
    corr_cols = {
        'pearson_col': pearson_col,
        'spearman_col': spearman_col,
        'pearson_symbol_col': pearson_symbol_col,
        'spearman_symbol_col': spearman_symbol_col,
        'pearson_CI_yerr_col': pearson_CI_yerr_col,
        'spearman_CI_yerr_col': spearman_CI_yerr_col
    }
    return corr_cols
    
def _order_by_prompt_category(df, orientation):
    dfs = []
    if orientation == "vertical":
        order = PROMPT_LABEL_ORDER
    else:
        order = PROMPT_LABEL_ORDER_REVERSED
    for prompt_label in order:
        dfs.append(df[df['text_col_label'].str.contains(prompt_label)])

    return pd.concat(dfs).reset_index(drop=True)


def _add_lines_prompt_categories(ax, col_index, subtitle_fontsize, linestyle='-', linewidth=0.5, orientation="vertical"):
    # category positions are along the categorical axis
    
    s12, s23, s34 = 5.5, 11.5, 17.5

    if orientation == "vertical":
        ax.axvline(s12, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axvline(s23, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axvline(s34, color='black', linestyle=linestyle, linewidth=linewidth)
       
        label_locations = [2.5, 8.5, 14.5, 20.5]
        prompt_labels = PROMPT_LABEL_ORDER
        y = 1.22
        for i, prompt_label in enumerate(prompt_labels):
            ax.text(label_locations[i], y, prompt_label, ha='center', va='bottom', fontsize=subtitle_fontsize)
        
    else:
        ax.axhline(s12, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axhline(s23, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axhline(s34, color='black', linestyle=linestyle, linewidth=linewidth)
        
        if col_index == 0:
            label_locations = [0, 7.5, 11.5, 20.5]
            x = LOCATION_PROMPT_CATEGORIES
            prompt_labels = PROMPT_LABEL_ORDER_REVERSED
            for i, prompt_label in enumerate(prompt_labels):
                if 'Score + Crit' in prompt_label:
                    prompt_label = f'{prompt_label}  |'
                elif 'Grade + Crit' in prompt_label:
                    prompt_label = f'|   {prompt_label}   |'
                ax.text(x, label_locations[i], prompt_label, ha='center', va='bottom', fontsize=subtitle_fontsize, rotation=90, fontweight='bold')
            

def _add_lines_readability_measures_categories(ax, col_index, resolution, main_plot, subtitle_fontsize, pair_plot, linestyle='-', linewidth=0.5, orientation="vertical"):
    len_traditional = (len(TRADITIONAL_MEASURES) - 1) if resolution == "sentence" else len(TRADITIONAL_MEASURES)
    len_modern = len(MODERN_MEASURES)
    len_LLM = len(MAIN_PROMPT_COLS)
    len_sys = len(SYSTEMS_MEASURES)
    len_psycho = len(PSYCHOLINGUISTIC_MEASURES) + 1 # +1 for PLL
    
    if orientation == "vertical":
        # order = ['traditional', 'modern', 'LLM', 'sys', 'psycho']
        sep_traditional = len_traditional - 0.5
        sep_modern = sep_traditional + len_modern
        sep_LLM = sep_modern + len_LLM
        sep_sys = sep_LLM + len_sys

        loc_traditional = len_traditional/2
        loc_modern = sep_traditional + len_modern/2
        loc_LLM = sep_modern + len_LLM/2
        loc_sys = sep_LLM + len_sys/2
        loc_psycho = sep_sys + len_psycho/2
    
    else:
        # order = ['psycho', 'sys', 'LLM', 'modern', 'traditional']
        sep_psycho = len_psycho - 0.5
        sep_sys = sep_psycho + len_sys
        sep_LLM = sep_sys + len_LLM
        sep_modern = sep_LLM + len_modern
        sep_traditional = sep_modern + len_traditional
        
        loc_psycho = len_psycho/2
        loc_sys = sep_psycho + len_sys/2
        loc_LLM = sep_sys + len_LLM/2
        loc_modern = sep_LLM + len_modern/2
        loc_traditional = sep_modern + len_traditional/2

    if orientation == "vertical":
        y = 1.22
        ax.text(loc_traditional, y, "Traditional", ha='center', va='bottom', fontsize=subtitle_fontsize)
        ax.axvline(sep_traditional, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.text(loc_modern, y, "Modern", ha='center', va='bottom', fontsize=subtitle_fontsize)
        ax.axvline(sep_modern, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.text(loc_LLM, y, "LLMs", ha='center', va='bottom', fontsize=subtitle_fontsize)
        ax.axvline(sep_LLM, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.text(loc_sys, y, "Sys.", ha='center', va='bottom', fontsize=subtitle_fontsize)
        ax.axvline(sep_sys, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.text(loc_psycho, y, "Psycholinguistic", ha='center', va='bottom', fontsize=subtitle_fontsize)
    else:
        ax.axhline(sep_psycho, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axhline(sep_modern, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axhline(sep_LLM, color='black', linestyle=linestyle, linewidth=linewidth)
        ax.axhline(sep_sys, color='black', linestyle=linestyle, linewidth=linewidth)
        
        if col_index == 0:
            if main_plot:
                x = LOCATION_READABILITY_CATEGORIES_MAIN_PLOT
            else:
                x = LOCATION_READABILITY_CATEGORIES
            loc_traditional = loc_traditional + 0.5
            loc_psycho = loc_psycho - 1
            subtitle_fontsize = subtitle_fontsize + 1
            if pair_plot:
                x = x - 0.1
                subtitle_fontsize = subtitle_fontsize + 2
            ax.text(x, loc_traditional, "\nTraditional\n", ha='left', va='center', fontsize=subtitle_fontsize, rotation=90, fontweight='bold')
            ax.text(x, loc_modern, "\n| Modern |\n", ha='left', va='center', fontsize=subtitle_fontsize, rotation=90, fontweight='bold')
            ax.text(x, loc_LLM, "\nLLMs\n", ha='left', va='center', fontsize=subtitle_fontsize, rotation=90, fontweight='bold')
            ax.text(x, loc_sys, "\n|Sys.|\n", ha='left', va='center', fontsize=subtitle_fontsize, rotation=90, fontweight='bold')
            ax.text(x, loc_psycho, "\nPsycholinguistic\n", ha='left', va='center', fontsize=subtitle_fontsize, rotation=90, fontweight='bold')


