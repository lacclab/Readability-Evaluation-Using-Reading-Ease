import numpy as np
from typing import Literal
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from src.utils.stat_analysis.Julia_models import fit_linear_model # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
from src.constants import PRED_COLS_FULL_LABELS
from src.utils.plot_utils import (
    SIGNIFICANCE_SHAPES, MARKER_COLORS_BY_FAMILY
)

def _single_corr_by_perplexity_plot(
    ax, col_index, 
    sub_corr_df, corr_to_plot, pred_col, 
    surp_to_model_name_with_size, surp_to_family, surp_to_ppl, 
    all_levels, 
    est_strategy: Literal["Regular", "CV", "Bootstrap"] = 'Regular',
    marksize = 8,
    fontsize_model = 14,
    fontsize_axis = 16
    ):
    if est_strategy == "Regular":
        pearson_col = 'pearson_corr_all'
        spearman_col = 'spearman_corr_all'
        pearson_symbol_col = 'pearson_p_all_symbol'
        spearman_symbol_col = 'spearman_p_all_symbol'
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
    
    # Define correlation “meta” so we can loop instead of duplicating code
    # Each item = (corr_col,  p_symbol_col, offset, hatch)
    if corr_to_plot == ['pearson_corr']:
        corr_metas = [
            (pearson_col,   pearson_symbol_col,   0, "")
        ]
    elif corr_to_plot == ['spearman_corr']:
        corr_metas = [
            (spearman_col, spearman_symbol_col,  0, "")
        ]
    else:
        corr_metas = [
            (pearson_col,   pearson_symbol_col,  -0.2, ""),
            (spearman_col, spearman_symbol_col,  +0.2, "///")
        ]
    
    # drop surp cols with Max
    sub_corr_df = sub_corr_df[~sub_corr_df['text_col'].str.contains('Max')].reset_index(drop=True)
    
    sub_corr_df['clean_surp_col'] = sub_corr_df['text_col'].apply(lambda x: x.split(' Mean')[0])
    # add perplexity to df
    sub_corr_df['perplexity'] = sub_corr_df['clean_surp_col'].apply(lambda x: surp_to_ppl[x])
    # log perplexity
    sub_corr_df['log_perplexity'] = np.log10(sub_corr_df['perplexity'])
    # family
    sub_corr_df['family'] = sub_corr_df['clean_surp_col'].apply(lambda x: surp_to_family[x])
    # model_name_with_size
    sub_corr_df['model_name_with_size'] = sub_corr_df['clean_surp_col'].apply(lambda x: surp_to_model_name_with_size[x])

    # Loop over each row => one “text_col”
    for i, row in sub_corr_df.iterrows():
        # For each correlation type => 2 bars
        for corr_col, symbol_col, offset, hatch_str in corr_metas:
            # If the columns exist, proceed
            if corr_col not in row or symbol_col not in row:
                continue

            corr_val        = abs(row[corr_col])
            signif          = row[symbol_col]
            perplexity_val  = row['log_perplexity']
            family          = row['family']
            model_name_with_size = row['model_name_with_size']
            color_          = MARKER_COLORS_BY_FAMILY[family]
            marker_shape    = SIGNIFICANCE_SHAPES[signif]
            dot_x           = perplexity_val + offset
            
            if est_strategy == "Regular":
                # Draw the bar without CI
                ax.plot(
                    dot_x,
                    corr_val,
                    marker=marker_shape,
                    markersize=marksize,
                    color=color_,
                    markeredgecolor='black',
                    linestyle='None'  # ensures no connecting line
                )
            else:
                CI_yerr_col = pearson_CI_yerr_col if corr_col == "pearson_corr" else spearman_CI_yerr_col
                CI_yerr = row[CI_yerr_col]

                # Draw the bar with CI
                ax.errorbar(
                    dot_x,
                    corr_val,
                    yerr=CI_yerr,
                    fmt=marker_shape,          # 'o' marker, no line
                    color=color_,
                    markeredgecolor='black',
                    capsize=2,
                    ecolor='black'    # color for error bar lines
                )
            # place numeric text above
            n_corrs = len(corr_metas)
            text_x = dot_x
            if n_corrs > 1 and corr_col == "spearman_corr":
                text_x = text_x + 0.05
            else:
                text_x = text_x
            # if surp col Pythia 70M
            if row['clean_surp_col'] == 'Pythia 70M':
                ax.text(
                    text_x,
                    corr_val + 0.05, # corr_val + CI_yerr + 0.02,
                    model_name_with_size,
                    ha='center', va='bottom', fontsize=fontsize_model,
                    rotation=90, #if n_corrs == 2 else 0,
                    color='grey'
                )
    
    if col_index == 0:
        if all_levels:
            pred_col_str = f"{PRED_COLS_FULL_LABELS[pred_col]}\n\n"
        else:
            pred_col_str = ""

        # Decide correlation label
        if corr_to_plot == ['pearson_corr']:
            corr_str = "$Pearson$ $r$"
        elif corr_to_plot == ['spearman_corr']:
            corr_str = "$Spearman$ $ρ$"
        else:
            corr_str = "$r$"

        y_axis_str = f"{pred_col_str}{corr_str}"
        ax.set_ylabel(y_axis_str, fontsize=fontsize_axis, fontweight='bold')    
    
    # 4) make sure y = 0, y = 1 in the plot
    ax.set_ylim([0, 1])    
    # add grey line at y=1
    ax.axhline(y=0.8, color='grey', linestyle='--', linewidth=0.4)
    ax.axhline(y=0.6, color='grey', linestyle='--', linewidth=0.4)
    ax.axhline(y=0.4, color='grey', linestyle='--', linewidth=0.4)
    ax.axhline(y=0.2, color='grey', linestyle='--', linewidth=0.4)
    
    # x axis label perplexity
    ax.set_xlabel("$Average$ $Surprisal$", fontsize=fontsize_axis, fontweight='bold')
    
    if all_levels:
        # check for correlation between perplexity and results
        comp_r, comp_p = pearsonr(sub_corr_df[pearson_col], sub_corr_df['perplexity'])
        log_comp_r, log_comp_p = pearsonr(sub_corr_df[pearson_col], sub_corr_df['log_perplexity'])
        mean_result = sub_corr_df[pearson_col].mean()
        
        formula = f"{pearson_col} ~ 1 + perplexity"
        coef_table = fit_linear_model(sub_corr_df, pearson_col, formula, silent=True, needed_cols=[pearson_col, 'perplexity'])
        coef_ppl_res = coef_table[coef_table['Name']=="perplexity"][['Coef.', 'Pr(>|t|)']].values.flatten()
        ppl_coef, ppl_coef_p = coef_ppl_res[0], coef_ppl_res[1]
        # return comp results in dict
        return {
            'comp_r': comp_r,
            'comp_p': comp_p,
            'log_comp_r': log_comp_r,
            'log_comp_p': log_comp_p,
            'ppl_coef': ppl_coef,
            'ppl_coef_p': ppl_coef_p,
            'mean_result': mean_result,
            'n_values': len(sub_corr_df),
            'models_families': sub_corr_df['family'].unique()
        }
    else:
        return None
   
def _build_legend_ppl_plot(
    fig, surp_cols, corr_df, 
    surp_to_family,
    fontsize_legend_text=9,
    markzise_legend=8
    ):
    # Build a single legend for significance colors
    handles = []
    for shape in ['o', '^']:
        if shape == 'o':
            label = "p < 0.05"
        elif shape == '^':
            label = "p >= 0.05"
        else:
            raise ValueError(f"Unknown shape: {shape}")
        # Create an invisible line with the given marker
        line = mlines.Line2D(
            [], [], 
            color='black',            # or any color to represent the marker edge
            marker=shape,
            label=label,
            markeredgecolor='black',
            markersize=markzise_legend,
            linestyle='None'
        )
        handles.append(line)
        
    # add to legend Familiy shapes
    # relevant families
    exisiting_surp_col = [col.split(' Mean')[0] for col in surp_cols if col in corr_df['text_col'].unique() and 'Max' not in col]
    exisiting_families = [surp_to_family[surp_col] for surp_col in exisiting_surp_col]
    for family, color in MARKER_COLORS_BY_FAMILY.items():
        if family not in exisiting_families:
            continue
        label = f"{family}"
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)


    fig.legend(
        handles=handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.0), 
        ncol=4, 
        fontsize=fontsize_legend_text, 
        frameon=False
    )
    return fig

def _get_models_data(src_path):
    perplexity_df = pd.read_csv(src_path / "Linguistic_Metrics/perplexity/models_names_with_ppl.csv")
    # rename col sentence_level_ppl to ppl
    perplexity_df = perplexity_df.rename(columns={'sentence_level_ppl': 'ppl'})
    # dict surp_col to perplexity
    surp_to_ppl = perplexity_df.set_index('surp_col')['ppl'].to_dict()
    # surp to family
    surp_to_family = perplexity_df.set_index('surp_col')['model_family'].to_dict()
    # surp to model_name_with_size
    surp_to_model_name_with_size = perplexity_df.set_index('surp_col')['model_name_with_size'].to_dict()
    return surp_to_ppl, surp_to_family, surp_to_model_name_with_size
  