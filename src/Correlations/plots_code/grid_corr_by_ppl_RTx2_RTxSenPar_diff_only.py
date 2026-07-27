import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Literal
from tqdm import tqdm
from loguru import logger

from src.utils.stat_analysis.stat_utils import add_p_val_symbols
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlation_by_ppl_plot import _single_corr_by_perplexity_plot, _build_legend_ppl_plot
from src.Correlations.analysis.ppl_trend.models_ppl import _get_models_data

def plot_corr_by_ppl_grid_RTx2_RTxSenPar_diff_only(
    src_path: str,
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    pred_cols: List[str], 
    surp_cols: List[str], 
    corr_to_plot: List[str], 
    output_file: str,
    est_strategy: Literal["Regular", "CV", "Bootstrap"],
    fontsize_title=20,
    fontsize_legend_text=16,
    markzise_legend=12,
):
    pass

    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {reading_regime} | {reader_type} | {pred_cols}")
    sentences_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_sentence.csv")
    paragraphs_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_paragraph.csv")

    # filter text_cols
    sentences_corr_df = sentences_corr_df[sentences_corr_df['text_col'].isin(surp_cols)]
    paragraphs_corr_df = paragraphs_corr_df[paragraphs_corr_df['text_col'].isin(surp_cols)]

    # get models data
    surp_to_ppl, surp_to_family, surp_to_model_name_with_size = _get_models_data(src_path)
    
    resolution_types = ['sentence', 'paragraph']
    level_type = 'diff'
    
    n_rows = len(pred_cols)
    n_cols = len(resolution_types)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*6.5, n_rows*5), sharey=True)

    # Set y-label on the left column, set column titles on top row
    for j, y_type in enumerate(resolution_types):
        y_labels = {'sentence': 'Sentences\n', 'paragraph': 'Passages\n'}
        axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')
    
    comp_res = []
    # Loop over pred_cols, level_types
    for i, pred_col in tqdm(enumerate(pred_cols)):
        for j, resolution in enumerate(resolution_types):
            ax = axs[i, j]
            corr_df = sentences_corr_df if resolution == 'sentence' else paragraphs_corr_df
            sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'] == level_type)].reset_index(drop=True)
            if sub_corr_df.empty:
                logger.warning(f"Empty sub_corr_df for {pred_col} at {resolution} level. Skipping...")
                continue
            
            comp_dict = _single_corr_by_perplexity_plot(
                ax, j, sub_corr_df, corr_to_plot, pred_col, 
                surp_to_model_name_with_size, surp_to_family, surp_to_ppl, 
                all_levels=True, est_strategy=est_strategy)
            # add pred_col, level_type to comp_dict
            comp_dict['pred_col'] = pred_col
            comp_dict['level_type'] = level_type
            comp_dict['resolution'] = resolution
            comp_res.append(comp_dict)
            
    comp_res_df = pd.DataFrame(comp_res)
    # add p symbols to comp_res_df
    comp_res_df = add_p_val_symbols(comp_res_df, 'comp_p')
    comp_res_df = add_p_val_symbols(comp_res_df, 'log_comp_p')
    comp_res_df = add_p_val_symbols(comp_res_df, 'ppl_coef_p')
    # save
    comp_res_df.to_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/ppl_comp_res_diff_only.csv", index=False)

    fig = _build_legend_ppl_plot(fig, surp_cols, corr_df, surp_to_family,
                                 fontsize_legend_text, markzise_legend)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
        
    _save_file_to_all_paths(
        resolution=resolution, 
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=pred_cols, 
        text_cols=surp_cols, 
        corr_to_plot=corr_to_plot, src_path=src_path, est_strategy=est_strategy
        )

if __name__ == "__main__":
    from pathlib import Path
    from src.Correlations.define_cols import (
        MAIN_RT_COLS, ALL_SURP_COLS
    )
    from src.utils.stat_analysis.Julia_models import setup_julia # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
    setup_julia()
    
    src_path = Path.cwd() / "src"
    L1_or_L2 = "L2" # "L1" or "L2"
    
    # @ Fig 3 Main
    plot_corr_by_ppl_grid_RTx2_RTxSenPar_diff_only(
        src_path=src_path,
        reader_type=L1_or_L2,
        reading_regime="Gathering0",
        pred_cols=MAIN_RT_COLS,
        surp_cols=ALL_SURP_COLS,
        corr_to_plot=["pearson_corr"],
        output_file="RTxSenPar_pearson_corr_by_perplexity_diff_only.pdf",
        est_strategy="Bootstrap" # "Bootstrap" | "Regular"
    )

