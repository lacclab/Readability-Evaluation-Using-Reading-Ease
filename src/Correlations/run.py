from loguru import logger
from typing import Literal
from pathlib import Path

from src.Correlations.calc_correlations import calc_correlations, agg_folds_correlations, calc_perm_test
from src.Correlations.plots_code.grid_corr_bar_RTx2_RTxSenPar_diff_only import plot_corr_grid_RTx2_RTxSenPar_diff_only
from src.Correlations.plots_code.grid_corr_bar_RTx3_RTxLevel import plot_correlations_all_levels
from src.Correlations.plots_code.grid_corr_by_ppl_RTx2_RTxSenPar_diff_only import plot_corr_by_ppl_grid_RTx2_RTxSenPar_diff_only
from src.Correlations.plots_code.grid_corr_by_ppl_RTx3_RTxLevel import plot_corr_by_perplexity_all_levels
from src.Correlations.plots_code.grid_perm_test_RTx2_RTxSenPar_diff_only import plot_perm_test_RTx2_RTxSenPar_diff_only
from src.Correlations.plots_code.grid_corr_between_all_measures_3x2_LevelxSenPar import plot_all_within_readability_measures_correlation

from src.utils.stat_analysis.Julia_models import setup_julia # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
from src.Correlations.define_cols import (
    MAIN_RT_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS, ALL_SURP_COLS, PROMPT_COLS_FULL_LABELS,
    MAIN_PROMPT_COLS, SM_PROMPT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3
)
# from src.constants import LEXTALE_BINS_NAMES, ADV_COMP_BINS_NAMES

# ----------
# Main funcs
# ----------

def run_corr(
    src_path,
    calc: bool,
    plot: bool,
    calc_for_L1_or_L2: Literal["L1", "L2", "L1_and_L2"],
    calc_for_resolutions:list = ["paragraph", "sentence"], # "article"
    calc_for_reading_regimes:list = ["Gathering0", "Hunting0"],
    fig_list:list = [1,3, 'SM'],
    agg_only: bool = False,
    calc_for_specific_text_cols: list = None
):
    if calc:
        # # ------------------------- Run correlations (For all Figures) -------------------------
        logger.info(f"------- #### {calc_for_L1_or_L2} #### -------")
        setup_julia()
        for reading_regime in calc_for_reading_regimes:
            for resolution in calc_for_resolutions:
                run_for_all_surp = True
                include_bootstrap = True
                if not agg_only:
                    calc_correlations(
                        src_path, resolution, calc_for_L1_or_L2, reading_regime, 
                        pred_type="RT", 
                        run_for_all_surp=run_for_all_surp, 
                        include_bootstrap=include_bootstrap,
                        run_for_specific_text_cols=calc_for_specific_text_cols)
                agg_folds_correlations(
                    src_path, resolution, calc_for_L1_or_L2, reading_regime, 
                    include_bootstrap=include_bootstrap)
    if plot:
        # ------------------------- Plot Fig 1, 3 -------------------------
        setup_julia()
        
        orientation = "horizontal"
        
        if 'main' in fig_list or 1 in fig_list:
            # @ Fig 1 Main
            plot_corr_grid_RTx2_RTxSenPar_diff_only(
                src_path=src_path,
                reader_type="L1_and_L2",
                reading_regime="FirstReading",
                pred_cols=MAIN_RT_COLS,
                text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
                corr_to_plot=["pearson_corr"],
                output_file="RTxSenPar_pearson_corr_FirstReading.pdf",
                est_strategy="Bootstrap",
                main_plot=True,
                orientation=orientation
            )
        
        if 'SM' in fig_list or 'SM_L1_L2' in fig_list:
            # Seperate L1 and L2
            for set_num, SM_RT_cols in zip(["main", 1,2,3], [MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
                plot_corr_grid_RTx2_RTxSenPar_diff_only(
                    src_path=src_path,
                    reader_type="L1_next_to_L2", # next to each other
                    reading_regime="FirstReading",
                    pred_cols=SM_RT_cols,
                text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
                corr_to_plot=["pearson_corr"],
                output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr.pdf",
                est_strategy="Bootstrap",
                orientation=orientation
                )
            
        if 'SM' in fig_list or 'SM_RT' in fig_list:
            # Additional RT
            for set_num, SM_RT_cols in zip([1,2,3], [SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
                plot_corr_grid_RTx2_RTxSenPar_diff_only(
                    src_path=src_path,
                    reader_type="L1_and_L2",
                    reading_regime="FirstReading",
                    pred_cols=SM_RT_cols,
                text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
                corr_to_plot=["pearson_corr"],
                output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr.pdf",
                est_strategy="Bootstrap",
                orientation=orientation
                )
            
        if 'SM' in fig_list or 'SM_hunting' in fig_list:
            # Hunting0
            for set_num, SM_RT_cols in zip(["main", 1,2,3], [MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
                plot_corr_grid_RTx2_RTxSenPar_diff_only(
                    src_path=src_path,
                    reader_type="L1_and_L2",
                    reading_regime="Gathering0_next_to_Hunting0", # next to each other
                    pred_cols=SM_RT_cols,
                text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
                corr_to_plot=["pearson_corr"],
                output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr.pdf",
                est_strategy="Bootstrap",
                orientation=orientation
                )
        
        if 'SM' in fig_list or 'SM_spearman' in fig_list:
            # Spearman
            for set_num, SM_RT_cols in zip(["main", 1,2,3], [MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
                plot_corr_grid_RTx2_RTxSenPar_diff_only(
                    src_path=src_path,
                    reader_type="L1_and_L2",
                    reading_regime="FirstReading",
                    pred_cols=SM_RT_cols,
                text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
                corr_to_plot=["pearson_corr", "spearman_corr"], # next to each other
                output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_next_to_spearman_corr.pdf",
                est_strategy="Bootstrap",
                orientation=orientation
                )
        
        if 'SM' in fig_list or 'SM_prompt' in fig_list:
            # Prompt Variants
            for set_num, SM_RT_cols in zip(["main", 1,2,3], [MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
                plot_corr_grid_RTx2_RTxSenPar_diff_only(
                    src_path=src_path,
                    reader_type="L1_and_L2",
                    reading_regime="FirstReading",
                    pred_cols=MAIN_RT_COLS,
                    text_cols=(MAIN_PROMPT_COLS+SM_PROMPT_COLS),
                    corr_to_plot=["pearson_corr"],
                    output_file=f"SM_prompt_RTxSenPar_pearson_corr_FirstReading_set_{set_num}.pdf",
                    est_strategy="Bootstrap",
                    text_cols_labels=PROMPT_COLS_FULL_LABELS,
                    SM_prompts_plot=True,
                    orientation=orientation
                )
            
        if 'SM' in fig_list or 'SM_all_levels' in fig_list:
            # all levels, seperate for paragraph and sentence
            for resolution in ["paragraph", "sentence"]:
                plot_correlations_all_levels(
                    src_path=src_path,
                    resolution=resolution, 
                    reader_type="L1_and_L2", 
                    reading_regime="FirstReading", 
                    pred_cols=MAIN_RT_COLS, 
                    text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
                    corr_to_plot=["pearson_corr"],
                    output_file=f"SM_all_levels_pearson_corr_{resolution}.pdf",
                    est_strategy="Bootstrap",
                    orientation=orientation
                )
        
        if 'ppl' in fig_list or 3 in fig_list:
            # @ Fig 3 Main
            plot_corr_by_ppl_grid_RTx2_RTxSenPar_diff_only(
                src_path=src_path,
                reader_type="L1_and_L2",
                reading_regime="FirstReading",
                pred_cols=MAIN_RT_COLS,
                surp_cols=ALL_SURP_COLS,
                corr_to_plot=["pearson_corr"],
                output_file="RTxSenPar_pearson_corr_by_perplexity_diff_only.pdf",
                est_strategy="Regular"
            )
        if 'SM' in fig_list or 'SM_ppl' in fig_list:
            # @ Fig 3 SM
            # all levels, seperate for paragraph and sentence
            for resolution in ["paragraph", "sentence"]:
                plot_corr_by_perplexity_all_levels(
                    src_path,
                    resolution,
                    "L1_and_L2",
                    "FirstReading", 
                    pred_cols=MAIN_RT_COLS,
                    surp_cols=ALL_SURP_COLS,
                    corr_to_plot=["pearson_corr"],
                    output_file=f"RTxLevel_pearson_corr_by_perplexity_{resolution}.pdf",
                    est_strategy="Regular"
                )

        if 'within_metrics_corr' in fig_list or 4 in fig_list:
            plot_all_within_readability_measures_correlation(
                src_path=src_path,
                reader_type="L1_and_L2",
                reading_regime="FirstReading",
            )

def run_corr_perm_tests(
    src_path,
    calc: bool,
    plot: bool,
    calc_for_L1_or_L2: Literal["L1", "L2", "L1_and_L2"],
    calc_for_resolutions:list = ["paragraph", "sentence"], # "article"
    calc_for_reading_regimes:list = ["Gathering0", "Hunting0"],
):
    
    # ------------------------- Run perm test for Fig 2 + Plot ------------------------
    
    for resolution in calc_for_resolutions:
        for reading_regime in calc_for_reading_regimes:
            
            if calc:
                calc_perm_test(
                    src_path, resolution, 
                    calc_for_L1_or_L2, reading_regime, 
                    est_strategy="Bootstrap",
                    surp_cols_to_run=MAIN_SURP_COLS)
            
    if plot:
        plot_perm_test_RTx2_RTxSenPar_diff_only(
            src_path=src_path,
            reader_type="L1_and_L2",
            reading_regime="FirstReading",
            est_strategy="Bootstrap",
            pred_type="RT"
        )

if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    calc_for_L1_or_L2 = "L1_and_L2" # "L1" or "L2" or "L1_and_L2"
    
    run_corr(
        src_path,
        calc=False,
        plot=True,
        calc_for_L1_or_L2=calc_for_L1_or_L2,
        calc_for_resolutions=["paragraph", "sentence"],
        calc_for_specific_text_cols=None,
        calc_for_reading_regimes=["FirstReading"],
        fig_list=['main', 'SM']
    )

    run_corr_perm_tests(
        src_path,
        calc=False,
        plot=False,
        calc_for_L1_or_L2=calc_for_L1_or_L2,
        calc_for_resolutions=["paragraph", "sentence"],
        calc_for_reading_regimes=["FirstReading"]
    )
    
    logger.info("------- Finish -------")