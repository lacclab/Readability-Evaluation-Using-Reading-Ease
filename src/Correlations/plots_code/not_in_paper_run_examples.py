from pathlib import Path

from src.utils.stat_analysis.Julia_models import setup_julia # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
from src.Correlations.define_cols import (
    MAIN_RT_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS, PROMPT_COLS_FULL_LABELS,
    MAIN_PROMPT_COLS, SM_PROMPT_COLS, SM_TEXT_COLS, SM_RT_COLS, SM_SURP_COLS, READING_COMPREHENSION_COLS
)
from src.Correlations.calc_correlations import calc_correlations, agg_folds_correlations
from src.Correlations.plots_code.grid_corr_bar_RTx2_RTcorrXPermTest import plot_corr_4x2_grid
from src.Correlations.plots_code.grid_corr_bar_2x2_RTxRT_diff_only import plot_correlations_diff_levels
from src.Correlations.plots_code.grid_corr_bar_RTx3_RTxLevel import plot_correlations_all_levels
from src.Correlations.plots_code.grid_perm_test_2x2_RTxRT_diff_only import plot_perm_test_results_diff_levels
from src.Correlations.plots_code.grid_perm_test_RTx3_RTxLevel import plot_perm_test_results_all_levels

src_path = Path.cwd() / "src"
L1_or_L2 = "L2" # "L1" or "L2"  

setup_julia()
for resolution in ["paragraph", "sentence"]: # article
    for reading_regime in ["Gathering0", "Hunting0"]:
        # @ Not in paper - Fig 1 combined
        plot_corr_4x2_grid(
            src_path, resolution, 
            L1_or_L2, reading_regime, 
            est_strategy="Bootstrap", 
            pred_type="RT",
            corr_to_plot=["pearson_corr"], text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS))

    # @ Fig 1 Main
    plot_correlations_diff_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Gathering0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"main_pearson_corr_{resolution}_diff_only.pdf",
        est_strategy="Bootstrap"
    )             
    
    # @ Fig 1 SM
    plot_correlations_all_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Gathering0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"main_pearson_corr_{resolution}.pdf",
        est_strategy="Bootstrap"
    )
    # @ Fig 1 SM - more RT cols
    plot_correlations_all_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Gathering0", 
        pred_cols=SM_RT_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"SM_RT_corr_{resolution}.pdf",
        est_strategy="Bootstrap"
    )
    # @ Fig 1 SM - spearman
    plot_correlations_all_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Gathering0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["spearman_corr"],
        output_file=f"main_spearman_corr_{resolution}.pdf",
        est_strategy="Bootstrap"
    )
    # @ Fig 1 SM - Hunting0
    plot_correlations_all_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Hunting0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"main_pearson_corr_{resolution}.pdf",
        est_strategy="Bootstrap"
    ) 
    # @ Fig 1 Main - Hunting0 diff only
    plot_correlations_diff_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Hunting0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"main_pearson_corr_{resolution}_diff_only.pdf",
        est_strategy="Bootstrap"
    )
    # @ Fig 1 SM - more text cols
    plot_correlations_all_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Gathering0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(SM_TEXT_COLS+SM_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"SM_text_corr_{resolution}.pdf",
        est_strategy="Bootstrap",
    )
    # @ Fig 1 SM - more prompt cols
    plot_correlations_all_levels(
        src_path,
        resolution, 
        L1_or_L2, 
        "Gathering0", 
        pred_cols=MAIN_RT_COLS, 
        text_cols=(MAIN_PROMPT_COLS+SM_PROMPT_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file=f"SM_prompt_corr_{resolution}.pdf",
        est_strategy="Bootstrap",
        plot_lines_measures_categories=False,
        text_cols_labels=PROMPT_COLS_FULL_LABELS,
        SM_prompts_plot=True
    )
                
    # @ Fig 1.2 SM
    plot_perm_test_results_diff_levels(
        src_path, resolution, 
        L1_or_L2, reading_regime, 
        est_strategy="Bootstrap", 
        pred_type="RT")
    

def run_corr_for_comprehension_cols(src_path, L1_or_L2, calc: bool, plot: bool):
    # ------------------------- Not in paper! --------------------------
    # -------------------------- Run correlations + perm test for Reading Comprehension cols ------------------------
    
    for resolution in ["article", "paragraph"]:
        for reading_regime in ["Gathering0", "Hunting0"]:
            for est_strategy in ["Regular", "CV", "Bootstrap"]:
                run_for_all_surp = False
                include_bootstrap = True
                if calc:
                    calc_correlations(src_path, resolution, L1_or_L2, reading_regime, pred_type="comprehension", run_for_all_surp=run_for_all_surp, include_bootstrap=include_bootstrap)
                    agg_folds_correlations(src_path, resolution, L1_or_L2, reading_regime, include_bootstrap=include_bootstrap)
                if plot:
                    plot_correlations_all_levels(
                        src_path,
                        resolution, 
                        L1_or_L2, 
                        reading_regime, 
                        pred_cols=READING_COMPREHENSION_COLS, 
                        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
                        corr_to_plot=["pearson_corr"],
                        output_file=f"comprehension_corr_{resolution}.pdf",
                        est_strategy=est_strategy
                    )
            
            # @ Fig Comprehension
            if plot:
                if resolution != "sentence":
                    plot_perm_test_results_all_levels(
                        src_path, resolution, 
                        L1_or_L2, reading_regime, 
                        est_strategy, pred_type="comprehension")