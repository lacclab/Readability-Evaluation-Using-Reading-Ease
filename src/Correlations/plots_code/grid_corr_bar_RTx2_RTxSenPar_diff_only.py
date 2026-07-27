import matplotlib.pyplot as plt
from typing import List, Literal
from loguru import logger
import pandas as pd

from src.constants import PRED_COLS_FULL_LABELS, LEXTALE_BINS_NAMES, ADV_COMP_BINS_NAMES
from src.utils.plot_utils import add_significance_legend, SIGNIFICANCE_SIGN_DIFF_COLORS, SIGNIFICANCE_SIGN_SLOPE_LABELS
from src.Correlations.define_cols import READING_COMPREHENSION_COLS, TEXT_COLS_FULL_LABELS
from src.Correlations.utils import _save_file_to_all_paths
from src.Correlations.plots_code.single_correlations_bar_plot import (
    _single_corr_plot, _single_corr_plot_of_bins, _single_corr_plot_bins_on_x, _pair_bars_or_not
)
from src.Correlations.plots_code.corr_plot_utils import ( 
    _add_legend_for_hatch, _add_signficance_legend, _load_corr_df, HATCH_STR_DICT_LABELS, _add_bin_names_legend, _load_corr_boot_df
    )
from src.utils.stat_analysis.Julia_models import setup_julia # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
from src.utils.files_utils import replace_results_in_file

def plot_corr_grid_RTx2_RTxSenPar(
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
    debug_mode: bool = False,
    diff_only: bool = True,
    SM_TEXT_plot: bool = False,
    DONT_CALC_PERM_TEST=True,
    pred_type="RT",
    hatch_labels: dict = None,
    level_labels_override: dict = None,
    title_override: str = None,
    compact: bool = False,
):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting {output_file} | {reading_regime} | {reader_type} | {pred_cols}")
    sentences_corr_df = _load_corr_df(reader_type, reading_regime, resolution='sentence', src_path=src_path, text_cols=text_cols)
    paragraphs_corr_df = _load_corr_df(reader_type, reading_regime, resolution='paragraph', src_path=src_path, text_cols=text_cols)

    L1_next_to_L2=True if reader_type == "L1_next_to_L2" else False
    Gathering0_next_to_Hunting0=True if reading_regime == "Gathering0_next_to_Hunting0" else False
    FirstReading_next_to_RepeatedReading=True if reading_regime == "FirstReading_next_to_RepeatedReading" else False
    FirstReading_next_to_Gathering0=True if reading_regime == "FirstReading_next_to_Gathering0" else False
    Pearson_next_to_Spearman=True if len(corr_to_plot) > 1 else False
    RE_next_to_delta_RE=True if not diff_only else False
    # Set default hatch labels based on pred_type if not provided
    if hatch_labels is None:
        if pred_type == "comprehension":
            hatch_labels = {'': 'Not Controlled', '///': 'Controlled'}
        else:
            hatch_labels = HATCH_STR_DICT_LABELS['RE_next_to_delta_RE']
    # Set default level labels override for comprehension
    if level_labels_override is None and pred_type == "comprehension":
        level_labels_override = {'all': 'Not Controlled\n\n', 'diff': 'Controlled\n\n'}
    bins_plot=True if "Lextale" in reader_type or "Adv_comp" in reader_type else False
    bins_plot=False if bins_on_x else bins_plot  # if bins_on_x, then bins_plot=False to avoid legend clash
    linear_fit_results_path = src_path / f"Correlations/{reader_type}/{reading_regime}/linear_fit_by_bin_results.csv"
    linear_fits_results = pd.read_csv(linear_fit_results_path) if linear_fit_results_path.exists() else None
    
    perm_test_res_results_path = src_path / f"Correlations/{reader_type}/{reading_regime}/pairs_perm_test_results.csv"
    if perm_test_res_results_path.exists():
        logger.warning("Loading existing permutation test results for pair plot")
        perm_test_res_results = pd.read_csv(perm_test_res_results_path)
    else:
        logger.info("pair plot - didnt find any existing permutation test results")
        perm_test_res_results = None
    
    pair_bars = _pair_bars_or_not(  
        L1_next_to_L2,  
        Pearson_next_to_Spearman,  
        Gathering0_next_to_Hunting0, 
        FirstReading_next_to_RepeatedReading,
        FirstReading_next_to_Gathering0,
        RE_next_to_delta_RE
        )

    if pred_type == "comprehension":
        resolution_types = ['paragraph']
    else:
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
            
        if pair_bars:
            sharey = False
        else:
            sharey = True

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*row_length, n_rows*5.5), sharey=sharey, squeeze=False)

        # Set y-label on the left column, set column titles on top row
        for j, y_type in enumerate(resolution_types):
            if title_override:
                title = f'{title_override}\n\n'
            else:
                title = {'sentence': 'Sentences\n\n', 'paragraph': 'Passages\n\n'}[y_type]
            axs[0, j].set_title(title, fontsize=fontsize_title, fontweight='bold')
    else:
        # create fig with subplots
        n_rows = len(resolution_types)
        n_cols = len(pred_cols)
        if main_plot:
            col_length = 7
        elif SM_prompts_plot:
            col_length = 9
        elif pair_bars:
            col_length = 10
        elif compact:
            col_length = 5
        else:
            col_length = 8

        if pair_bars:
            sharey = False
            row_length = 7
        else:
            sharey = 'row'
            row_length = 4 if compact else 5.5

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*row_length, n_rows*col_length), sharex=True, sharey=sharey, squeeze=False)

        # Set y-label on the left column, set row titles on top row
        for j, y_type in enumerate(pred_cols):
            y_labels = {pred_col: f'{PRED_COLS_FULL_LABELS[pred_col]}\n' for pred_col in pred_cols}
            axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')

    new_perm_test_res = []
    new_linear_fits_results = []
    # Loop over pred_cols, level_types
    for i, pred_col in enumerate(pred_cols):
        for j, resolution in enumerate(resolution_types):
            ax = axs[i, j] if orientation == "vertical" else axs[j, i]
            row_index = i if orientation == "vertical" else j
            col_index = j if orientation == "vertical" else i
            corr_df = sentences_corr_df if resolution == 'sentence' else paragraphs_corr_df
            if diff_only:
                sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'] == 'diff')].reset_index(drop=True)
            else:
                sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col)].reset_index(drop=True)
            if sub_corr_df.empty:
                logger.warning(f"Empty sub_corr_df for {pred_col} at {resolution} level. Skipping...")
                continue
            
            if bins_on_x:
                corr_boot_dfs = []
                for bin_name in (LEXTALE_BINS_NAMES if "Lextale" in reader_type else ADV_COMP_BINS_NAMES):
                    reader_type_bin = f"Lextale_{bin_name}" if "Lextale" in reader_type else f"Adv_comp_{bin_name}"
                    corr_boot_df = pd.read_csv(src_path / f"Correlations/{reader_type_bin}/{reading_regime}/correlations_{resolution}_{pred_col}.csv")
                    corr_boot_df = corr_boot_df[(corr_boot_df['pred_col'] == pred_col) & (corr_boot_df['level_type'] == level_type)].reset_index(drop=True)
                    corr_boot_df['bin_name'] = bin_name
                    corr_boot_df = corr_boot_df[corr_boot_df['fold'] == "bootstrap_all"]
                    corr_boot_dfs.append(corr_boot_df)
                sub_corr_boot_df = pd.concat(corr_boot_dfs, ignore_index=True)
                
                new_linear_fits = _single_corr_plot_bins_on_x(
                    resolution, ax, row_index, col_index,
                    sub_corr_df, corr_to_plot, pred_col, text_cols, 
                    all_levels=True,  est_strategy=est_strategy, text_cols_labels=text_cols_labels, 
                    orientation=orientation,
                    linear_fits_results=linear_fits_results,
                    sub_corr_boot_df=sub_corr_boot_df,
                    debug_mode=debug_mode
                )
                new_linear_fits_results.extend(new_linear_fits)
            elif bins_plot:
                _single_corr_plot_of_bins(
                    resolution, ax, row_index, col_index,
                    sub_corr_df, corr_to_plot, pred_col, text_cols, 
                    all_levels=True, est_strategy=est_strategy, text_cols_labels=text_cols_labels, 
                    SM_prompts_plot=SM_prompts_plot,
                    orientation=orientation
                )
            else:
                sub_corr_boot_df = _load_corr_boot_df(reader_type, reading_regime, resolution, src_path, pred_col)
                if perm_test_res_results is not None:
                    sub_perm_results = perm_test_res_results[perm_test_res_results['resolution'] == resolution]
                    sub_perm_results = sub_perm_results[sub_perm_results['pred_col'] == pred_col]
                    if sub_perm_results.empty:
                        need_to_calc_perm_test = True
                    else:
                        need_to_calc_perm_test = False
                else:
                    sub_perm_results = None
                    need_to_calc_perm_test = True
                if DONT_CALC_PERM_TEST:
                    need_to_calc_perm_test = False
                    
                if L1_next_to_L2:
                    path_steiger_res = src_path / f"Correlations/L1_next_to_L2/FirstReading/steiger_test_between_RT_cols_{resolution}_L1_next_to_L2.csv"
                    steiger_res = pd.read_csv(path_steiger_res)
                    steiger_res = steiger_res[(steiger_res['level_type'] == level_type) & (steiger_res['pred_col'] == pred_col)]
                elif Gathering0_next_to_Hunting0:
                    path_steiger_res = src_path / f"Correlations/L1_and_L2/Gathering0_next_to_Hunting0/steiger_test_between_RT_cols_{resolution}_Gathering0_next_to_Hunting0.csv"
                    steiger_res = pd.read_csv(path_steiger_res)
                    steiger_res = steiger_res[(steiger_res['level_type'] == level_type) & (steiger_res['pred_col'] == pred_col)]
                else:
                    steiger_res = None

                perm_test_res = _single_corr_plot(
                    resolution, ax, row_index, col_index, 
                    sub_corr_df, sub_corr_boot_df, corr_to_plot, pred_col, text_cols, 
                    all_levels=False, est_strategy=est_strategy, text_cols_labels=text_cols_labels, 
                    SM_prompts_plot=SM_prompts_plot, main_plot=main_plot,
                    L1_next_to_L2=L1_next_to_L2,
                    Gathering0_next_to_Hunting0=Gathering0_next_to_Hunting0,
                    FirstReading_next_to_RepeatedReading=FirstReading_next_to_RepeatedReading,
                    FirstReading_next_to_Gathering0=FirstReading_next_to_Gathering0,
                    Pearson_next_to_Spearman=Pearson_next_to_Spearman,
                    RE_next_to_delta_RE=RE_next_to_delta_RE,
                    orientation=orientation,
                    pair_bars=pair_bars,
                    perm_test_res=sub_perm_results,
                    steiger_res=steiger_res,
                    need_to_calc_perm_test=need_to_calc_perm_test,
                    SM_TEXT_plot=SM_TEXT_plot,
                    level_labels_override=level_labels_override
                )
                if perm_test_res is not None:
                    perm_test_res['reader_type'] = reader_type
                    perm_test_res['reading_regime'] = reading_regime
                    perm_test_res['resolution'] = resolution
                    perm_test_res['pred_col'] = pred_col
                    new_perm_test_res.append(perm_test_res)

    if len(new_perm_test_res) > 0:
        new_perm_test_res_df = pd.concat(new_perm_test_res)
        replace_results_in_file(perm_test_res_results_path, new_perm_test_res_df)
    
    if bins_on_x and len(new_linear_fits_results) > 0:
        new_linear_fits_results_df = pd.DataFrame(new_linear_fits_results)
        replace_results_in_file(linear_fit_results_path, new_linear_fits_results_df)


    if bins_plot:
        fig = _add_bin_names_legend(fig, legend_text_fontsize, reader_type)
    elif bins_on_x:
        # plt.tight_layout(rect=[0, 0.08, 1, 1])
        # plt.legend(frameon=False, ncol=7, loc='lower left', bbox_to_anchor=(-0.9, -0.35))
        fig = add_significance_legend(
            fig, 
            significance_colors=SIGNIFICANCE_SIGN_DIFF_COLORS,
            significance_labels=SIGNIFICANCE_SIGN_SLOPE_LABELS,
            fontsize_legend_text=legend_text_fontsize
            )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        if pair_bars:
            fig = _add_signficance_legend(fig, legend_text_fontsize=14)
        else:
            fig = _add_signficance_legend(fig, legend_text_fontsize)

    if L1_next_to_L2:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['L1_next_to_L2'], legend_text_fontsize=14)
    if Gathering0_next_to_Hunting0:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['Gathering0_next_to_Hunting0'], legend_text_fontsize=14)
    if FirstReading_next_to_RepeatedReading:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['FirstReading_next_to_RepeatedReading'], legend_text_fontsize=14)
    if FirstReading_next_to_Gathering0:
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['FirstReading_next_to_Gathering0'], legend_text_fontsize=14)
    if RE_next_to_delta_RE:
        fig = _add_legend_for_hatch(fig, hatch_labels, legend_text_fontsize=14)
    if len(corr_to_plot) > 1 and 'pearson' in corr_to_plot[0].lower() and 'spearman' in corr_to_plot[1].lower():
        fig = _add_legend_for_hatch(fig, HATCH_STR_DICT_LABELS['pearson_spearman'], legend_text_fontsize=14)

    if n_rows == 1 and len(fig.legends) > 1:
        # Stack legends vertically to avoid overlap when there's only one row
        for i, legend in enumerate(fig.legends):
            legend.set_bbox_to_anchor((0.5, i * 0.05), transform=fig.transFigure)
            legend._loc = 8  # 'lower center'
        bottom_margin = len(fig.legends) * 0.05 + 0.05
        plt.tight_layout(rect=[0, bottom_margin, 1, 1])
    elif not bins_plot and not bins_on_x:
        bottom_margin = 0.06 if (orientation == "vertical" and n_rows == 1) else 0.03
        plt.tight_layout(rect=[0, bottom_margin, 1, 1])
    elif bins_plot:
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
    _save_file_to_all_paths(resolution, reader_type, reading_regime, output_file, pred_cols, text_cols, corr_to_plot, src_path, est_strategy)

if __name__ == "__main__":
    from pathlib import Path
    from src.Correlations.define_cols import (
        MAIN_RT_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS,
        SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3,
    )
    # setup_julia()
    src_path = Path.cwd() / "src"
    L1_or_L2 = "L1_and_L2" # "L1" or "L2"  or "L1_and_L2" or "general_reader"
    
    plot_corr_grid_RTx2_RTxSenPar(
        src_path=src_path,
        reader_type="L1_and_L2", 
        reading_regime="FirstReading", 
        pred_cols=READING_COMPREHENSION_COLS, 
        text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS), 
        corr_to_plot=["pearson_corr"],
        output_file="SM_comprehension_all_levels_pearson_corr_RE_next_to_delta_RE.pdf",
        est_strategy="Bootstrap",
        orientation="horizontal",
        diff_only=False,
        pred_type="comprehension"
    )
    
    # # Seperate L1 and L2
    # for set_num, SM_RT_cols in zip(["main", 1,2,3], [MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
    #     plot_corr_grid_RTx2_RTxSenPar(
    #         src_path=src_path,
    #         reader_type="L1_next_to_L2", # next to each other
    #         reading_regime="FirstReading",
    #         pred_cols=SM_RT_cols,
    #     text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #     corr_to_plot=["pearson_corr"],
    #     output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr.pdf",
    #     est_strategy="Bootstrap",
    #     orientation="horizontal"
    #     )
    
    # plot_corr_grid_RTx2_RTxSenPar_diff_only(
    #     src_path=src_path,
    #     reader_type="L1_and_L2",
    #     reading_regime="FirstReading",
    #     pred_cols=MAIN_RT_COLS,
    #     text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #     corr_to_plot=["pearson_corr"],
    #     output_file="RTxSenPar_pearson_corr_FirstReading_horizontal.pdf",
    #     est_strategy="Bootstrap",
    #     main_plot=True,
    #     orientation="horizontal"
    # )
    
    # for set_num, SM_RT_cols in zip([1,2,3, "main"], [SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3, MAIN_RT_COLS]):
    #     plot_corr_grid_RTx2_RTxSenPar_diff_only(
    #     src_path=src_path,
    #     reader_type="L1_and_L2",
    #     reading_regime="FirstReading",
    #     pred_cols=SM_RT_cols,
    #     text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #     corr_to_plot=["pearson_corr"],
    #     output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr_horizontal.pdf",
    #     est_strategy="Bootstrap",
    #     orientation="horizontal"
    #     )
    #     plot_corr_grid_RTx2_RTxSenPar_diff_only(
    #     src_path=src_path,
    #     reader_type="L1_and_L2",
    #     reading_regime="FirstReading",
    #     pred_cols=SM_RT_cols,
    #     text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #     corr_to_plot=["pearson_corr"],
    #     output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr_vertical.pdf",
    #     est_strategy="Bootstrap",
    #     orientation="vertical"
    #     )

        
    # for set_num, SM_RT_cols in zip([1,2,3, "main"], [SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3, MAIN_RT_COLS]):
    #     plot_corr_grid_RTx2_RTxSenPar_diff_only(
    #         src_path=src_path,
    #         reader_type="L1_and_L2",
    #         reading_regime="FirstReading_next_to_RepeatedReading",
    #         pred_cols=SM_RT_cols,
    #     text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #     corr_to_plot=["pearson_corr"],
    #     output_file=f"SM_RT_{set_num}_RTxSenPar_pearson_corr.pdf",
    #     est_strategy="Bootstrap"
    #     )

    # for set_num, SM_RT_cols in zip(["main", 1,2,3], [MAIN_RT_COLS, SM_RT_COLS_SET1, SM_RT_COLS_SET2, SM_RT_COLS_SET3]):
    #     orientation = "vertical"
    #         # Adv_comp Bins
    #     plot_corr_grid_RTx2_RTxSenPar_diff_only(
    #         src_path=src_path,
    #         reader_type="Adv_comp",
    #         reading_regime="FirstReading",
    #         pred_cols=SM_RT_cols,
    #         text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #         corr_to_plot=["pearson_corr"],
    #         output_file=f"SM_advcomp_bins_on_x_{set_num}_RTxSenPar_pearson_corr_{orientation}.pdf",
    #         est_strategy="Bootstrap",
    #         orientation=orientation,
    #         bins_on_x=True,
    #         debug_mode=False
    #     )
    #     # Lextale Bins
    #     plot_corr_grid_RTx2_RTxSenPar_diff_only(
    #         src_path=src_path,
    #         reader_type="Lextale",
    #         reading_regime="FirstReading",
    #         pred_cols=SM_RT_cols,
    #         text_cols=(MAIN_TEXT_COLS+MAIN_SURP_COLS),
    #         corr_to_plot=["pearson_corr"],
    #         output_file=f"SM_lextale_bins_on_x_{set_num}_RTxSenPar_pearson_corr_{orientation}.pdf",
    #         est_strategy="Bootstrap",
    #         orientation=orientation,
    #         bins_on_x=True
    #     )




