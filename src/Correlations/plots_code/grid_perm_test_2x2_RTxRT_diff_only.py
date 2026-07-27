import matplotlib.pyplot as plt
from typing import Literal
from loguru import logger

from src.constants import PRED_COLS_FULL_LABELS
from src.utils.plot_utils import add_significance_legend, SIGNIFICANCE_SIGN_DIFF_COLORS, SIGNIFICANCE_SIGN_DIFF_LABELS
from src.Correlations.utils import _save_file_to_all_paths, _del_leg_file_if_exists
from src.Correlations.plots_code.single_permutation_test_plot import _single_plot_perm_test, _load_perm_test_res_df

def plot_perm_test_results_diff_levels(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    est_strategy: Literal["CV", "Bootstrap"],
    pred_type: Literal["RT", "comprehension"],
    fontsize_title=20,
    fontsize_legend_text=14,
    ):
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    logger.info(f"{resolution=} | {reading_regime=}")
    
    # define level type
    level_type = 'diff'
    
    first_row_cols = ['reading_speed', 'mean_nonzero_TF']
    first_row_cols_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[col]}" for col in first_row_cols]
    second_row_cols = ['SkipRateTotal', 'RegRateTotal']
    second_row_cols_labels = [f"$\Delta$ {PRED_COLS_FULL_LABELS[col]}" for col in second_row_cols]
    
    perm_test_df = _load_perm_test_res_df(results_dir, est_strategy, resolution)
    
    # create fig with subplots
    n_rows = 2
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7.5, n_rows*7.5), sharey=True)
    
    # Set y-label on the left column, set column titles on top row
    for j in range(n_cols):
        axs[0, j].set_title(first_row_cols_labels[j], fontsize=fontsize_title, fontweight='bold')
        axs[1, j].set_title(second_row_cols_labels[j], fontsize=fontsize_title, fontweight='bold')
    
    # Loop over pred_cols, level_types
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            pred_col = first_row_cols[j] if i == 0 else second_row_cols[j]
            sub_perm_test_df = perm_test_df[(perm_test_df['pred_col'] == pred_col) & (perm_test_df['level_type'] == level_type)]
            _single_plot_perm_test(ax, sub_perm_test_df, pred_col, i, level_type, all_levels=False, resolution=resolution)
            
    # add significance legend
    fig = add_significance_legend(
        fig, 
        significance_colors=SIGNIFICANCE_SIGN_DIFF_COLORS,
        significance_labels=SIGNIFICANCE_SIGN_DIFF_LABELS,
        fontsize_legend_text=fontsize_legend_text
        )
    
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # # add space between subplots
    # plt.subplots_adjust(hspace=0.5, wspace=0.05)    

    legacy_file = f"perm_test_{resolution}_diff_only.pdf"
    _del_leg_file_if_exists(legacy_file, results_dir)
    legacy_file = f"{pred_type}_perm_test_{resolution}_diff_only.pdf"
    _del_leg_file_if_exists(legacy_file, results_dir)

    output_file = f"{pred_type}_perm_test_{resolution}_{est_strategy}_diff_only.pdf"

    _save_file_to_all_paths(
        resolution=resolution, 
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=None, 
        text_cols=None, 
        corr_to_plot=None, src_path=src_path, est_strategy=est_strategy
    )
