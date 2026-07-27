import matplotlib.pyplot as plt
from typing import Literal
from loguru import logger

from src.utils.plot_utils import add_significance_legend, SIGNIFICANCE_SIGN_DIFF_COLORS, SIGNIFICANCE_SIGN_DIFF_LABELS
from src.Correlations.define_cols import MAIN_RT_COLS, READING_COMPREHENSION_COLS
from src.Correlations.utils import _save_file_to_all_paths, _del_leg_file_if_exists, OVERLEAF_PATH_1
from src.Correlations.plots_code.single_permutation_test_plot import _single_plot_perm_test, _load_perm_test_res_df

def plot_perm_test_results_all_levels(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    est_strategy: Literal["CV", "Bootstrap"],
    pred_type: Literal["RT", "comprehension"],
    fontsize_legend_text=12,
    ):
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    logger.info(f"{resolution=} | {reading_regime=}")
    
    if pred_type == "RT":
        pred_cols = MAIN_RT_COLS
    elif pred_type == "comprehension":
        pred_cols = READING_COMPREHENSION_COLS
    
    perm_test_df = _load_perm_test_res_df(results_dir, est_strategy, resolution)
    
    # create fig with subplots
    n_rows = len(pred_cols)
    n_cols = len(['Adv', 'Ele', 'diff'])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7.5, n_rows*7.5), sharey=True)
    
    # Loop over pred_cols, level_types
    for i, pred_col in enumerate(pred_cols):
        for j, level_type in enumerate(['Adv', 'Ele', 'diff']):
            ax = axs[i, j]
            sub_perm_test_df = perm_test_df[(perm_test_df['pred_col'] == pred_col) & (perm_test_df['level_type'] == level_type)]
            _single_plot_perm_test(ax, sub_perm_test_df, pred_col, i, level_type, all_levels=True, resolution=resolution)
            
    # add significance legend
    fig = add_significance_legend(
        fig, 
        significance_colors=SIGNIFICANCE_SIGN_DIFF_COLORS,
        significance_labels=SIGNIFICANCE_SIGN_DIFF_LABELS,
        fontsize_legend_text=fontsize_legend_text
        )
    
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    legacy_file = f"perm_test_{resolution}.pdf"
    _del_leg_file_if_exists(legacy_file, results_dir)
    legacy_file = f"{pred_type}_perm_test_{resolution}.pdf"
    _del_leg_file_if_exists(legacy_file, results_dir)
    
    output_file = f"{pred_type}_perm_test_{resolution}_{est_strategy}.pdf"
    
    overleaf_dir = OVERLEAF_PATH_1 / f"{reader_type}/{reading_regime}/Corr"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(results_dir / output_file)
    plt.savefig(overleaf_dir / output_file)
        
    _save_file_to_all_paths(
        resolution=resolution,
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=None, 
        text_cols=None, 
        corr_to_plot=None, src_path=src_path, est_strategy=est_strategy
    )

