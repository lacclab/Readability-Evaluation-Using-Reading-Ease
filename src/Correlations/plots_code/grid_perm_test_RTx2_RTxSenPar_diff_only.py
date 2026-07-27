import matplotlib.pyplot as plt
from typing import Literal
from loguru import logger
import pandas as pd

from src.utils.plot_utils import add_significance_legend, SIGNIFICANCE_SIGN_DIFF_COLORS, SIGNIFICANCE_SIGN_DIFF_LABELS
from src.Correlations.define_cols import MAIN_RT_COLS, READING_COMPREHENSION_COLS
from src.Correlations.utils import _save_file_to_all_paths, _del_leg_file_if_exists
from src.Correlations.plots_code.single_permutation_test_plot import _single_plot_perm_test, _load_perm_test_res_df

def plot_perm_test_RTx2_RTxSenPar_diff_only(
    src_path: str,
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    est_strategy: Literal["CV", "Bootstrap"],
    pred_type: Literal["RT", "comprehension"],
    fontsize_title=16,
    fontsize_legend_text=12,
    ):
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p_symbol, spearman_p_symbol
    logger.info(f"Plotting | {reading_regime} | {reader_type}")
    
    if pred_type == "RT":
        pred_cols = MAIN_RT_COLS
    elif pred_type == "comprehension":
        pred_cols = READING_COMPREHENSION_COLS
    
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    sentences_perm_test_df = _load_perm_test_res_df(results_dir, est_strategy, "sentence")
    paragraphs_perm_test_df = _load_perm_test_res_df(results_dir, est_strategy, "paragraph")
    
    resolution_types = ['sentence', 'paragraph']
    level_type = 'diff'
    
    # create fig with subplots
    n_rows = len(pred_cols)
    n_cols = len(resolution_types)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*7.5, n_rows*7.5)) # , sharey=True
    
    # Set y-label on the left column, set column titles on top row
    for j, y_type in enumerate(resolution_types):
        y_labels = {'sentence': 'Sentences\n\n', 'paragraph': 'Passages\n\n'}
        axs[0, j].set_title(y_labels[y_type], fontsize=fontsize_title, fontweight='bold')
    
    # Loop over pred_cols, level_types
    for i, pred_col in enumerate(pred_cols):
        for j, resolution in enumerate(resolution_types):
            ax = axs[i, j]
            perm_test_df = sentences_perm_test_df if resolution == 'sentence' else paragraphs_perm_test_df
            sub_perm_test_df = perm_test_df[(perm_test_df['pred_col'] == pred_col) & (perm_test_df['level_type'] == level_type)]
            if sub_perm_test_df.empty:
                logger.warning(f"Empty sub_perm_test_df for {pred_col} at {resolution} level. Skipping...")
                continue
            
            path_steiger_res = src_path / f"Correlations/{reader_type}/{reading_regime}/steiger_test_between_readability_formulas_{resolution}.csv"
            steiger_res = pd.read_csv(path_steiger_res)
            steiger_res = steiger_res[(steiger_res['level_type'] == level_type) & (steiger_res['pred_col'] == pred_col)]
            
            _single_plot_perm_test(
                ax=ax, sub_perm_test_df=sub_perm_test_df, steiger_res=steiger_res,
                pred_col=pred_col, row_index=i, col_index=j,
                level_type=level_type, all_levels=False, resolution=resolution, RTxSenPar=True)
            
            
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
    
    output_file = f"{pred_type}_perm_test_grid_RTxSenPar.pdf"
    
    # overleaf_dir = OVERLEAF_PATH_1 / f"{reader_type}/{reading_regime}/Corr"
    # overleaf_dir.mkdir(parents=True, exist_ok=True)
        
    _save_file_to_all_paths(
        resolution=resolution,
        reader_type=reader_type, 
        reading_regime=reading_regime, 
        output_file=output_file, 
        pred_cols=None, 
        text_cols=None, 
        corr_to_plot=None, src_path=src_path, est_strategy=est_strategy
    )

if __name__ == "__main__":
    from pathlib import Path

    src_path = Path.cwd() / "src"
    L1_or_L2 = "L2" # "L1" or "L2" or "L1_and_L2"
    
    plot_perm_test_RTx2_RTxSenPar_diff_only(
        src_path=src_path,
        reader_type=L1_or_L2,
        reading_regime="Gathering0",
        est_strategy="Bootstrap",
        pred_type="RT"
    )
