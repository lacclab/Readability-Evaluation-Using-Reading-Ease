from pathlib import Path
import matplotlib.pyplot as plt
from src.Correlations.define_cols import TEXT_COLS_FULL_LABELS

OVERLEAF_PATH_1 = Path('~/66e12f60cd9d9e737ca6d75a/Plots/all').expanduser() # ARR
OVERLEAF_PATH_2 = Path('~/68998e16eecd06d41561b8a3/Plots/all').expanduser() # PNAS
OVERLEAF_PATH_3 = Path('~/68aac4893381c1f962751f07/Plots/all').expanduser() # PNAS SI
OVERLEAF_PATH_4 = Path('~/695a3ba742696807861e77f6/Plots/all').expanduser() # CL Journal

OVERLEAF_PATHS = [OVERLEAF_PATH_4]
# OVERLEAF_PATHS = [OVERLEAF_PATH_1, OVERLEAF_PATH_2, OVERLEAF_PATH_3]

def _save_file_to_all_paths(resolution, reader_type, reading_regime, output_file, pred_cols, text_cols, corr_to_plot, src_path, est_strategy):
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for overleaf_path in OVERLEAF_PATHS:
        overleaf_dir = overleaf_path / f"{reader_type}/{reading_regime}/Corr"
        overleaf_dir.mkdir(parents=True, exist_ok=True)
        
        new_output_file = _update_name_by_est_strategy_and_del_legacy(
            est_strategy, output_file, results_dir, overleaf_dir
            )
        plt.savefig(overleaf_dir / new_output_file, bbox_inches='tight')
        
        # call save_latex_figure
        save_latex_figure(
            overleaf_dir=overleaf_dir,
            pdf_file_name=new_output_file,
            resolution=resolution,
            pred_cols=pred_cols,
            text_cols=text_cols,
            corr_to_plot=corr_to_plot,
            reading_regime=reading_regime,
            reader_type=reader_type
        )
    
    plt.savefig(results_dir / new_output_file, bbox_inches='tight')
    plt.close()
    
def _update_name_by_est_strategy_and_del_legacy(
    est_strategy, output_file, results_dir, overleaf_dir
    ):
    if est_strategy == "CV":
        new_output_file = output_file.replace(".pdf", "_CV.pdf")
    elif est_strategy == "Bootstrap":
        new_output_file = output_file.replace(".pdf", "_boot.pdf")
    else:
        new_output_file = output_file
    
    # delete legacy file with suffix _no_CV if exist
    if "_no_CV" in output_file:
        legacy_file = output_file.replace(".pdf", "_no_CV.pdf")
        _del_leg_file_if_exists(legacy_file, results_dir, overleaf_dir)
    
    return new_output_file


def _del_leg_file_if_exists(output_file, results_dir, overleaf_dir = None):
    if (results_dir / output_file).exists():
        (results_dir / output_file).unlink()
    if overleaf_dir and (overleaf_dir / output_file).exists():
        (overleaf_dir / output_file).unlink()

def save_latex_figure(
    overleaf_dir: Path,
    pdf_file_name: str,
    resolution: str,
    reader_type: str = None,
    reading_regime: str = None,
    pred_cols: list = None,
    text_cols: list = None,
    corr_to_plot: list = None,
    result_dir: Path = None
):
    """
    Creates a .tex file in 'overleaf_dir' referencing 'pdf_file_name'.
    The figure caption is built using 'resolution', 'pred_cols', and 'text_cols'.
    """

    # # 1) Construct the .tex file name by replacing .pdf with .tex
    # if reader_type and reading_regime:
    #     overleaf_dir = overleaf_dir / f"{reader_type}/{reading_regime}/Corr"

    latex_file_name = pdf_file_name.replace(".pdf", ".tex")
    latex_file_name = f"tex_{latex_file_name}"
    latex_path = overleaf_dir / latex_file_name
    
    # if latex file already exists, skip
    if latex_path.exists():
        return

    # 2) Build a caption referencing the user data = 
    if reading_regime == "Hunting0":
        reading_regime_str = r" \textit{{Information Seeking}} Reading Regime."
    else:
        reading_regime_str = ""
        
    if reader_type == "general_reader":
        reader_type_str = r" Reading Times are generated using EZ-Reader with the default hyperparameters."
    else:
        reader_type_str = ""
    if text_cols:
        if 'SM_text' in pdf_file_name:
            relevant_text_labels = [TEXT_COLS_FULL_LABELS[col] for col in text_cols]
            text_list_str = ", ".join(relevant_text_labels)
            text_str = f"Text columns: {text_list_str}. "
        else:
            text_str = ""
    else:
        text_str = ""
    
    if 'corr' in pdf_file_name:
        if corr_to_plot == ["pearson_corr"]:
            which_corr_str = "Pearson"
        elif corr_to_plot == ["spearman_corr"]:
            which_corr_str = "Spearman"
        else:
            which_corr_str = "Pearson and Spearman"
    
        caption_text = (
            rf"\textbf{{{which_corr_str} Correlations for \textit{{{resolution}}} alignment.{reading_regime_str}{reader_type_str}}} "
            rf"{text_str}"
            r"Error bars represent 95\% confidence intervals. Colors represent correlation significance."
        )
    elif 'perm_test' in pdf_file_name:
        caption_text = (
            rf"\textbf{{Pearson Correlations Comparison for \textit{{{resolution}}} alignment.{reading_regime_str}{reader_type_str}}} "
            rf"Correlations are between reading times metrics and text metrics. "
            r"Each cell $(i,j)$ compares the average correlation for metric $i$ to that for metric $j$, "
            r"each computed over ten cross‐validation folds. "
            r"The color reflects the difference in those mean correlations and indicates whether this difference is statistically significant, "
            r"as assessed by a paired permutation test."
        )
    elif 'align' in pdf_file_name:
        caption_text = "align surp"
    
    if reading_regime == "Gathering0" and "main_pearson" in pdf_file_name:
        # full_size_str = "figure"
        # width = 0.48
        full_size_str = "figure*"
        width = 1
    else:
        full_size_str = "figure*"
        width = 1

    # 3) Build a short LaTeX figure environment referencing the same PDF
    figure_label = pdf_file_name.replace(".pdf", "")  # e.g. "Ele_effects_on_RT"
    if reading_regime == "Hunting0":
        figure_label = f"{figure_label}_Hunting0"
    if reader_type == "general_reader":
        figure_label = f"{figure_label}_EZ_general"
    
    if not result_dir:
        result_dir = f"Plots/all/{reader_type}/{reading_regime}/Corr"
    figure_path = Path(result_dir) / pdf_file_name
    latex_content = rf"""\begin{{{full_size_str}}}[ht]
    \centering
    \includegraphics[width={{{width}}}\textwidth]{{{figure_path}}}
    \caption{{{caption_text}}}
    \label{{fig:{figure_label}}}
\end{{{full_size_str}}}
    """

    # 4) Write the .tex file
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
