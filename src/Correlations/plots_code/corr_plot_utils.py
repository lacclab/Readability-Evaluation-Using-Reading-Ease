import pandas as pd
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from src.utils.plot_utils import SIGNIFICANCE_COLORS, SIGNIFICANCE_LABELS
from src.constants import LEXTALE_BINS_NAMES, ADV_COMP_BINS_NAMES, LEXTALE_BIN_LABELS, ADV_COMP_BIN_LABELS, LEXTALE_BIN_COLORS, ADV_COMP_BIN_COLORS

HATCH_STR_DICT_LABELS = {
    'pearson_spearman': {'': 'Pearson', '///': 'Spearman'},
    'L1_next_to_L2': {'': 'L1', '///': 'L2'},
    'Gathering0_next_to_Hunting0': {'': 'Ordinary Reading', '///': 'Information Seeking'},
    'FirstReading_next_to_RepeatedReading': {'': 'First Reading', '///': 'Repeated Reading'},
    'FirstReading_next_to_Gathering0': {'': 'First Reading', '///': 'Ordinary Reading'},
}
HATCH_STR_DICT = {
    'pearson_spearman': {'Pearson': '', 'Spearman': '///'},
    'L1_next_to_L2': {'L1': '', 'L2': '///'},
    'Gathering0_next_to_Hunting0': {'Gathering0': '', 'Hunting0': '///'},
    'FirstReading_next_to_RepeatedReading': {'FirstReading': '', 'RepeatedReading': '///'},
    'FirstReading_next_to_Gathering0': {'FirstReading': '', 'Gathering0': '///'},
}
OFFSET_DICT = {
    'pearson_spearman': {'Pearson': -0.2, 'Spearman': 0.2},
    'L1_next_to_L2': {'L1': -0.2, 'L2': 0.2},
    'Gathering0_next_to_Hunting0': {'Gathering0': -0.2, 'Hunting0': 0.2},
    'FirstReading_next_to_RepeatedReading': {'FirstReading': -0.2, 'RepeatedReading': 0.2},
    'FirstReading_next_to_Gathering0': {'FirstReading': -0.2, 'Gathering0': 0.2},
}

def _load_corr_df(reader_type, reading_regime, resolution, src_path, text_cols):
    if reader_type == "L1_next_to_L2":
        L1_corr_df = pd.read_csv(src_path / f"Correlations/L1/{reading_regime}/agg_folds_corr_{resolution}.csv")
        L2_corr_df = pd.read_csv(src_path / f"Correlations/L2/{reading_regime}/agg_folds_corr_{resolution}.csv")
        L1_corr_df = _filter_corr_df_by_text_cols(L1_corr_df, text_cols, resolution=resolution)
        L2_corr_df = _filter_corr_df_by_text_cols(L2_corr_df, text_cols, resolution=resolution)
        L1_corr_df['reader_type'] = 'L1'
        L2_corr_df['reader_type'] = 'L2'
        L2_corr_df['reading_regime'] = reading_regime
        L1_corr_df['reading_regime'] = reading_regime
        # Union the two dataframes
        corr_df = pd.concat([L1_corr_df, L2_corr_df], ignore_index=True)
    elif reading_regime == "Gathering0_next_to_Hunting0":
        Gathering_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/Gathering0/agg_folds_corr_{resolution}.csv")
        Hunting_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/Hunting0/agg_folds_corr_{resolution}.csv")
        Gathering_corr_df = _filter_corr_df_by_text_cols(Gathering_corr_df, text_cols, resolution=resolution)
        Hunting_corr_df = _filter_corr_df_by_text_cols(Hunting_corr_df, text_cols, resolution=resolution)
        Gathering_corr_df['reader_type'] = reader_type
        Hunting_corr_df['reader_type'] = reader_type
        Gathering_corr_df['reading_regime'] = 'Gathering0'
        Hunting_corr_df['reading_regime'] = 'Hunting0'
        # Union the two dataframes
        corr_df = pd.concat([Gathering_corr_df, Hunting_corr_df], ignore_index=True)
    elif reading_regime == "FirstReading_next_to_RepeatedReading":
        first_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/FirstReading/agg_folds_corr_{resolution}.csv")
        repeated_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/RepeatedReading/agg_folds_corr_{resolution}.csv")
        first_corr_df = _filter_corr_df_by_text_cols(first_corr_df, text_cols, resolution=resolution)
        repeated_corr_df = _filter_corr_df_by_text_cols(repeated_corr_df, text_cols, resolution=resolution)
        first_corr_df['reader_type'] = reader_type
        repeated_corr_df['reader_type'] = reader_type
        first_corr_df['reading_regime'] = 'FirstReading'
        repeated_corr_df['reading_regime'] = 'RepeatedReading'
        # Union the two dataframes
        corr_df = pd.concat([first_corr_df, repeated_corr_df], ignore_index=True)
    elif reading_regime == "FirstReading_next_to_Gathering0":
        first_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/FirstReading/agg_folds_corr_{resolution}.csv")
        gathering_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/Gathering0/agg_folds_corr_{resolution}.csv")
        first_corr_df = _filter_corr_df_by_text_cols(first_corr_df, text_cols, resolution=resolution)
        gathering_corr_df = _filter_corr_df_by_text_cols(gathering_corr_df, text_cols, resolution=resolution)
        first_corr_df['reader_type'] = reader_type
        gathering_corr_df['reader_type'] = reader_type
        first_corr_df['reading_regime'] = 'FirstReading'
        gathering_corr_df['reading_regime'] = 'Gathering0'
        # Union the two dataframes
        corr_df = pd.concat([first_corr_df, gathering_corr_df], ignore_index=True)
    elif reader_type == "Lextale":
        bins_dfs = []
        for bin_name in LEXTALE_BINS_NAMES:
            reader_type_bin = f"Lextale_{bin_name}"
            bin_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type_bin}/{reading_regime}/agg_folds_corr_{resolution}.csv")
            bin_corr_df = _filter_corr_df_by_text_cols(bin_corr_df, text_cols, resolution=resolution)
            bin_corr_df['reader_type'] = reader_type_bin
            bin_corr_df['reading_regime'] = reading_regime
            bin_corr_df['bin_name'] = bin_name
            bins_dfs.append(bin_corr_df)
        corr_df = pd.concat(bins_dfs, ignore_index=True)
         
    elif reader_type == "Adv_comp":
        bins_dfs = []
        for bin_name in ADV_COMP_BINS_NAMES:
            reader_type_bin = f"Adv_comp_{bin_name}"
            bin_corr_df = pd.read_csv(src_path / f"Correlations/{reader_type_bin}/{reading_regime}/agg_folds_corr_{resolution}.csv")
            bin_corr_df = _filter_corr_df_by_text_cols(bin_corr_df, text_cols, resolution=resolution)
            bin_corr_df['reader_type'] = reader_type_bin
            bin_corr_df['reading_regime'] = reading_regime
            bin_corr_df['bin_name'] = bin_name
            bins_dfs.append(bin_corr_df)
        corr_df = pd.concat(bins_dfs, ignore_index=True)
    else:
        corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
        corr_df = _filter_corr_df_by_text_cols(corr_df, text_cols, resolution=resolution)
        corr_df['reader_type'] = reader_type
        corr_df['reading_regime'] = reading_regime
    return corr_df

def _add_legend_for_hatch(fig, hatch_str_dict):
    # 5) A small legend in the top-right
    #  We'll define two patches for Pearson vs. Spearman
    pear_patch = Patch(facecolor='white', edgecolor='black', hatch='',    label=hatch_str_dict[''])
    spear_patch= Patch(facecolor='white', edgecolor='black', hatch='///', label=hatch_str_dict['///'])

    handels = [pear_patch, spear_patch]
        
    fig.legend(
        handles=handels,
        loc='lower right',
        fontsize=8,
        # title="Correlation Type"
    )
    return fig

def _add_signficance_legend(fig, legend_text_fontsize):
    # Build a single legend for significance colors
    handles = []
    for sig_symbol, color in SIGNIFICANCE_COLORS.items():
        label = f"{sig_symbol} {SIGNIFICANCE_LABELS[sig_symbol]}"
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)

    fig.legend(
        handles=handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.0), 
        ncol=4, 
        fontsize=legend_text_fontsize, 
        frameon=False
    )
    return fig

def _add_bin_names_legend(fig, legend_text_fontsize, bin_type):
    labels = LEXTALE_BIN_LABELS if bin_type == "Lextale" else ADV_COMP_BIN_LABELS
    colors = LEXTALE_BIN_COLORS if bin_type == "Lextale" else ADV_COMP_BIN_COLORS
    bin_type_label = "Lextale" if bin_type == "Lextale" else "Comprehension Score (Based on Advanced Texts)"
    # Build a single legend for bin names colors
    handles = []
    for bin_name, color in colors.items():
        label = labels[bin_name]
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)

    fig.legend(
        handles=handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.0), 
        ncol=5, 
        fontsize=legend_text_fontsize, 
        frameon=False,
        title=f"{bin_type_label} Bin Ranges"
    )
    return fig

def _filter_corr_df_by_text_cols(corr_df, text_cols, resolution):
    # filter text_cols
    corr_df = corr_df[corr_df['text_col'].isin(text_cols)]
    return corr_df