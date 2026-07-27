from pathlib import Path
import itertools
import pandas as pd
from typing import List, Literal
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import numpy as np
from loguru import logger
import os
from src.utils.stat_analysis.stat_utils import add_p_val_symbols, get_mean_ci, p_to_star
from src.utils.data_utils import get_text_id_cols, add_id_cols, add_reading_regime_col
from src.utils.files_utils import replace_results_in_file
from src.readability_metrics.add_metrics_utils import merge_and_save
from src.Correlations.define_cols import (
    MAIN_RT_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS, ALL_SURP_COLS, SM_TEXT_COLS, SM_RT_COLS, SM_SURP_COLS, SM_PROMPT_COLS, READING_COMPREHENSION_COLS, OPPOSITE_DIRECTION_METRICS
)

N_CV_FOLDS = 10
N_BOOTSTRAP = 200

# ----------
# Main Funcs
# ----------

def calc_correlations(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    pred_type: Literal["RT", "comprehension"],
    run_for_all_surp: bool = False,
    include_bootstrap: bool = False,
    run_for_specific_text_cols: List = None
    ):
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"--- {resolution=} | {reading_regime=} ---")
    
    if pred_type == "RT":
        pred_cols = MAIN_RT_COLS + SM_RT_COLS
    elif pred_type == "comprehension":
        pred_cols = READING_COMPREHENSION_COLS
    
    if run_for_all_surp:
        surp_cols_to_run = list(ALL_SURP_COLS)
    else:
        surp_cols_to_run = MAIN_SURP_COLS + SM_SURP_COLS
    
    # get metrics
    level_metrics_df = _get_ele_adv_metrics_df(src_path, resolution, reading_regime, reader_type, surp_cols_to_run, pred_type)
    existing_surp_cols = [col for col in surp_cols_to_run if col in level_metrics_df.columns]    
    
    if run_for_specific_text_cols:
        run_for_text_cols = run_for_specific_text_cols
    else:
        run_for_text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + existing_surp_cols
    # dedupe while preserving order (MAIN_TEXT_COLS and SM_TEXT_COLS may overlap)
    run_for_text_cols = list(dict.fromkeys(run_for_text_cols))
    all_cols = list(dict.fromkeys(pred_cols + run_for_text_cols))
    
    # add diff metrics
    all_metrics_df = _add_diff_metrics(resolution, all_cols, level_metrics_df)
    
    # save all_metrics_df
    # When running with specific text cols, merge into the existing CSV so we replace
    # overlapping cols and add new ones without losing previously-computed cols.
    # When running with the full col set, overwrite (the new df is authoritative).
    csv_path = results_dir / f"{pred_type}_all_metrics_df_{resolution}.csv"
    if run_for_specific_text_cols and csv_path.exists():
        merge_keys = get_text_id_cols(resolution)
        logger.info(f"Merging {len(all_metrics_df.columns)} cols into existing {csv_path.name} (merge_on={merge_keys})")
        merge_and_save(all_metrics_df, csv_path, merge_on=merge_keys)
    else:
        all_metrics_df.to_csv(csv_path, index=False)
    
    # --- Calculate Correlations ---
    corr_df = _get_corr_df(resolution, all_metrics_df, run_for_text_cols, reading_regime, pred_cols, results_dir, include_bootstrap)
    
    # save each pred_col to separate file
    for pred_col in pred_cols:
        # if exists legacy file of all pred_cols, delete it
        legacy_file = results_dir / f"correlations_{resolution}.csv"
        if legacy_file.exists():
            logger.warning(f"Deleting legacy file: {legacy_file}")
            os.remove(legacy_file)
        replace_results_in_file(results_dir / f"correlations_{resolution}_{pred_col}.csv", corr_df[corr_df['pred_col'] == pred_col], second_col='text_col')

def agg_folds_correlations(src_path, resolution, L1_or_L2, reading_regime, include_bootstrap, pred_cols):
    results_dir = src_path / f"Correlations/{L1_or_L2}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{resolution=} | {reading_regime=}")
    
    # groupby corr_df by pred_col, text_col, level_type
    agg_corr_dfs = []
    for pred_col in pred_cols:
        corr_df = pd.read_csv(results_dir / f"correlations_{resolution}_{pred_col}.csv")
        for (pred_col, level_type, text_col), group_df in corr_df.groupby(['pred_col', 'level_type', 'text_col']):
            # get pearson, spearman, p_val over all
            group_df_all = group_df[group_df['fold'] == 'all']
            pearson_corr_all = group_df_all['pearson_corr'].item()
            spearman_corr_all = group_df_all['spearman_corr'].item()
            pearson_p_all = group_df_all['pearson_p'].item()
            spearman_p_all = group_df_all['spearman_p'].item()
            
            # agg CV results
            # filter out fold=all
            # CV_df = group_df[(group_df['fold'] != 'all') & (group_df['fold'] != 'bootstrap_all')]
            # agged_CV = _agg_over_groups(CV_df)
            
            # bootstrap results]
            if include_bootstrap:
                boot_df = group_df[group_df['fold'] == 'bootstrap_all']
                agged_boot = _agg_over_groups(boot_df)
            else:
                agged_boot = {
                    'n_bootstraps': None,
                    'pearson_corr': None,
                    'spearman_corr': None,
                    'pearson_p': None,
                    'spearman_p': None,
                    'std_pearson_corr': None,
                    'std_spearman_corr': None,
                    'CI_yerr_pearson': None,
                    'CI_yerr_spearman': None
                }
            
            # append to agg_corr_dfs
            agg_corr_dfs.append({
                'pred_col': pred_col,
                'text_col': text_col,
                'level_type': level_type,
                
                'pearson_corr_all': pearson_corr_all,
                'spearman_corr_all': spearman_corr_all,
                'pearson_p_all': pearson_p_all,
                'spearman_p_all': spearman_p_all,
                
                # 'pearson_corr_CV': agged_CV['pearson_corr'],
                # 'spearman_corr_CV': agged_CV['spearman_corr'],
                # 'pearson_p_CV': agged_CV['pearson_p'],
                # 'spearman_p_CV': agged_CV['spearman_p'],
                # 'std_pearson_corr_CV': agged_CV['std_pearson_corr'],
                # 'std_spearman_corr_CV': agged_CV['std_spearman_corr'],
                # 'CI_yerr_pearson_CV': agged_CV['CI_yerr_pearson'],
                # 'CI_yerr_spearman_CV': agged_CV['CI_yerr_spearman'],
                
                'n_bootstraps': agged_boot['n_bootstraps'],
                'pearson_corr_boot': agged_boot['pearson_corr'],
                'spearman_corr_boot': agged_boot['spearman_corr'],
                'pearson_p_boot': agged_boot['pearson_p'],
                'spearman_p_boot': agged_boot['spearman_p'],
                'std_pearson_corr_boot': agged_boot['std_pearson_corr'],
                'std_spearman_corr_boot': agged_boot['std_spearman_corr'],
                'CI_yerr_pearson_boot': agged_boot['CI_yerr_pearson'],
                'CI_yerr_spearman_boot': agged_boot['CI_yerr_spearman']
            })
        
    agg_corr_df = pd.DataFrame(agg_corr_dfs)
    
    # check for rows of bootsrap with less than N_BOOTSTRAP n_bootstraps
    not_full_boot = agg_corr_df[(agg_corr_df['n_bootstraps'].notna()) & (agg_corr_df['n_bootstraps'] < N_BOOTSTRAP)]
    if len(not_full_boot) > 0:
        logger.warning(f"Some bootstrap aggregations have less than {N_BOOTSTRAP} bootstraps:")
        # save to file
        not_full_boot.to_csv(results_dir / f"warning_not_full_bootstrap_{resolution}.csv", index=False)
    
    # add symbols
    agg_corr_df = add_p_val_symbols(agg_corr_df, 'pearson_p_all')
    agg_corr_df = add_p_val_symbols(agg_corr_df, 'spearman_p_all')
    # agg_corr_df = add_p_val_symbols(agg_corr_df, 'pearson_p_CV')
    # agg_corr_df = add_p_val_symbols(agg_corr_df, 'spearman_p_CV')
    agg_corr_df = add_p_val_symbols(agg_corr_df, 'pearson_p_boot')
    agg_corr_df = add_p_val_symbols(agg_corr_df, 'spearman_p_boot')
    
    agg_corr_df['pearson_corr_all'] = agg_corr_df['pearson_corr_all'].abs()
    agg_corr_df['spearman_corr_all'] = agg_corr_df['spearman_corr_all'].abs()
    # agg_corr_df['pearson_corr_CV'] = agg_corr_df['pearson_corr_CV'].abs()
    # agg_corr_df['spearman_corr_CV'] = agg_corr_df['spearman_corr_CV'].abs()
    agg_corr_df['pearson_corr_boot'] = agg_corr_df['pearson_corr_boot'].abs()
    agg_corr_df['spearman_corr_boot'] = agg_corr_df['spearman_corr_boot'].abs()
    
    replace_results_in_file(results_dir / f"agg_folds_corr_{resolution}.csv", agg_corr_df)

def calc_RT_corr_for_pair_plots(src_path, resolution):
    RT_cols = MAIN_RT_COLS + SM_RT_COLS
    
    # L1 next to L2
    reading_regime = "FirstReading"
    reader_type = 'L1_next_to_L2'
    L1_metrics_df, L2_metrics_df = _load_L1_and_L2_RT_metrics(
        src_path, 
        resolution, 
        RT_cols,
        reading_regime,
        reader_type
    )
    corrs = calc_corr_between_cols(RT_cols, L1_metrics_df, L2_metrics_df)
    corrs['reader_type'] = reader_type
    corrs['reading_regime'] = reading_regime
    corrs['resolution'] = resolution
    # save corrs
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    corrs.to_csv(results_dir / f"RT_corrs_{resolution}.csv", index=False)
    
    # Gathering0 next to Hunting0
    reading_regime = 'Gathering0_next_to_Hunting0'
    reader_type = "L1_and_L2"
    Gathering0_metrics_df, Hunting0_metrics_df = _load_Gathering0_and_Hunting0_RT_metrics(
        src_path, 
        resolution, 
        RT_cols,
        reading_regime,
        reader_type
    )
    corrs = calc_corr_between_cols(RT_cols, Gathering0_metrics_df, Hunting0_metrics_df)
    corrs['reader_type'] = reader_type
    corrs['reading_regime'] = reading_regime
    corrs['resolution'] = resolution
    # save corrs
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    corrs.to_csv(results_dir / f"RT_corrs_{resolution}.csv", index=False)

def calc_corr_between_formulas_for_pair_plots(src_path, resolution):
    text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + MAIN_SURP_COLS + SM_SURP_COLS
    # dedupe (MAIN_TEXT_COLS and SM_TEXT_COLS may overlap)
    text_cols = list(dict.fromkeys(text_cols))

    text_id_cols = get_text_id_cols(resolution)
    merge_cols = text_id_cols + ["level"]
    metrics_df = _load_readability_metrics(src_path, resolution)
    # PPL Pythia 70M is derived from 'Pythia 70M Mean' — ensure it's merged in.
    surp_cols_for_this_call = list(MAIN_SURP_COLS)
    if "PPL Pythia 70M" in text_cols and "Pythia 70M Mean" not in surp_cols_for_this_call:
        surp_cols_for_this_call.append("Pythia 70M Mean")
    metrics_df, exisiting_surp_cols = _add_surprisal_metrics(src_path, metrics_df, resolution, merge_cols, surp_cols_for_this_call)
    metrics_df = _add_ppl_metrics(metrics_df)
    metrics_df = _add_integration_cost_metrics(src_path, metrics_df, resolution, merge_cols)
    metrics_df = _add_pll_metrics(src_path, metrics_df, resolution, merge_cols)
    
    # add diff metrics
    metrics_df = _add_diff_metrics(
        resolution, 
        metrics_cols=text_cols, 
        level_metrics_df=metrics_df
        )
    
    corrs = []
    for level_type in ['Adv', 'Ele', 'diff']:
        for r_i, r_name in enumerate(text_cols):
            for c_i, c_name in enumerate(text_cols):
                if r_name == c_name:
                    continue
                if (c_name, r_name) in corrs or (r_name, c_name) in corrs:
                    continue
                    
                # calc corr between cols
                col_a_name = f"{level_type}_{r_name}"
                col_b_name = f"{level_type}_{c_name}"
                clean_df = metrics_df[[col_a_name, col_b_name]].dropna()
                # log how many rows where dropped
                dropped = len(metrics_df) - len(clean_df)
                N_SENTENCES_NO_MATCH = 94
                if dropped > N_SENTENCES_NO_MATCH:
                    print(f"Dropping {dropped} out of {len(metrics_df)} rows for correlation between {col_a_name} and {col_b_name}")
                pearson_corr, pearson_p = pearsonr(clean_df[col_a_name], clean_df[col_b_name])

                symbol = p_to_star(pearson_p)
                corrs.append({
                    'level_type': level_type,
                    'text_col_1': r_name,
                    'text_col_2': c_name,
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'pearson_p_symbol': symbol,
                })

    corrs = pd.DataFrame(corrs)
    corrs['resolution'] = resolution
    # save corrs
    results_dir = src_path / "readability_metrics/correlations_between_formulas"
    corrs.to_csv(results_dir / f"formulas_corrs_{resolution}.csv", index=False)

# -------             
# Helpers
# ------- 

def calc_corr_between_cols(cols, df_1, df_2):
    # calc corr for each col between df_1 and df_2
    corr_dfs = []
    for col in cols:
        for prefix in ['Adv_', 'Ele_', 'diff_']:
            col_name = prefix + col
            if col_name in df_1.columns and col_name in df_2.columns:
                corr = df_1[col_name].corr(df_2[col_name], method='pearson')
                corr_dfs.append({
                    'col': col,
                    'full_col': col_name,
                    'pearson_corr': corr
                })
            else:
                logger.warning(f"Column {col} not found in both dataframes")
    return pd.DataFrame(corr_dfs)

def _add_diff_metrics(
    resolution: Literal["sentence", "paragraph", "article"], 
    metrics_cols: List,
    level_metrics_df: pd.DataFrame,
):  
    # calc diff Adv - Ele
    diff_metrics_df = _calc_diff_in_values_between_levels(level_metrics_df, cols=metrics_cols, resolution=resolution)
    # create df with all metrics Adv, Ele, Diff
    all_metrics_df = _get_merged_metrics_df(metrics_cols, level_metrics_df, diff_metrics_df, resolution)
    # Add batch_article_id col
    all_metrics_df = add_id_cols(all_metrics_df, batch_article_id=True)
    return all_metrics_df

def _get_ele_adv_metrics_df(
    src_path, 
    resolution: Literal["sentence", "paragraph", "article"], 
    reading_regime: str, 
    reader_type: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    surp_cols_to_run: List,
    pred_type: Literal["RT", "comprehension"]
    ):
    text_id_cols = get_text_id_cols(resolution)
    merge_cols = text_id_cols + ["level"]
    logger.info(f"Loading metrics for resolution={resolution} | merge_keys={merge_cols}")

    # PPL Pythia 70M is a main col derived from 'Pythia 70M Mean' (surp col) — ensure it gets merged.
    if "PPL Pythia 70M" in MAIN_TEXT_COLS and "Pythia 70M Mean" not in surp_cols_to_run:
        surp_cols_to_run = list(surp_cols_to_run) + ["Pythia 70M Mean"]

    metrics_df = _load_readability_metrics(src_path, resolution)
    metrics_df, exisiting_surp_cols = _add_surprisal_metrics(src_path, metrics_df, resolution, merge_cols, surp_cols_to_run)
    metrics_df = _add_ppl_metrics(metrics_df)
    metrics_df = _add_integration_cost_metrics(src_path, metrics_df, resolution, merge_cols)
    metrics_df = _add_pll_metrics(src_path, metrics_df, resolution, merge_cols)
    metrics_df = _add_eye_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols)
    metrics_df = _add_reading_speed_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols)

    if resolution != "sentence" and pred_type == "comprehension":
        metrics_df = _add_reading_comprehension_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols)
        # select cols
        select_cols = merge_cols + MAIN_TEXT_COLS + SM_TEXT_COLS + MAIN_RT_COLS + SM_RT_COLS + SM_PROMPT_COLS + exisiting_surp_cols + READING_COMPREHENSION_COLS
    else:
        select_cols = merge_cols + MAIN_TEXT_COLS + SM_TEXT_COLS + MAIN_RT_COLS + SM_RT_COLS + SM_PROMPT_COLS + exisiting_surp_cols

    # dedupe while preserving order (MAIN_TEXT_COLS and SM_TEXT_COLS may overlap)
    select_cols = list(dict.fromkeys(select_cols))

    return metrics_df[select_cols].sort_values(by=merge_cols)


def _load_readability_metrics(src_path, resolution):
    metrics_df = pd.read_csv(src_path / f"readability_metrics/data/{resolution}s_metrics_cleaned.csv")
    metrics_df = metrics_df.rename(columns={"text_length": "n_words"})
    if resolution == "sentence":
        metrics_df = metrics_df.dropna(subset=["sentence"])
    elif resolution == "article":
        metrics_df = add_id_cols(metrics_df, de_unique_article_id=True)
    return metrics_df 

def _load_surprisal_metrics(src_path, resolution):
    surp_df = pd.read_csv(src_path / f"Linguistic_Metrics/data/OneStop_{resolution}s_with_surprisal_renamed.csv")
    # add id cols
    if resolution == "article":
        surp_df = add_id_cols(surp_df, de_unique_article_id=True)
    else:
        surp_df = add_id_cols(surp_df, de_unique_paragraph_id=True)
        surp_df = add_id_cols(surp_df, text_id=True)
    return surp_df

def _merge_with_surp_df(metrics_df, surp_df, merge_cols, surp_cols_to_run):
    # select cols
    exisiting_surp_cols = [col for col in surp_df.columns if col in surp_cols_to_run]
    surp_df = surp_df[merge_cols + exisiting_surp_cols]
    metrics_df = metrics_df.merge(surp_df, on=merge_cols, how="left")
    return metrics_df, exisiting_surp_cols

def _add_surprisal_metrics(src_path, metrics_df, resolution, merge_cols, surp_cols_to_run):
    surp_df = _load_surprisal_metrics(src_path, resolution)
    metrics_df, exisiting_surp_cols = _merge_with_surp_df(metrics_df, surp_df, merge_cols, surp_cols_to_run)
    
    not_existing_surp_cols = [col for col in surp_cols_to_run if col not in exisiting_surp_cols]
    # take not_existing_surp_cols from another df
    if not_existing_surp_cols:
        surp_df = pd.read_csv(src_path / f"Linguistic_Metrics/data/OneStop_{resolution}s_with_surprisal_renamed.csv")
        # add id cols
        if resolution == "article":
            surp_df = add_id_cols(surp_df, de_unique_article_id=True)
        else:
            surp_df = add_id_cols(surp_df, de_unique_paragraph_id=True)
            surp_df = add_id_cols(surp_df, text_id=True)
        # select cols
        exisiting_surp_cols = [col for col in surp_df.columns if col in not_existing_surp_cols]
        surp_df = surp_df[merge_cols + exisiting_surp_cols]
        metrics_df = metrics_df.merge(surp_df, on=merge_cols, how="left")
        
    # log how many surp cols are in metrics_df and how many are missing
    exisiting_surp_cols = [col for col in surp_cols_to_run if col in metrics_df]
    missing_surp_cols = [col for col in surp_cols_to_run if col not in metrics_df]
    logger.info(f"Existing surp cols: {len(exisiting_surp_cols)} | Missing surp cols: {missing_surp_cols}")
    
    return metrics_df, exisiting_surp_cols

def _add_ppl_metrics( metrics_df):
    if 'Pythia 70M Mean' in metrics_df.columns: 
        metrics_df['PPL Pythia 70M'] = np.exp(metrics_df['Pythia 70M Mean']) # calc using exp of mean surprisal
    return metrics_df

def _add_integration_cost_metrics(src_path, metrics_df, resolution, merge_cols):
    integ_df = pd.read_csv(src_path / f"Linguistic_Metrics/integration_cost/results/{resolution}s_df_cleaned_results.csv")
    metrics_df = metrics_df.merge(integ_df, on=merge_cols, how="left")
    # check for null values in col 'avg_integration_cost'
    if metrics_df['avg_integration_cost'].isna().sum() > 0:
        logger.warning(f"Null values in avg_integration_cost: {metrics_df['avg_integration_cost'].isna().sum()}")
    return metrics_df

def _add_pll_metrics(src_path, metrics_df, resolution, merge_cols):
    pll_df = pd.read_csv(src_path / f"Linguistic_Metrics/pseudo_cloze/data/{resolution}s_df_cleaned_with_PLL.csv")
    metrics_df = metrics_df.merge(pll_df, on=merge_cols, how="left")
    return metrics_df

def _add_eye_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols):
    # load eye metrics
    for eye_col in ['TF', 'RR', 'SR', 'GD', 'FF', 'NF', 'FD', 'FirstPassGD', 'FirstPassFF', 'HigherPassFixation']:
        if reader_type == "general_reader":
            eye_metric_df = pd.read_csv(src_path / f"Cognitive_Model/data/{reader_type}/{reading_regime}/{resolution}_{eye_col}_df.csv")
        else:
            eye_metric_df = pd.read_csv(src_path / f"Eye_metrics/data/{reader_type}/{reading_regime}/metric_tables/{resolution}_{eye_col}_df.csv")
        
        # remove cols n_subjects n_rows from eye_metric_df if exist
        if "n_subjects" in eye_metric_df.columns:
            eye_metric_df = eye_metric_df.drop(columns=["n_subjects", "n_rows"])
        
        metrics_df = metrics_df.merge(eye_metric_df, on=merge_cols, how="left")
    return metrics_df

def _add_reading_speed_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols):
    # load reading speed metrics
    speed_df = pd.read_csv(src_path / f"Eye_metrics/data/{reader_type}/reading_speed/{reading_regime}/speed_by={resolution}_and_level.csv")
    if resolution == "article":
        speed_col = "Art_words_per_sec_based_P_RT"
    elif resolution == "paragraph":
        speed_col = "words_per_sec_based_P_RT"
    elif resolution == "sentence":
        speed_col = "words_per_sec_based_Sen_RT"
    speed_df = speed_df[merge_cols + [speed_col]]
    # rename speed_col
    speed_df = speed_df.rename(columns={speed_col: "reading_speed"})
    metrics_df = metrics_df.merge(speed_df, on=merge_cols, how="left")
    return metrics_df

def _add_reading_comprehension_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols):
    path = src_path / f"Reading_Comprehension/data/{reader_type}/{reading_regime}/comprehension_scores_by={resolution}_and_level.csv"
    comprehension_df = pd.read_csv(path)
    if resolution == "article":
        comprehension_df = add_id_cols(comprehension_df, de_unique_article_id=True)
    # filter by reading_regime
    comprehension_df = comprehension_df[comprehension_df['reading_regime'] == reading_regime]
    metrics_df = metrics_df.merge(comprehension_df[merge_cols+["comprehension_score"]], on=merge_cols, how="left")
    # QA_RT
    qa_rt_df = pd.read_csv(src_path / f"Eye_metrics/data/{reader_type}/{reading_regime}/metric_tables/{resolution}_QA_RT_df.csv")
    metrics_df = metrics_df.merge(qa_rt_df, on=merge_cols, how="left")
    return metrics_df

def _calc_diff_in_values_between_levels(metrics_df: pd.DataFrame, cols: List, resolution: Literal["sentence", "paragraph", "article"]):
    # Calculate the difference in mean TF per sentence between levels Adv and Ele
    pivot_by_cols = get_text_id_cols(resolution)
    all_diff_df = pd.DataFrame()
    for col in cols:
        diff_df = metrics_df.pivot_table(index=(pivot_by_cols), columns='level', values=col).reset_index()
        diff_df[f'diff_{col}'] = diff_df['Adv'] - diff_df['Ele']
        if all_diff_df.empty:
            all_diff_df = diff_df[pivot_by_cols+[f'diff_{col}']]
        else:
            all_diff_df = all_diff_df.merge(diff_df[pivot_by_cols+[f'diff_{col}']], on=pivot_by_cols, how='outer')

    return all_diff_df

def _get_merged_metrics_df(metrics_cols, level_metrics_df, diff_metrics_df, resolution):
    """
    Merge the metrics DataFrames to create a single DataFrame with all metrics for both levels and the difference between levels.
    """
    # Determine the merge keys based on resolution
    merge_keys = get_text_id_cols(resolution)
    
    # Step 1: Split metrics_df into Adv and Ele based on the level column
    select_cols = merge_keys + metrics_cols
    adv_df = level_metrics_df[level_metrics_df['level'] == 'Adv'][select_cols].copy()
    ele_df = level_metrics_df[level_metrics_df['level'] == 'Ele'][select_cols].copy()

    # Rename columns to indicate Adv and Ele
    adv_df = adv_df.rename(lambda x: f"Adv_{x}" if x in metrics_cols else x, axis=1)
    ele_df = ele_df.rename(lambda x: f"Ele_{x}" if x in metrics_cols else x, axis=1)

    # Step 2: Merge Adv and Ele DataFrames on the merge keys
    combined_metrics = adv_df.merge(
        ele_df,
        on=merge_keys,
        how='left'
    )
    # check that n rows is the same
    assert len(adv_df) == len(combined_metrics)

    # Step 3: Merge the combined metrics DataFrame with all_diff_df on the merge keys
    final_df = combined_metrics.merge(
        diff_metrics_df,
        on=merge_keys,
        how='left'
    )
    # check that n rows is the same
    assert len(adv_df) == len(combined_metrics)
    # log number of rows
    logger.info(f"Number of rows in final_df: {len(final_df)}")
    # log n rows with null values
    logger.info(f"Number of rows with null values: {final_df.isna().any(axis=1).sum()}")
    return final_df

def _get_curr_corr(
    metrics_df, col_a, col_b,
    pred_col, text_col, level_type, 
    reading_regime, text_id_col, fold, bootstrap_iter
    ):
    if level_type != 'all':
        sub_df = metrics_df[['batch', 'article_id', text_id_col, 'fold', col_a, col_b]].dropna()
    else:
        # take both Adv and Ele
        ele_df = metrics_df[['batch', 'article_id', text_id_col, 'fold', f'Ele_{text_col}', f'Ele_{pred_col}']].dropna()
        ele_df = ele_df.rename(columns={f'Ele_{text_col}': col_a, f'Ele_{pred_col}': col_b})
        ele_df['level'] = 'Ele'
        adv_df = metrics_df[['batch', 'article_id', text_id_col, 'fold', f'Adv_{text_col}', f'Adv_{pred_col}']].dropna()
        adv_df = adv_df.rename(columns={f'Adv_{text_col}': col_a, f'Adv_{pred_col}': col_b})
        adv_df['level'] = 'Adv'
        sub_df = pd.concat([ele_df, adv_df], ignore_index=True)
        assert len(ele_df) + len(adv_df) == len(sub_df)
    
    pearson_corr, pearson_p = pearsonr(sub_df[col_a], sub_df[col_b])
    spearman_corr, spearman_p = spearmanr(sub_df[col_a], sub_df[col_b])  
    return {
        'pred_col': pred_col,
        'text_col': text_col,
        'level_type': level_type,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'pearson_p': pearson_p,
        'spearman_p': spearman_p,
        'n_vals': len(sub_df),
        'reading_regime': reading_regime,
        'fold': fold,
        'bootstrap_iter': bootstrap_iter,
        'n_batches': len(sub_df['batch'].unique()),
        'n_articles': len(sub_df['article_id'].unique()),
        'n_texts': len(sub_df[text_id_col].unique()),
        'n_rows': len(sub_df)
    }

def _validate_df_before_corr(metrics_df, col_a, col_b, run_name):
    col_a_None = False
    col_b_None = False
    col_a_const = False
    col_b_const = False
    valid = True
    
    if metrics_df[col_a].dropna().empty:
        col_a_None = True
        valid = False
    if metrics_df[col_b].dropna().empty:
        col_b_None = True
        valid = False
    if metrics_df[col_a].nunique() == 1:
        col_a_const = True
        valid = False
    if metrics_df[col_b].nunique() == 1:
        col_b_const = True
        valid = False
    
    # build df
    valid_df = pd.DataFrame({
        'run_name': [run_name],
        'col_a': [col_a],
        'col_b': [col_b],
        'col_a_None': [col_a_None],
        'col_b_None': [col_b_None],
        'col_a_const': [col_a_const],
        'col_b_const': [col_b_const],
        'n_vals': [len(metrics_df)]
    })
    
    return valid, valid_df
    
def _get_corr_df(resolution, all_metrics_df, text_cols, reading_regime, pred_cols, results_dir, include_bootstrap):
    if resolution == "article":
        text_id_col = "batch_article_id"
    else:
        text_id_col = "text_id"
    
    # create fold column by article_id
    all_metrics_df['fold'] = all_metrics_df['article_id'].astype(int)
    # Order by a UNIQUE row key (fold + text id, + sentence align_idx where present) and reset the
    # index, so the seeded bootstrap resample below (sample(random_state=...)) is REPRODUCIBLE from
    # the saved RT_all_metrics. Sorting by the non-unique 'fold' alone left ties ordered
    # non-deterministically, so the resample couldn't be reproduced. Only the bootstrap depends on
    # row order; fold='all' and per-CV-fold correlations are order-invariant, so this changes no
    # point estimate (only which rows each bootstrap draw picks, which is arbitrary anyway).
    sort_keys = ['fold', text_id_col] + (['align_idx'] if 'align_idx' in all_metrics_df.columns else [])
    all_metrics_df = all_metrics_df.sort_values(by=sort_keys).reset_index(drop=True)
    
    # Initialize lists to store results
    corr_dfs = []
    invalid_dfs = []
    combinations = list(itertools.product(pred_cols, text_cols, ['all', 'Adv', 'Ele', 'diff']))  # Convert to list for tqdm
    # save combinations to df
    combinations_df = pd.DataFrame(combinations, columns=['pred_col', 'text_col', 'level_type'])
    combinations_df.to_csv(results_dir / f"combinations_{resolution}.csv", index=False)
    
    # Calculate correlations
    for pred_col, text_col, level_type in tqdm(combinations, desc="Calculating correlations"):
        col_a = f'{level_type}_{text_col}'
        col_b = f'{level_type}_{pred_col}'
        
        if level_type != 'all':
            valid, valid_df = _validate_df_before_corr(all_metrics_df, col_a, col_b, 'Regular')
            if not valid:
                invalid_dfs.append(valid_df)
                continue
        
        corr_dfs.append(_get_curr_corr(
            all_metrics_df, col_a, col_b, 
            pred_col, text_col, level_type, 
            reading_regime, text_id_col, fold="all", bootstrap_iter=None))
        
        # Corr for each fold
        for fold, fold_df in all_metrics_df.groupby('fold'):
            if level_type != 'all':
                valid, valid_df = _validate_df_before_corr(fold_df, col_a, col_b, 'CV')
                if not valid:
                    invalid_dfs.append(valid_df)
                    continue
            
            corr_dfs.append(_get_curr_corr(
                fold_df, col_a, col_b, 
                pred_col, text_col, level_type, 
                reading_regime, text_id_col, fold, bootstrap_iter=None))
                
    logger.info(f"N valid: {len(corr_dfs)} | N invalid: {len(invalid_dfs)}")
    
    # bootstrap over full data
    # resample data
    if include_bootstrap:
        for i in tqdm(range(1, N_BOOTSTRAP+1), desc="Bootstrapping"):
            resample_df = all_metrics_df.sample(frac=1, replace=True, random_state=(42+2*i))
            for pred_col, text_col, level_type in combinations:
                col_a = f'{level_type}_{text_col}'
                col_b = f'{level_type}_{pred_col}'
                
                if level_type != 'all':
                    valid, valid_df = _validate_df_before_corr(resample_df, col_a, col_b, 'Bootstrap')
                    if not valid:
                        continue
                    
                corr_dfs.append(_get_curr_corr(
                    resample_df, col_a, col_b, 
                    pred_col, text_col, level_type, 
                    reading_regime, text_id_col, fold="bootstrap_all", bootstrap_iter=i))
    
    # Create DataFrame from list of dictionaries
    corr_df = pd.DataFrame(corr_dfs)
    # Add p val symbols
    corr_df = add_p_val_symbols(corr_df, 'pearson_p')
    corr_df = add_p_val_symbols(corr_df, 'spearman_p')
    
    # concat invalid dfs
    if invalid_dfs:
        invalid_df = pd.concat(invalid_dfs)
        invalid_df.to_csv(results_dir / f"invalid_corrs_{resolution}.csv", index=False)
    
    return corr_df

def _agg_over_groups(
    group_df, 
    ):
    # get mean of pearson_corr
    mean_pearson_corr = group_df['pearson_corr'].mean()
    # get std of pearson_corr
    std_pearson_corr = group_df['pearson_corr'].std()
    # CI of pearson_corr
    _, _, CI_yerr_pearson = get_mean_ci(group_df, 'pearson_corr', sem_or_std='std')
    # get mean of spearman_corr
    mean_spearman_corr = group_df['spearman_corr'].mean()
    # get std of spearman_corr
    std_spearman_corr = group_df['spearman_corr'].std()
    # CI of spearman_corr
    _, _, CI_yerr_spearman = get_mean_ci(group_df, 'spearman_corr', sem_or_std='std')
    # agg p val
    mean_pearson_p = group_df['pearson_p'].mean()
    mean_spearman_p = group_df['spearman_p'].mean()
    
    return {
        'n_bootstraps': len(group_df),
        'pearson_corr': mean_pearson_corr,
        'std_pearson_corr': std_pearson_corr,
        'CI_yerr_pearson': CI_yerr_pearson,
        'spearman_corr': mean_spearman_corr,
        'std_spearman_corr': std_spearman_corr,
        'CI_yerr_spearman': CI_yerr_spearman,
        'pearson_p': mean_pearson_p,
        'spearman_p': mean_spearman_p
    }
  
def _handle_opposite_direction_metrics(all_diff_df: pd.DataFrame, metrics_to_plot: List):
    updated_metrics = metrics_to_plot.copy()
    for metric in metrics_to_plot:
        if metric in OPPOSITE_DIRECTION_METRICS:
            all_diff_df[f'diff_{metric}_(-)'] = -all_diff_df[f'diff_{metric}']
            # replace
            updated_metrics[updated_metrics.index(metric)] = f'{metric}_(-)'
    return all_diff_df, updated_metrics
    

def _load_Gathering0_and_Hunting0_RT_metrics(src_path, resolution, RT_cols, reading_regime, reader_type):
    # load metrics
    Gathering0_level_metrics_df = _get_ele_adv_metrics_df(
        src_path=src_path, 
        resolution=resolution, 
        reading_regime="Gathering0", 
        reader_type=reader_type, 
        surp_cols_to_run=[], 
        pred_type="RT"
    )
    Hunting0_level_metrics_df = _get_ele_adv_metrics_df(
        src_path=src_path, 
        resolution=resolution, 
        reading_regime="Hunting0", 
        reader_type=reader_type, 
        surp_cols_to_run=[], 
        pred_type="RT"
    )
    # add diff metrics
    Gathering0_metrics_df = _add_diff_metrics(
        resolution, 
        metrics_cols=RT_cols, 
        level_metrics_df=Gathering0_level_metrics_df
        )
    Hunting0_metrics_df = _add_diff_metrics(
        resolution,
        metrics_cols=RT_cols,
        level_metrics_df=Hunting0_level_metrics_df
    )
    return Gathering0_metrics_df, Hunting0_metrics_df


def _load_L1_and_L2_RT_metrics(src_path, resolution, RT_cols, reading_regime, reader_type):
    # load metrics
    L1_level_metrics_df = _get_ele_adv_metrics_df(
        src_path=src_path, 
        resolution=resolution, 
        reading_regime=reading_regime, 
        reader_type="L1", 
        surp_cols_to_run=[], 
        pred_type="RT"
    )
    L2_level_metrics_df = _get_ele_adv_metrics_df(
        src_path=src_path, 
        resolution=resolution, 
        reading_regime=reading_regime, 
        reader_type="L2", 
        surp_cols_to_run=[], 
        pred_type="RT"
    )
    # add diff metrics
    L1_metrics_df = _add_diff_metrics(
        resolution, 
        metrics_cols=RT_cols, 
        level_metrics_df=L1_level_metrics_df
        )
    L2_metrics_df = _add_diff_metrics(
        resolution,
        metrics_cols=RT_cols,
        level_metrics_df=L2_level_metrics_df
    )
    return L1_metrics_df, L2_metrics_df


if __name__== "__main__":
    src_path = Path.cwd() / "src"
    calc_corr_between_formulas_for_pair_plots(src_path=src_path, resolution="sentence")
    calc_corr_between_formulas_for_pair_plots(src_path=src_path, resolution="paragraph")
    # calc_corr_between_formulas_for_pair_plots(src_path=src_path, resolution="article")