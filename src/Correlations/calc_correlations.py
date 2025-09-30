import itertools
import numpy as np
import pandas as pd
from typing import List, Literal
from scipy.stats import pearsonr, spearmanr, permutation_test
from tqdm import tqdm
from loguru import logger
import os
from src.utils.stat_analysis.stat_utils import add_p_val_symbols, get_mean_ci
from src.utils.data_utils import get_text_id_cols, add_id_cols, add_reading_regime_col
from src.utils.files_utils import replace_results_in_file
from src.Correlations.define_cols import (
    MAIN_RT_COLS, MAIN_TEXT_COLS, MAIN_SURP_COLS, ALL_SURP_COLS, SM_TEXT_COLS, SM_RT_COLS, SM_SURP_COLS, SM_PROMPT_COLS, READING_COMPREHENSION_COLS, OPPOSITE_DIRECTION_METRICS
)
from src.Correlations.utils import _del_leg_file_if_exists
from src.constants import DEFAULT_RANDOM_STATE

N_CV_FOLDS = 10
N_BOOTSTRAP = 200
N_PERMUTATION = 1000

# ----------
# Main Funcs
# ----------

def calc_correlations(
    src_path: str,
    resolution: Literal["sentence", "paragraph"], 
    L1_or_L2: Literal["L1", "L2", "L1_and_L2"],
    reading_regime: str,
    pred_type: Literal["RT", "comprehension"],
    run_for_all_surp: bool = False,
    include_bootstrap: bool = False,
    run_for_specific_text_cols: List = None
    ):
    results_dir = src_path / f"Correlations/{L1_or_L2}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"--- {resolution=} | {reading_regime=} ---")
    
    if pred_type == "RT":
        pred_cols = MAIN_RT_COLS + SM_RT_COLS
    
    if run_for_all_surp:
        surp_cols_to_run = list(ALL_SURP_COLS)
    else:
        surp_cols_to_run = MAIN_SURP_COLS + SM_SURP_COLS
    
    # get metrics
    level_metrics_df = _get_metrics_df(src_path, resolution, reading_regime, L1_or_L2, surp_cols_to_run, pred_type)
    existing_surp_cols = [col for col in surp_cols_to_run if col in level_metrics_df.columns]    
    
    if run_for_specific_text_cols:
        run_for_text_cols = run_for_specific_text_cols
    else:
        run_for_text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + existing_surp_cols
    all_cols = pred_cols + run_for_text_cols
    
    # calc diff Adv - Ele
    diff_metrics_df = _calc_diff_in_values_between_levels(level_metrics_df, cols=all_cols, resolution=resolution)
    # create df with all metrics Adv, Ele, Diff
    all_metrics_df = _get_merged_metrics_df(all_cols, level_metrics_df, diff_metrics_df, resolution)
    # save all_metrics_df
    if not run_for_specific_text_cols:
        all_metrics_df.to_csv(results_dir / f"{pred_type}_all_metrics_df_{resolution}.csv", index=False)
    # Add batch_article_id col
    all_metrics_df = add_id_cols(all_metrics_df, batch_article_id=True)

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

def agg_folds_correlations(src_path, resolution, L1_or_L2, reading_regime, include_bootstrap):
    results_dir = src_path / f"Correlations/{L1_or_L2}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{resolution=} | {reading_regime=}")
    
    pred_cols = MAIN_RT_COLS + SM_RT_COLS
    
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
    agg_corr_df = add_p_val_symbols(agg_corr_df, 'pearson_p_boot')
    agg_corr_df = add_p_val_symbols(agg_corr_df, 'spearman_p_boot')
    
    agg_corr_df['pearson_corr_all'] = agg_corr_df['pearson_corr_all'].abs()
    agg_corr_df['spearman_corr_all'] = agg_corr_df['spearman_corr_all'].abs()
    agg_corr_df['pearson_corr_boot'] = agg_corr_df['pearson_corr_boot'].abs()
    agg_corr_df['spearman_corr_boot'] = agg_corr_df['spearman_corr_boot'].abs()
    
    replace_results_in_file(results_dir / f"agg_folds_corr_{resolution}.csv", agg_corr_df)

def calc_perm_test(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    L1_or_L2: Literal["L1", "L2", "general_reader", "L1_and_L2"],
    reading_regime: str,
    est_strategy: Literal["CV", "Bootstrap"],
    surp_cols_to_run: List,
    ):
    results_dir = src_path / f"Correlations/{L1_or_L2}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{resolution=} | {reading_regime=}")
    
    # define statistic
    def _statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    
    # columns should iclude : pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p, spearman_p, n_vals, reading_regime, fold
    
    # groupby corr_df by pred_col, text_col, level_type
    # for each pair of groups: filter fold!=all and calc permutation test between pearson_corr of the two groups
    text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + surp_cols_to_run

    perm_test_dfs = []
    skipped = 0
    skipped_pairs = []
    
    pred_cols = MAIN_RT_COLS + SM_RT_COLS
    for pred_col in tqdm(pred_cols, desc="Calculating permutation tests..."):
        # corr df
        corr_df = pd.read_csv(results_dir / f"correlations_{resolution}_{pred_col}.csv")
        for level_type, group_df in corr_df.groupby('level_type'):
            # iterate on each pair of text_cols
            combinations = list(itertools.combinations(text_cols, 2))
            for text_col_1, text_col_2 in combinations:
                skip_current = False
                # abs corr
                group_df['pearson_corr'] = group_df['pearson_corr'].abs()
                # filter group_df by text_col_1 and text_col_2
                group_1 = group_df[group_df['text_col'] == text_col_1]
                group_2 = group_df[group_df['text_col'] == text_col_2]
                
                if est_strategy == "CV":
                    # filter fold!=all and fold!=bootstrap_all
                    group_1 = group_1[group_1['fold'] != 'bootstrap_all']
                    group_2 = group_2[group_2['fold'] != 'bootstrap_all']
                    group_1 = group_1[group_1['fold'] != 'all']
                    group_2 = group_2[group_2['fold'] != 'all']
                    
                    if len(group_1.dropna()) != N_CV_FOLDS or len(group_2.dropna()) != N_CV_FOLDS:
                        skipped += 1
                        skipped_pairs.append((pred_col, level_type, text_col_1, text_col_2))
                        skip_current = True
                elif est_strategy == "Bootstrap":
                    # based on bootstrap
                    group_1 = group_1[group_1['fold'] == 'bootstrap_all']
                    group_2 = group_2[group_2['fold'] == 'bootstrap_all']
                    if len(group_1.dropna()) != N_BOOTSTRAP or len(group_2.dropna()) != N_BOOTSTRAP:
                        skipped += 1
                        skipped_pairs.append((pred_col, level_type, text_col_1, text_col_2))
                        skip_current = True
                else:
                    raise ValueError(f"Invalid est_strategy: {est_strategy}")
                
                if skip_current:
                    perm_p, perm_stat = None, None
                else:
                    # calc 
                    # tation test
                    perm_test = permutation_test(
                        (group_1['pearson_corr'], group_2['pearson_corr']), 
                        _statistic, vectorized=True, permutation_type='samples', n_resamples=N_PERMUTATION, random_state=DEFAULT_RANDOM_STATE)
                    perm_p, perm_stat = perm_test.pvalue, perm_test.statistic
                
                # append to perm_test_dfs
                perm_test_dfs.append({
                    'pred_col': pred_col,
                    'level_type': level_type,
                    'text_col_1': text_col_1,
                    'mean_corr_1': group_1['pearson_corr'].mean(),
                    'std_corr_1': group_1['pearson_corr'].std(),
                    'n_noNan_1': len(group_1.dropna()),
                    'text_col_2': text_col_2,
                    'mean_corr_2': group_2['pearson_corr'].mean(),
                    'std_corr_2': group_2['pearson_corr'].std(),
                    'n_noNan_2': len(group_2.dropna()),
                    'perm_p': perm_p,
                    'perm_stat': perm_stat
                })
                
        if skipped > 0:
            logger.warning(f"Skipped {skipped} pairs due to missing values")
            
        # skipped_pairs
        if skipped_pairs:
            skipped_pairs_df = pd.DataFrame(skipped_pairs, columns=['pred_col', 'level_type', 'text_col_1', 'text_col_2'])
            skipped_pairs_df.to_csv(results_dir / f"skipped_pairs_{resolution}_{est_strategy}.csv", index=False)
        
        perm_test_df = pd.DataFrame(perm_test_dfs)
        # add p val symbols
        perm_test_df = add_p_val_symbols(perm_test_df, 'perm_p')
        replace_results_in_file(results_dir / f"perm_test_{resolution}_{est_strategy}.csv", perm_test_df)
        
        legacy_file = f"perm_test_{resolution}.csv"
        _del_leg_file_if_exists(legacy_file, results_dir)
        
# -------             
# Helpers
# ------- 

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

def _add_integration_cost_metrics(src_path, metrics_df, resolution, merge_cols):
    integ_df = pd.read_csv(src_path / f"Linguistic_Metrics/integration_cost/results/{resolution}s_df_cleaned_results.csv")
    metrics_df = metrics_df.merge(integ_df, on=merge_cols, how="left")
    # check for null values in col 'avg_integration_cost'
    if metrics_df['avg_integration_cost'].isna().sum() > 0:
        logger.warning(f"Null values in avg_integration_cost: {metrics_df['avg_integration_cost'].isna().sum()}")
    return metrics_df

def _add_ppl_metrics(src_path, metrics_df, resolution, merge_cols):
    ppl_df = pd.read_csv(src_path / f"Linguistic_Metrics/pseudo_cloze/data/{resolution}s_df_cleaned_with_PLL.csv")
    metrics_df = metrics_df.merge(ppl_df, on=merge_cols, how="left")
    return metrics_df

def _add_eye_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols):
    # load eye metrics
    for eye_col in ['TF', 'RR', 'SR', 'GD', 'FF', 'NF', 'FD', 'FirstPassGD', 'FirstPassFF', 'HigherPassFixation']:
        if reader_type == "general_reader":
            eye_metric_df = pd.read_csv(src_path / f"Cognitive_Model/data/{reader_type}/{reading_regime}/{resolution}_{eye_col}_df.csv")
        else:
            eye_metric_df = pd.read_csv(src_path / f"Eye_metrics/data/{reader_type}/{reading_regime}/{resolution}_{eye_col}_df.csv")
        
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
    comprehension_df = add_reading_regime_col(comprehension_df)
    comprehension_df = comprehension_df[comprehension_df['reading_regime'] == reading_regime]
    metrics_df = metrics_df.merge(comprehension_df[merge_cols+["comprehension_score"]], on=merge_cols, how="left")
    # QA_RT
    qa_rt_df = pd.read_csv(src_path / f"Eye_metrics/data/{reader_type}/{reading_regime}/{resolution}_QA_RT_df.csv")
    metrics_df = metrics_df.merge(qa_rt_df, on=merge_cols, how="left")
    return metrics_df

def _get_metrics_df(
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

    metrics_df = _load_readability_metrics(src_path, resolution)
    metrics_df, exisiting_surp_cols = _add_surprisal_metrics(src_path, metrics_df, resolution, merge_cols, surp_cols_to_run)
    metrics_df = _add_integration_cost_metrics(src_path, metrics_df, resolution, merge_cols)
    metrics_df = _add_ppl_metrics(src_path, metrics_df, resolution, merge_cols)
    metrics_df = _add_eye_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols)
    metrics_df = _add_reading_speed_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols)

    if resolution != "sentence" and pred_type == "comprehension":
        metrics_df = _add_reading_comprehension_metrics(src_path, metrics_df, resolution, reading_regime, reader_type, merge_cols) 
        # select cols 
        select_cols = merge_cols + MAIN_TEXT_COLS + SM_TEXT_COLS + MAIN_RT_COLS + SM_RT_COLS + SM_PROMPT_COLS + exisiting_surp_cols + READING_COMPREHENSION_COLS
    else:
        select_cols = merge_cols + MAIN_TEXT_COLS + SM_TEXT_COLS + MAIN_RT_COLS + SM_RT_COLS + SM_PROMPT_COLS + exisiting_surp_cols
    
    return metrics_df[select_cols].sort_values(by=merge_cols)

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
    sub_df = metrics_df[['batch', 'article_id', text_id_col, 'fold', col_a, col_b]].dropna()
    
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
    # order by fold
    all_metrics_df = all_metrics_df.sort_values(by=['fold'])
    
    # Initialize lists to store results
    corr_dfs = []
    invalid_dfs = []
    combinations = list(itertools.product(pred_cols, text_cols, ['Adv', 'Ele', 'diff']))  # Convert to list for tqdm
    # save combinations to df
    combinations_df = pd.DataFrame(combinations, columns=['pred_col', 'text_col', 'level_type'])
    combinations_df.to_csv(results_dir / f"combinations_{resolution}.csv", index=False)
    
    # Calculate correlations
    for pred_col, text_col, level_type in tqdm(combinations, desc="Calculating correlations"):
        col_a = f'{level_type}_{text_col}'
        col_b = f'{level_type}_{pred_col}'
        
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
  