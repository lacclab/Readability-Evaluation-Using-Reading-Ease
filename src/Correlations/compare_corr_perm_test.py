import itertools
import numpy as np
import pandas as pd
from typing import List, Literal
from scipy.stats import permutation_test
from tqdm import tqdm
from loguru import logger
from src.utils.stat_analysis.stat_utils import add_p_val_symbols, p_to_star
from src.utils.files_utils import replace_results_in_file
from src.Correlations.define_cols import (
    MAIN_RT_COLS, MAIN_TEXT_COLS, SM_TEXT_COLS, SM_RT_COLS, SM_PROMPT_COLS
)
from src.Correlations.calc_correlations import N_BOOTSTRAP, N_CV_FOLDS
from src.Correlations.utils import _del_leg_file_if_exists
from src.constants import DEFAULT_RANDOM_STATE

N_PERMUTATION = 1000


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
    
    # columns should iclude : pred_col, text_col, level_type, pearson_corr, spearman_corr, pearson_p, spearman_p, n_vals, reading_regime, fold
    
    # groupby corr_df by pred_col, text_col, level_type
    # for each pair of groups: filter fold!=all and calc permutation test between pearson_corr of the two groups
    text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + surp_cols_to_run
    # dedupe — MAIN_TEXT_COLS and SM_TEXT_COLS may overlap, which would produce
    # duplicate (text_col_1, text_col_2) pairs and break the downstream pivot.
    text_cols = list(dict.fromkeys(text_cols))

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
        

def _compare_corr_using_perm_test(group_1, group_2):
    # assert 200 rows in each group
    assert len(group_1.dropna()) == N_BOOTSTRAP and len(group_2.dropna()) == N_BOOTSTRAP, f"Not enough bootstrap samples: {len(group_1.dropna())}, {len(group_2.dropna())}"
    
    # permutation test
    perm_test = permutation_test(
        (group_1, group_2), 
        _statistic, vectorized=True, permutation_type='samples', n_resamples=N_PERMUTATION, random_state=DEFAULT_RANDOM_STATE)
    perm_p, perm_stat = perm_test.pvalue, perm_test.statistic
    perm_star = p_to_star(perm_p)
    return perm_p, perm_stat, perm_star

# define statistic
def _statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)