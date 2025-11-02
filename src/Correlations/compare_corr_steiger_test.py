from pathlib import Path
import itertools
import pandas as pd
from typing import List, Literal
from tqdm import tqdm
from loguru import logger
from src.utils.stat_analysis.stat_utils import add_p_val_symbols
from utils.stat_analysis.cocor import cocor_test, init_cocor, setup_julia # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
from src.Correlations.define_cols import (
    MAIN_RT_COLS, MAIN_TEXT_COLS, SM_TEXT_COLS, SM_RT_COLS, SM_PROMPT_COLS, MAIN_SURP_COLS
)

def calc_steiger_test_between_RT(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    pair_plot_type: str,
    surp_cols_to_run: List,
    ):
    if pair_plot_type == "L1_next_to_L2":
        reader_type = "L1_next_to_L2"
        reading_regime = "FirstReading"
        corr_dir_1 = src_path / f"Correlations/L1/{reading_regime}"
        corr_dir_2 = src_path / f"Correlations/L2/{reading_regime}"
    elif pair_plot_type == "Gathering0_next_to_Hunting0":
        reader_type = "L1_and_L2"
        reading_regime = "Gathering0_next_to_Hunting0"
        corr_dir_1 = src_path / f"Correlations/{reader_type}/Gathering0"
        corr_dir_2 = src_path / f"Correlations/{reader_type}/Hunting0"
    else:
        raise ValueError(f"Unknown pair_plot_type: {pair_plot_type}")
    
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{resolution=} | {reading_regime=}")
    
    text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + surp_cols_to_run
    corr_between_RT_cols = _load_corr_between_RT_cols(results_dir, resolution)

    cocor = init_cocor()

    dfs = []
    pred_cols = MAIN_RT_COLS + SM_RT_COLS
    for pred_col in tqdm(pred_cols, desc="Calculating Steiger tests between RT columns..."):
        # corr df
        corr_df_1 = _load_corr_df(corr_dir_1, resolution, pred_col)
        corr_df_1['group'] = 'group_1'
        corr_df_2 = _load_corr_df(corr_dir_2, resolution, pred_col)
        corr_df_2['group'] = 'group_2'
        corr_df = pd.concat([corr_df_1, corr_df_2], ignore_index=True)
        # filter only fold == all
        corr_df = corr_df[corr_df['fold'] == 'all']
        for level_type, level_df in corr_df.groupby('level_type'):
            if level_type == "all":
                continue
            
            # abs corr
            level_df['pearson_corr'] = level_df['pearson_corr'].abs()
            
            # corr between RT
            corr_h_k = corr_between_RT_cols[(corr_between_RT_cols['col'] == pred_col) & (corr_between_RT_cols['full_col'].str.contains(level_type))]['pearson_corr'].item()
            
            for text_col in text_cols:
                # filter group_df by text_col_1 and text_col_2
                sub_df = level_df[level_df['text_col'] == text_col]

                group_1 = sub_df[sub_df['group'] == 'group_1']
                group_2 = sub_df[sub_df['group'] == 'group_2']

                corr_j_h = group_1['pearson_corr'].item()
                corr_j_k = group_2['pearson_corr'].item()
                
                n1 = group_1['n_rows'].item()
                n2 = group_2['n_rows'].item()
                assert n1 == n2, f"n1 != n2: {n1} != {n2}"
                n = n1
                
                # steiger test
                p_val, test_stat = cocor_test(cocor, corr_j_h, corr_j_k, corr_h_k, n)
                
                # append to dfs
                dfs.append({
                    'pair_plot_type': pair_plot_type,
                    'pred_col': pred_col,
                    'level_type': level_type,
                    'text_col': text_col,
                    'corr_1': group_1['pearson_corr'].mean(),
                    'corr_2': group_2['pearson_corr'].mean(),
                    'corr_h_k': corr_h_k,
                    'n': n,
                    'test_stat': test_stat,
                    'p_val': p_val,
                })

    result_df = pd.DataFrame(dfs)
    result_df['resolution'] = resolution
    result_df['reading_regime'] = reading_regime
    result_df['reader_type'] = reader_type
    # add p val symbols
    result_df = add_p_val_symbols(result_df, 'p_val')
    
    # save df
    result_df.to_csv(results_dir / f"steiger_test_between_RT_cols_{resolution}_{pair_plot_type}.csv", index=False)


def calc_steiger_test_between_readability_formulas(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"], 
    pair_plot_type: str,
    surp_cols_to_run: List,
    ):
    if pair_plot_type == "readability_formulas":
        reader_type = "L1_and_L2"
        reading_regime = "FirstReading"
    else:
        raise ValueError(f"Unknown pair_plot_type: {pair_plot_type}")
    
    results_dir = src_path / f"Correlations/{reader_type}/{reading_regime}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{resolution=} | {reading_regime=}")
    
    text_cols = MAIN_TEXT_COLS + SM_TEXT_COLS + SM_PROMPT_COLS + surp_cols_to_run
    corr_between_formulas = _load_corr_between_formulas(src_path, resolution)

    cocor = init_cocor()

    dfs = []
    pred_cols = MAIN_RT_COLS + SM_RT_COLS
    for pred_col in tqdm(pred_cols, desc="Calculating Steiger tests between readability formulas..."):
        # corr df
        corr_df = _load_corr_df(results_dir, resolution, pred_col)
        # filter only fold == all
        corr_df = corr_df[corr_df['fold'] == 'all']
        for level_type, group_df in corr_df.groupby('level_type'):
            if level_type == "all":
                continue
            
            level_df = corr_between_formulas[(corr_between_formulas['level_type'] == level_type)]
            
            # iterate on each pair of text_cols
            combinations = list(itertools.combinations(text_cols, 2))
            for text_col_1, text_col_2 in combinations:
                # abs corr
                group_df['pearson_corr'] = group_df['pearson_corr'].abs()
                # filter group_df by text_col_1 and text_col_2
                group_1 = group_df[group_df['text_col'] == text_col_1]
                group_2 = group_df[group_df['text_col'] == text_col_2]
                
                corr_j_h = group_1['pearson_corr'].item()
                corr_j_k = group_2['pearson_corr'].item()
                
                n1 = group_1['n_rows'].item()
                n2 = group_2['n_rows'].item()
                assert (n1 - n2 < 2), f"n1 != n2: {n1} != {n2}"
                n = n1
                
                # corr between formulas
                corr_h_k = level_df[
                    ((level_df['text_col_1'] == text_col_1) & (level_df['text_col_2'] == text_col_2))
                    ]['pearson_corr'].item()
                
                # steiger test
                p_val, test_stat = cocor_test(cocor, corr_j_h, corr_j_k, corr_h_k, n)
                
                # append to dfs
                dfs.append({
                    'pred_col': pred_col,
                    'level_type': level_type,
                    'text_col_1': text_col_1,
                    'corr_1': group_1['pearson_corr'].mean(),
                    'text_col_2': text_col_2,
                    'corr_2': group_2['pearson_corr'].mean(),
                    'corr_h_k': corr_h_k,
                    'n': n,
                    'test_stat': test_stat,
                    'p_val': p_val,
                })

    result_df = pd.DataFrame(dfs)
    result_df['resolution'] = resolution
    result_df['reading_regime'] = reading_regime
    result_df['reader_type'] = reader_type
    # add p val symbols
    result_df = add_p_val_symbols(result_df, 'p_val')
    
    # save df
    result_df.to_csv(results_dir / f"steiger_test_between_readability_formulas_{resolution}.csv", index=False)

#### Helper functions

def _load_corr_between_formulas(src_path, resolution):
    path = src_path / f"readability_metrics/formulas_corrs_{resolution}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    else:
        return pd.read_csv(path)

def _load_corr_df(results_dir, resolution, pred_col):
    path = results_dir / f"correlations_{resolution}_{pred_col}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    else:
        df = pd.read_csv(path)
        return df

def _load_corr_between_RT_cols(results_dir, resolution):
    path = results_dir / f"RT_corrs_{resolution}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    else:
        df = pd.read_csv(path)
        return df
    
#########

if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    setup_julia()
    calc_for_resolutions = ["paragraph", "sentence"]
    
    for resolution in calc_for_resolutions:
        for pair_plot_type in ["L1_next_to_L2", "Gathering0_next_to_Hunting0"]:
            calc_steiger_test_between_RT(
                src_path, resolution, 
                pair_plot_type,
                surp_cols_to_run=MAIN_SURP_COLS)
        calc_steiger_test_between_readability_formulas(
            src_path, resolution, 
            pair_plot_type="readability_formulas",
            surp_cols_to_run=MAIN_SURP_COLS)