from pathlib import Path
import numpy as np
import pandas as pd
from typing import Literal
from tqdm import tqdm
from loguru import logger
from scipy.stats import pearsonr
from src.utils.stat_analysis.stat_utils import add_p_val_symbols
from src.Correlations.analysis.ppl_trend.models_ppl import _get_models_data
from src.utils.stat_analysis.Julia_models import setup_julia
from src.utils.stat_analysis.Julia_models import fit_linear_model # Julia install - run: curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
from src.Correlations.define_cols import (
    MAIN_RT_COLS, ALL_SURP_COLS, 
)
   
def test_ppl_trend(
    src_path: str,
    resolution: Literal["sentence", "paragraph", "article"],
):
    reader_type="L1_and_L2"
    reading_regime="FirstReading"
    pred_cols=MAIN_RT_COLS
    surp_cols=ALL_SURP_COLS
    est_strategy= 'Regular'
    if est_strategy == "Regular":
        pearson_col = 'pearson_corr_all'
    else:
        raise NotImplementedError(f"look at _single_corr_by_perplexity_plot implementation for est_strategy {est_strategy}")
    
    # corr_df has columns: pred_col, text_col, level_type, pearson_corr, spearman_corr,
    #   pearson_p_symbol, spearman_p_symbol
    logger.info(f"Fitting {resolution} | {reading_regime} | {reader_type} | {pred_cols}")
    corr_df = pd.read_csv(src_path / f"Correlations/{reader_type}/{reading_regime}/agg_folds_corr_{resolution}.csv")
    # get models data
    surp_to_ppl, surp_to_family, surp_to_model_name_with_size = _get_models_data(src_path)
    
    # filter text_cols
    corr_df = corr_df[corr_df['text_col'].isin(surp_cols)]
    
    
    comp_res = []
    # Loop over pred_cols, level_types
    for i, pred_col in tqdm(enumerate(pred_cols)):
        sub_corr_df = corr_df[(corr_df['pred_col'] == pred_col) & (corr_df['level_type'].isin(["all", "diff"]))].reset_index(drop=True)
        sub_corr_df = _process_df(sub_corr_df, surp_to_ppl)
        comp_dict = fit_corr_by_ppl_and_EvalMethod(sub_corr_df, pearson_col)
        # add pred_col, level_type to comp_dict
        comp_dict['pred_col'] = pred_col
        comp_dict['level_type'] = "all and diff"
        comp_res.append(comp_dict)
            
            
    comp_res_df = pd.DataFrame(comp_res)
    # add p symbols to comp_res_df
    comp_res_df = add_p_val_symbols(comp_res_df, 'comp_p')
    comp_res_df = add_p_val_symbols(comp_res_df, 'log_comp_p')
    comp_res_df = add_p_val_symbols(comp_res_df, 'ppl_coef_p')
    comp_res_df = add_p_val_symbols(comp_res_df, 'ppl_EvalMethod_coef_p')
    # save
    comp_res_df.to_csv(src_path / f"Correlations/analysis/ppl_trend/test_ppl_trend_{reader_type}_{reading_regime}_{resolution}.csv", index=False)

def _process_df(sub_corr_df, surp_to_ppl):
        
    # drop surp cols with Max
    sub_corr_df = sub_corr_df[~sub_corr_df['text_col'].str.contains('Max')].reset_index(drop=True)
    
    sub_corr_df['clean_surp_col'] = sub_corr_df['text_col'].apply(lambda x: x.split(' Mean')[0])
    # add perplexity to df
    sub_corr_df['perplexity'] = sub_corr_df['clean_surp_col'].apply(lambda x: surp_to_ppl[x])
    # log perplexity
    sub_corr_df['log_perplexity'] = np.log10(sub_corr_df['perplexity'])
    
    # new column EvalMethod: 0 if level_type == 'diff' else 1 if level_type == 'all' else raise error
    sub_corr_df['EvalMethod'] = sub_corr_df['level_type'].apply(lambda x: 0 if x == 'diff' else 1 if x == 'all' else None)
    
    return sub_corr_df
    
def fit_corr_by_ppl_and_EvalMethod(sub_corr_df, pearson_col):
    # check for correlation between perplexity and results
    comp_r, comp_p = pearsonr(sub_corr_df[pearson_col], sub_corr_df['perplexity'])
    log_comp_r, log_comp_p = pearsonr(sub_corr_df[pearson_col], sub_corr_df['log_perplexity'])
    mean_result = sub_corr_df[pearson_col].mean()
    
    formula = f"{pearson_col} ~ 1 + perplexity*EvalMethod"
    coef_table = fit_linear_model(sub_corr_df, pearson_col, formula, silent=True, needed_cols=[pearson_col, 'perplexity', 'EvalMethod'])
    coef_ppl_res = coef_table[coef_table['Name']=="perplexity"][['Coef.', 'Pr(>|t|)']].values.flatten()
    ppl_coef, ppl_coef_p = coef_ppl_res[0], coef_ppl_res[1]
    
    coef_ppl_EvalMethod_res = coef_table[coef_table['Name']=="perplexity & EvalMethod"][['Coef.', 'Pr(>|t|)']].values.flatten()
    ppl_EvalMethod_coef, ppl_EvalMethod_coef_p = coef_ppl_EvalMethod_res[0], coef_ppl_EvalMethod_res[1]
    
    # return comp results in dict
    return {
        'comp_r': comp_r,
        'comp_p': comp_p,
        'log_comp_r': log_comp_r,
        'log_comp_p': log_comp_p,
        'ppl_coef': ppl_coef,
        'ppl_coef_p': ppl_coef_p,
        'ppl_EvalMethod_coef': ppl_EvalMethod_coef,
        'ppl_EvalMethod_coef_p': ppl_EvalMethod_coef_p,
        'mean_result': mean_result,
        'n_values': len(sub_corr_df),
    }
    
if __name__ == "__main__":
    src_path = Path.cwd() / "src"
    setup_julia()
    
    for resolution in ["sentence", "paragraph"]:
        test_ppl_trend(
            src_path=src_path,
            resolution=resolution,
        )