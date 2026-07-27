from loguru import logger
import pandas as pd
import numpy as np
import scipy.stats
import time

import juliapkg
juliapkg.require_julia("=1.12.4")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401

###############################################################################
# Julia Setup
###############################################################################
def setup_julia(needs_cocor=False):
    """
    Install and import the needed Julia packages.
    """
    print(jl.seval("VERSION"))
    jl.seval("import Pkg")
    jl.seval('Pkg.add("GLM")')
    jl.seval('Pkg.add("MixedModels")')
    jl.seval('Pkg.add("DataFrames")')
    jl.seval('Pkg.add("Distributions")')
    jl.seval('Pkg.add("RCall")')  # for R packages
    jl.seval("using MixedModels")
    jl.seval("using DataFrames")
    jl.seval("using Distributions")
    jl.seval("using GLM")
    jl.seval("using RCall")
    
    if needs_cocor:
        import rpy2.robjects.packages as rpackages  # noqa: E402
        # Check if 'cocor' R package is installed
        utils = rpackages.importr('utils')
        if not rpackages.isinstalled("cocor"):
            from rpy2.robjects.vectors import StrVector  # noqa: E402
            print("R package 'cocor' not found. Installing now...")
            utils.chooseCRANmirror(ind=1)  # Select a CRAN mirror
            utils.install_packages(StrVector(['cocor']))
            print("R package 'cocor' has been installed.")
        else:
            print("R package 'cocor' is already installed.")
    
    # set number of threads in BLAS for reproducibility and speed (if needed, adjust the number of threads based on your system)
    N_THREADS = 40
    jl.seval("using LinearAlgebra")
    n_threads_default = jl.seval("LinearAlgebra.BLAS.get_num_threads()")
    jl.seval(f"LinearAlgebra.BLAS.set_num_threads({N_THREADS})")
    n_threads_after = (jl.seval("LinearAlgebra.BLAS.get_num_threads()")) # Verify it was applied
    if n_threads_default != n_threads_after:
        logger.info(f"Set BLAS threads to {n_threads_after} (was {n_threads_default})")
    
    logger.info("Julia environment is set up with MixedModels and DataFrames.")
    
###############################################################################
# Helpers
###############################################################################
def add_CIs_to_coef_df(coef_df: pd.DataFrame, dof: int) -> pd.DataFrame:
    """
    Adds 95% CI columns (l_conf, u_conf) to a coefficient table.

    Parameters
    ----------
    coef_df : pd.DataFrame
        The coefficient table from MixedModels.
    dof : int
        Degrees of freedom, used to compute the t-quantile.

    Returns
    -------
    pd.DataFrame
        The original coef_df with additional columns for lower and upper CI.
    """
    coef_df_copy = coef_df.copy()
    t_quantile = scipy.stats.t(df=dof).ppf(0.975)
    coef_df_copy["l_conf"] = coef_df_copy["Coef."] - t_quantile * coef_df_copy["Std. Error"]
    coef_df_copy["u_conf"] = coef_df_copy["Coef."] + t_quantile * coef_df_copy["Std. Error"]
    coef_df_copy["dof"] = dof
    coef_df_copy["t_quantile"] = t_quantile
    return coef_df_copy

def choose_link_dist(df, outcome_variable: str, all_normal=False):
    """

    Args:
        outcome_variable (str): can be: ['TF', 'SkipTotal', 'SkipFirstPass', 'IsReg', 'RegCountTotal', 'RegCountFirstPass']

    Raises:
        ValueError: if the outcome_variable is not recognized

    Returns:
        jl.Distributions: the appropriate link distribution for the given outcome variable
    """
    if all_normal:
        return jl.Distributions.Normal()
    if outcome_variable in ["SkipTotal", "SkipFirstPass", "IsReg"]:
        # validate that this col is binary
        unique_vals = df[outcome_variable].unique()
        if len(unique_vals) != 2 or set(unique_vals) != {0, 1}:
            raise ValueError(f"Column '{outcome_variable}' is not binary.")
        # Bernoulli distribution
        link_dist = jl.Distributions.Bernoulli()
    elif outcome_variable in [
        "RegCountTotal",
        "RegCountFirstPass",
    ]:
        # validate that this col is integer
        if not df[outcome_variable].dtype == int:
            raise ValueError(f"Column '{outcome_variable}' is not integer.")
        # check if there aren't values above 1
        if not (df[outcome_variable] > 1).any():
            logger.warning(f"Column '{outcome_variable}' has no values above 1.")
        # Poisson distribution
        link_dist = jl.Distributions.Poisson()
    elif outcome_variable in ['TF']:
        # Normal distribution
        link_dist = jl.Distributions.Normal()
    else:
        raise ValueError(f"Unknown outcome variable: {outcome_variable}")
    return link_dist

def convert_coef_by_link_dist(coef_df: pd.DataFrame, link_dist) -> pd.DataFrame:
    # TODO: implement this function
    return coef_df

def _validate_nulls(df, needed_cols = 'all'):
    if needed_cols == 'all':
        # get n nulls
        n_nulls = df.isnull().any(axis=1).sum()
        # Drop nulls
        df = df.dropna()
    else:
        # get n nulls only in needed_cols
        n_nulls = df[needed_cols].isnull().any(axis=1).sum()
        # Drop nulls only in needed_cols
        df = df.dropna(subset=needed_cols)
    
    if n_nulls > 0:
        logger.warning(f"Dropping {n_nulls} rows out of {len(df)} ({n_nulls/len(df)*100:.1f}%) with null values")
        
    if df.empty:
        raise ValueError("No valid rows remain after dropping NA.")
    return df

def _pass_df_to_julia(df: pd.DataFrame):
    # Pass dataframe to Julia
    jl.seval("global j_df = 0") # need to define before assigning
    jl.j_df = jlconvert(jl.PyTable, df)
    
def _pass_convert_df_to_julia():
    # Define a Julia function to convert Julia tables to pandas DataFrames
    # This function uses PythonCall to facilitate the conversion
    jl.seval(
    """
        function table_to_pd(x)
            # Materialize columns as Python lists so pandas never sees
            # Julia-wrapped arrays: PythonCall 0.9.31's __array__ forwards
            # copy=None to numpy, which numpy<2 rejects.
            # TODO use PythonCall.Compat.pytable(x) if possible
            jdf = DataFrames.DataFrame(x)
            pd = PythonCall.pyimport("pandas")
            cols = PythonCall.pydict()
            for name in DataFrames.names(jdf)
                cols[name] = PythonCall.pylist(jdf[!, name])
            end
            pd.DataFrame(cols)
        end
    """
    )

def validate_unique_vals(df, predict_col, formula):
    # checks that pred_col is not constant
    if len(df[predict_col].unique()) == 1:
        logger.warning(f"Column '{predict_col}' is constant. Skipping model fit.")
        return False
    return True

###############################################################################
# Models
###############################################################################
def fit_mixed_effects_model(df: pd.DataFrame, predict_col: str, formula: str, silent: bool = False) -> pd.DataFrame:
    """
    Fit a mixed effects model to the given data.
    """
    df = _validate_nulls(df)
    _pass_df_to_julia(df)
    _pass_convert_df_to_julia()
    ok_flag = validate_unique_vals(df, predict_col, formula)
    if not ok_flag:
        return "skipped"
    
    # Create formula in Julia
    jl.seval(f"j_formula = @formula({formula})")
    # Choose link distribution
    jl.seval("global link_dist = 0")
    link_dist = choose_link_dist(df, predict_col, all_normal=True)
    jl.link_dist = link_dist
    # Fit the model
    if not silent:
        logger.info("Fitting model...")
    # save time for logging
    start_time = time.time()
    jl.seval("model_res = fit(MixedModel, j_formula, j_df, link_dist, progress=false)")
    model_res_name = "model_res"
    # log time in minutes and seconds
    elapsed_time = time.time() - start_time
    if not silent:
        logger.info(f"Model fit in {elapsed_time//60:.0f} minutes, {elapsed_time%60:.0f} seconds")
    # Extract coefficient table and degrees of freedom
    mm_coeftable = jl.table_to_pd(
        jl.MixedModels.coeftable(getattr(jl, model_res_name))
    )
    # convert coef by link dist
    mm_coeftable['link_dist'] = str(link_dist)
    mm_coeftable = convert_coef_by_link_dist(mm_coeftable, link_dist)
    # Add confidence intervals
    mm_dof = jl.MixedModels.dof(getattr(jl, model_res_name))
    mm_coeftable = add_CIs_to_coef_df(mm_coeftable, mm_dof)
    # Add formula
    mm_coeftable["formula"] = formula
    return mm_coeftable

def fit_linear_model(df: pd.DataFrame, predict_col: str, formula: str, silent: bool = False, needed_cols='all') -> pd.DataFrame:
    """
    Fit a mixed effects model to the given data.
    """
    df = _validate_nulls(df, needed_cols)
    _pass_df_to_julia(df)
    _pass_convert_df_to_julia()
    
    # Create formula in Julia
    jl.seval(f"j_formula = @formula({formula})")
    # Fit the model
    if not silent:
        logger.info("Fitting model...")
    jl.seval("model_res = lm(j_formula, j_df)")
    model_res_name = "model_res"
    # Extract coefficient table and degrees of freedom
    mm_coeftable = jl.table_to_pd(
        jl.GLM.coeftable(getattr(jl, model_res_name))
    )
    # Add confidence intervals
    mm_dof = jl.GLM.dof(getattr(jl, model_res_name))
    mm_coeftable = add_CIs_to_coef_df(mm_coeftable, mm_dof)
    
    # Retrieve predictions from Julia => pass them back to Python
    #    so we can compute correlation
    preds_jl = jl.seval("predict(model_res, j_df)")  # Julia Vector of predictions
    # np.fromiter converts element-wise; np.array on the Julia wrapper hits the
    # same __array__ copy=None incompatibility as table_to_pd
    fitted_y = np.fromiter(preds_jl, dtype=float, count=len(preds_jl))

    # Compute Pearson r between the actual outcome & fitted predictions
    actual_y = df[predict_col].values.astype(float).ravel()
    r_val, p_val = scipy.stats.pearsonr(actual_y, fitted_y)

    mm_coeftable["Pearson_r"] = r_val
    mm_coeftable["Pearson_pval"] = p_val
    
    # Add formula
    mm_coeftable["formula"] = formula
    return mm_coeftable


#########
# Correlation comparison
#########

def init_cocor():
    # if needed - run: pip install julia rpy2
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector
    
    # Load the cocor package
    cocor = rpackages.importr('cocor')
    return cocor

def cocor_test(cocor, corr_j_h, corr_j_k, corr_h_k, n):
    """
    Compare two correlations using the cocor package in R via RCall in Julia.
    Args:
        group_1 (pd.Series): First group of correlation values.
        group_2 (pd.Series): Second group of correlation values.
    Returns:
        tuple: p-value and test statistic from the cocor test.
    """
    
    # rpy2 will now know how to convert the NumPy floats
    result = cocor.cocor_dep_groups_overlap(
        corr_j_h,
        corr_j_k,
        corr_h_k,
        n=n,
        alternative="two.sided",
        test="all"
    )
    
    # Extract the Steiger test result, as 'all' returns multiple tests
    steiger = result.slots['steiger1980']
    
    # distribution = steiger.rx2('distribution')
    statistic = steiger.rx2('statistic')
    p_value = steiger.rx2('p.value')
    
    statistic_py = statistic[0]
    p_value_py = p_value[0]
    
    return p_value_py, statistic_py


# -------------------------------------------
# if we need r vectors to pass to cocor_test
# -------------------------------------------

# import rpy2.robjects as ro
# Convert pandas Series to R vectors
# r_group_1 = ro.FloatVector(corr_j_h.dropna().tolist())
# r_group_2 = ro.FloatVector(corr_j_k.dropna().tolist())
# r_corr_h_k = ro.FloatVector(corr_h_k.dropna().tolist())

########
# Test cocor_test
########

def run_cocor_example():
    # Example data
    # vector of readability scores using random values using normal distribution
    readability_scores = pd.Series(np.random.normal(loc=0.3, scale=0.6, size=100))
    # vector of RT L1
    rt_L1 = pd.Series(np.random.normal(loc=0.5, scale=0.1, size=100))
    # vector of RT L2
    rt_L2 = pd.Series(np.random.normal(loc=0.4, scale=0.1, size=100))
    # calc correlation between L1 and L2
    corr_RT = float(rt_L1.corr(rt_L2))
    # calc correlation between readability and L1
    corr_text_L1 = float(readability_scores.corr(rt_L1))
    # calc correlation between readability and L2
    corr_text_L2 = float(readability_scores.corr(rt_L2))
    # n
    n = len(readability_scores)

    cocor = init_cocor()
    # Run the cocor test
    p_val, test_stat = cocor_test(cocor, corr_text_L1, corr_text_L2, corr_RT, n)
    print(f"Cocor test statistic: {test_stat}, p-value: {p_val}")
    logger.info("----- Success Cocor -----")

def run_fit_mixed_effects_model_example():
    # Example data
    df = pd.DataFrame({
        'outcome': np.random.normal(size=100),
        'predictor': np.random.normal(size=100),
        'subject_id': np.random.choice(['subj1', 'subj2', 'subj3', 'subj4', 'subj5'], size=100)
    })
    formula = "outcome ~ predictor + (1|subject_id)"
    
    coef_df = fit_mixed_effects_model(df, predict_col='outcome', formula=formula)
    print(coef_df)
    print(coef_df.columns)
    logger.info("----- Success Mixed Effects Model -----")
    
def run_fit_linear_model_example():
    # Example data
    df = pd.DataFrame({
        'outcome': np.random.normal(size=100),
        'predictor': np.random.normal(size=100)
    })
    formula = "outcome ~ predictor"
    
    coef_df = fit_linear_model(df, predict_col='outcome', formula=formula)
    print(coef_df)
    print(coef_df.columns)
    logger.info("----- Success Linear Model ----- ")

if __name__ == "__main__":
    # setup_julia(needs_cocor=True)
    # run_cocor_example()
    
    setup_julia(needs_cocor=False)
    run_fit_mixed_effects_model_example()
    run_fit_linear_model_example()
