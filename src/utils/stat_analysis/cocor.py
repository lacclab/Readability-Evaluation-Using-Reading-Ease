from loguru import logger
import pandas as pd
import numpy as np
import scipy.stats
import time

import juliapkg
juliapkg.require_julia("=1.12.4")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401

# pip install julia rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

###############################################################################
# Julia Setup
###############################################################################
def setup_julia():
    """
    Install and import the needed Julia packages.
    """
    print(jl.seval("VERSION"))
    jl.seval("import Pkg")
    jl.seval('Pkg.add("RCall")')  # for R packages
    jl.seval("using RCall")
    
    
    # # Check if 'cocor' R package is installed
    # utils = rpackages.importr('utils')
    # if not rpackages.isinstalled("cocor"):
    #     print("R package 'cocor' not found. Installing now...")
    #     utils.chooseCRANmirror(ind=1) # Select a CRAN mirror
    #     utils.install_packages(StrVector(['cocor']))
    #     print("R package 'cocor' has been installed.")
    # else:
    #     print("R package 'cocor' is already installed.")
    
    logger.info("Julia environment is set up with MixedModels and DataFrames.")

#########
# Correlation comparison
#########

def init_cocor():
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

########
# Test cocor_test
########

if __name__ == "__main__":
    setup_julia()
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
    logger.info("----- Success -----")
    
    
