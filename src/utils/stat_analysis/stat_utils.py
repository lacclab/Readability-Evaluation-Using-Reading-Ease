import pandas as pd
import numpy as np
import scipy.stats
from typing import Literal


def add_p_val_symbols(df, p_val_col):
    """
    Create a new column in df with significance symbols based on p-values,
    using a vectorized approach for efficiency.

    Rules:
    - '***' for p < 0.001
    - '**'  for p < 0.01
    - '*'   for p < 0.05
    - 'ns'  for 0.05 <= p <= 1
    - None  otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    p_val_col : str
        Column name in df containing the p-values.

    Returns
    -------
    pd.DataFrame
        df with an additional column `p_val_col + "_symbol"`.
    """
    conditions = [
        (df[p_val_col] < 0.001),
        (df[p_val_col] < 0.01),
        (df[p_val_col] < 0.05),
        (df[p_val_col].between(0.05, 1, inclusive='both'))
    ]
    choices = ['***', '**', '*', 'ns']

    new_col_name = f"{p_val_col}_symbol"
    df[new_col_name] = np.select(conditions, choices, default=None)

    return df

def p_to_star(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    elif p_val < 1:
        return 'ns'
    else:
        return None
    
def stars_to_p(stars: str) -> float:
    if stars == '***':
        return 'p < 0.001'
    elif stars == '**':
        return 'p < 0.01'
    elif stars == '*':
        return 'p < 0.05'
    elif stars == 'ns':
        return 'p > 0.05'
    else:
        return np.nan

def get_mean_ci(df: pd.DataFrame, col: str, sem_or_std: Literal['sem', 'std'], confidence: float=0.95) -> pd.DataFrame:
    """
    Compute mean & confidence interval for `col` in DataFrame `df`.
    Add two new columns: 'low_conf' and 'up_conf' containing the lower/upper bound
    of the CI for the overall mean of `col`.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing your data.
    col : str
        Column name whose mean and confidence interval you want to calculate.
    confidence : float, optional
        Confidence level. Default 0.95 => 95% CI.

    Returns
    -------
    pd.DataFrame
        Same DataFrame, but with two new columns, 'low_conf' & 'up_conf',
        each filled with the same CI bounds for the mean of `col`.
    """
    # 1) Extract the data
    data = df[col].dropna().values  # remove NaNs
    
    # 2) Basic stats
    n = len(data)
    if n < 2:
        # Not enough data to compute a meaningful t-based CI
        df["low_conf"] = np.nan
        df["up_conf"]  = np.nan
        return df
    
    mean_val = np.mean(data)
    sem = scipy.stats.sem(data)  # standard error of the mean
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    
    if sem_or_std == 'std':
        margin = t_crit * np.std(data, ddof=1)
    elif sem_or_std == 'sem':
        margin = t_crit * sem
    else:
        raise ValueError("sem_or_std must be 'sem' or 'std'")
    
    low_ci = mean_val - margin
    up_ci  = mean_val + margin
    
    return low_ci, up_ci, margin

def bootstrap_ci(metric_func, y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """
    Computes bootstrap confidence intervals using normal approximation
    and returns the symmetrical confidence interval width.

    Args:
        metric_func (function): Metric function (e.g., accuracy_score).
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        n_bootstrap (int): Number of bootstrap samples.
        alpha (float): Confidence level (default 95%).

    Returns:
        tuple: (mean_metric, ci_radius)
    """
    n = len(y_true)
    boot_metrics = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        boot_y_true = np.array(y_true)[indices]
        boot_y_pred = np.array(y_pred)[indices]
        boot_metrics.append(metric_func(boot_y_true, boot_y_pred))

    mean_metric = np.mean(boot_metrics)
    std_metric = np.std(boot_metrics, ddof=1)  # Sample standard deviation

    # Normal approximation for CI
    z = 1.96  # Approximate 95% confidence level (for normal distribution)
    ci_radius = z * std_metric  # Symmetric confidence interval radius

    return mean_metric, ci_radius

if __name__ == "__main__":
    data = {
        "term": ["Intercept", "X1", "X2", "X3"],
        "p_value": [0.0005, 0.02, 0.40, 2.0],  # 2.0 is outside [0,1] just to demonstrate
    }
    df = pd.DataFrame(data)
    df = add_p_val_symbols(df, "p_value")
    print(df)