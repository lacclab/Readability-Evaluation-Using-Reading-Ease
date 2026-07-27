"""
Extract per-participant per-item eye-tracking measures (TF, RR, SR)
for inter-rater agreement analysis.

Each text_id + level combination is treated as a unique item
(i.e., advanced and elementary versions are separate items).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
from itertools import combinations
from scipy.stats import kendalltau
from scipy.stats import chi2

DATA_DIR = Path(__file__).resolve().parents[3] / "Eye_metrics" / "data"


# currently doesnt actually handle repreated reading, but the structure is there to add it in the future if needed

def load_agreement_data(l1_or_l2: Literal["L1", "L2", "L1_and_L2"],
                        text_level:Literal["paragraph", "article", "sentence"],
                        preview: Literal["Gathering", "Hunting"],
                        repeated_Reading:bool=False,
                        eye_cols = ["TF", "RR", "SR", "FD", "FF", "GD", "NF", ],
                        invert_measures: list[str] = None) -> pd.DataFrame:
    """Load and merge TF, RR, SR at the participant x item level.

    Returns a DataFrame with columns:
        subject_id, text_id, level, batch, item_id,
        mean_TF, RegRateTotal, SkipRateTotal

    item_id = text_id + "_" + level  (treating each version as a separate item)
    """
    r_string = "0" if not repeated_Reading else "1"
    condition_string = f"{preview}{r_string}"
    data_dir = DATA_DIR / l1_or_l2 / condition_string / "metric_tables"
    merge_cols = ["subject_id", "text_id", "level", "batch"]
    if text_level == "sentence":
        merge_cols.append("align_idx")

    df = None

    for col in eye_cols:
        new_df = pd.read_csv(data_dir / f"{text_level}_{col}_agg_by_subject_df.csv")
        # print(f"✓ Loaded: {data_dir / f'{text_level}_{col}_agg_by_subject_df.csv'}")
        if col == "RR":
            value_col = "RegRateTotal"
        elif col == "SR":
            value_col = "SkipRateTotal"
        else:
            value_col = f"mean_{col}"
        new_df = new_df[merge_cols + [value_col]]

        if df is not None:
            df = df.merge(new_df, on=merge_cols)
        else:
            df = new_df
        

    # Invert specified measures (e.g., SR where high = easy, so 1 - SR makes easy = low rank)
    if invert_measures:
        for col in invert_measures:
            if col == "SR":
                df["SkipRateTotal"] = 1 - df["SkipRateTotal"]
            elif col == "RR":
                df["RegRateTotal"] = 1 - df["RegRateTotal"]
            elif col == "TF":
                df["mean_TF"] = -df["mean_TF"]
            elif col in ["FD", "FF", "GD", "NF"]:
                df[f"mean_{col}"] = -df[f"mean_{col}"]

    # Create a unique item identifier that distinguishes Adv vs Ele versions
    df["item_id"] = df["text_id"] + "_" + df["level"]
    if text_level == "sentence":
        df["item_id"] += "_" + df["align_idx"].astype(str)

    return df

def filter_data(df: pd.DataFrame, batch: int = None, version_0:bool=True) -> pd.DataFrame:
    """Filter the agreement data by batch."""
    if batch is not None:
        df = df[df["batch"] == batch]
    if version_0:
        df = df[df["subject_id"].apply(lambda x: int(x.split("_")[0][1:])) % 2 == 0]
    else:
        df = df[df["subject_id"].apply(lambda x: int(x.split("_")[0][1:])) % 2 == 1]

    # filter out participants with fewer then 54 items:
    participant_counts = df[['subject_id', "text_id"]].drop_duplicates()["subject_id"].value_counts()
    valid_participants = participant_counts[participant_counts == 54].index
    df = df[df['subject_id'].isin(valid_participants)]
    return df


def agreement_summary_table() -> pd.DataFrame:
    """Compute mean agreement (Kendall's tau) across all combinations of:
    - l1_or_l2: L1, L2
    - batch: 1, 2, 3
    - text_level: paragraph, article, sentence
    - preview: Gathering, Hunting (not RepeatedReading)
    - measure: TF, RR, SR (SR inverted so easy texts rank first)

    Returns a DataFrame with columns:
        l1_or_l2, batch, text_level, preview, measure, mean_tau, sd_tau, n_pairs, mean_n_shared
    """
    results = []

    for l1_or_l2 in ["L1", "L2"]:
        for batch in [1, 2, 3]:
            for text_level in ["paragraph", "sentence"]:
                for preview in ["Gathering", "Hunting"]:
                    for measure in ["TF", "RR", "SR"]:
                        for version_0 in [True, False]:
                            try:
                                # Load data with SR inverted
                                df = load_agreement_data(
                                    l1_or_l2=l1_or_l2,
                                    text_level=text_level,
                                    preview=preview,
                                    repeated_Reading=False,
                                    eye_cols=[measure],
                                    invert_measures=["SR"] if measure == "SR" else None,
                                )

                                # Filter to batch
                                df = filter_data(df, batch=batch, version_0=version_0)

                                if df.empty:
                                    continue

                                # Determine which column to use for tau calculation
                                if measure == "TF":
                                    measure_col = "mean_TF"
                                elif measure == "RR":
                                    measure_col = "RegRateTotal"
                                elif measure == "SR":
                                    measure_col = "SkipRateTotal"

                                # Compute pairwise tau
                                tau_df = pairwise_kendall_tau(df, measure_col)

                                if len(tau_df) > 0:
                                    mean_tau = tau_df["tau"].mean()
                                    sd_tau = tau_df["tau"].std()
                                    n_pairs = len(tau_df)
                                    mean_n_shared = tau_df["n_shared"].mean()
                                    results.append({
                                        "l1_or_l2": l1_or_l2,
                                        "batch": batch,
                                        "text_level": text_level,
                                        "preview": preview,
                                        "version_0":version_0,
                                        "measure": measure,
                                        "mean_tau": mean_tau,
                                        "sd_tau": sd_tau,
                                        "n_pairs": n_pairs,
                                        "mean_n_shared": mean_n_shared,
                                    })
                            except Exception as e:
                                continue

    return pd.DataFrame(results)


def export_participant_items(
    l1_or_l2: str = "L1",
    text_level: str = "paragraph",
    preview: str = "Gathering",
    output_path: str = "participant_items.csv"
) -> None:
    """Export a CSV listing all items read by each participant.

    Parameters
    ----------
    l1_or_l2 : str
        'L1', 'L2', or 'L1_and_L2'
    text_level : str
        'paragraph', 'article', or 'sentence'
    preview : str
        'Gathering' or 'Hunting'
    output_path : str
        Path to save the CSV file
    """
    df = load_agreement_data(
        l1_or_l2=l1_or_l2,
        text_level=text_level,
        preview=preview,
        eye_cols=["TF"]
    )

    # Create one row per participant with all their items
    rows = []
    for subject in sorted(df['subject_id'].unique()):
        subj_data = df[df['subject_id'] == subject].sort_values('item_id')
        batch_val = subj_data['batch'].iloc[0]  # batch is same for all rows of a subject
        item_ids = subj_data['item_id'].tolist()
        rows.append({
            'subject_id': subject,
            'batch': batch_val,
            'n_items': len(item_ids),
            'items': '|'.join(item_ids)
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"✓ Exported to: {output_path}")
    print(f"  {result_df['subject_id'].nunique()} participants")
    print(f"  Batches: {sorted(result_df['batch'].unique())}")
    print(f"\nFirst 3 rows:")
    print(result_df.head(3).to_string(index=False))




def pairwise_kendall_tau(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Compute pairwise Kendall's tau between all participant pairs on shared items.

    For each pair (p, q), restricts to the overlapping item set O_pq = T_p ∩ T_q,
    then computes Kendall's tau on the shared rankings.

    Kendall (1938) — "A New Measure of Rank Correlation":
        tau = (C - D) / (n choose 2)
    where C = concordant pairs, D = discordant pairs, n = |O_pq|.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'subject_id', 'item_id', and the measure column.
    measure : str
        Column name to rank on (e.g. 'mean_TF', 'RegRateTotal', 'SkipRateTotal').

    Returns
    -------
    DataFrame with columns:
        subject_p, subject_q, n_shared, tau, p_value
    """
    # Pivot to subject x item matrix (NaN where subject didn't read item)
    pivot = df.pivot_table(index="subject_id", columns="item_id", values=measure)
    subjects = pivot.index.tolist()

    results = []
    # print(list(combinations(subjects, 2)))
    # for i, p in enumerate(subjects):
    #     for j, q in enumerate(subjects[:i]):
            
    for p, q in combinations(subjects, 2):
        scores_p = pivot.loc[p]
        # print(scores_p)
        scores_q = pivot.loc[q]
        # Shared items: both have non-NaN values
        shared_mask = scores_p.notna() & scores_q.notna()
        n_shared = shared_mask.sum()
        if n_shared < 2:
            continue
        tau, p_value = kendalltau(scores_p[shared_mask], scores_q[shared_mask])
        results.append({
            "subject_p": p,
            "subject_q": q,
            "n_shared": n_shared,
            "tau": tau,
            "p_value": p_value,
        })

    return pd.DataFrame(results)


def kendalls_w_from_judgements(df: pd.DataFrame) -> tuple[float, float]:
    """Calculate Kendall's W agreement from a DataFrame with judgement columns.

    Designed for datasets like complexity_ds_en where each row is an item (sentence)
    and columns are individual rater judgments (judgement1, judgement2, etc).

    Parameters
    ----------
    df : DataFrame
        Must have columns named 'judgement1', 'judgement2', etc.
        Each row is an item, each column is a rater's judgment.

    Returns
    -------
    tuple[float, float]
        (kendalls_w, p_value) - Kendall's W statistic and chi-square p-value

    Notes
    -----
    Kendall's W ranges from 0 (no agreement) to 1 (perfect agreement).
    Formula: W = 12*S / (k^2 * (n^3 - n))
    where S = sum of squared deviations, k = number of raters, n = number of items
    """
    # Extract the judgement columns
    judgement_cols = [col for col in df.columns if col.startswith("judgement")]
    if not judgement_cols:
        raise ValueError("No judgement columns found. Columns should start with 'judgement'.")

    # Create matrix: rows = raters, columns = items
    ratings_matrix = df[judgement_cols].values.T
    n_raters, n_items = ratings_matrix.shape

    if n_raters < 2 or n_items < 2:
        return np.nan, np.nan

    # Rank each rater's scores (0-based ranking, but W is scale-invariant)
    ranked = np.argsort(np.argsort(ratings_matrix, axis=1), axis=1)

    # Sum of ranks for each item
    rank_sums = ranked.sum(axis=0)
    mean_rank_sum = rank_sums.mean()

    # Sum of squared deviations
    S = ((rank_sums - mean_rank_sum) ** 2).sum()

    # Kendall's W
    W = 12 * S / (n_raters ** 2 * (n_items ** 3 - n_items))

    # Chi-square test for significance
    chi2_stat = n_raters * (n_items - 1) * W
    p_value = 1 - chi2.cdf(chi2_stat, df=n_items - 1)

    return W, p_value


def kendalls_w(df: pd.DataFrame, measure: str) -> tuple[float, float]:
    """Calculate Kendall's W (concordance coefficient) for agreement among multiple raters.

    Kendall's W ranges from 0 (no agreement) to 1 (perfect agreement).
    Formula: W = 12*S / (k^2 * (n^3 - n))
    where S = sum of squared deviations, k = number of raters, n = number of items

    Parameters
    ----------
    df : DataFrame
        Must contain 'subject_id', 'item_id', and measure column
    measure : str
        Column name to rank on

    Returns
    -------
    (kendalls_w_value, p_value)
    """

    # Pivot to rater x item matrix
    pivot = df.pivot_table(index="subject_id", columns="item_id", values=measure)
    pivot = pivot.dropna(axis=1, how='any')  # Keep only items all raters evaluated

    if len(pivot) < 2 or len(pivot.columns) < 2:
        return np.nan, np.nan

    k = len(pivot)  # number of raters
    n = len(pivot.columns)  # number of items

    # Rank each rater's scores
    ranked = pivot.rank(axis=1)

    # Sum of ranks for each item
    rank_sums = ranked.sum(axis=0)

    # Mean rank sum
    mean_rank_sum = rank_sums.mean()

    # Sum of squared deviations
    S = ((rank_sums - mean_rank_sum) ** 2).sum()

    # Kendall's W
    W = 12 * S / (k ** 2 * (n ** 3 - n))

    # Chi-square test for significance
    chi2_stat = k * (n - 1) * W
    p_value = 1 - chi2.cdf(chi2_stat, df=n - 1)

    return W, p_value




def average_pairwise_spearmans_rho(df: pd.DataFrame, measure: str) -> float:
    """Calculate average Spearman's rho across all participant pairs on shared items.

    Parameters
    ----------
    df : DataFrame
        Must contain 'subject_id', 'item_id', and measure column
    measure : str
        Column name to correlate on

    Returns
    -------
    float
        Average Spearman's rho across all pairs
    """
    from scipy.stats import spearmanr

    pivot = df.pivot_table(index="subject_id", columns="item_id", values=measure)
    subjects = pivot.index.tolist()

    correlations = []
    for p, q in combinations(subjects, 2):
        scores_p = pivot.loc[p]
        scores_q = pivot.loc[q]
        shared_mask = scores_p.notna() & scores_q.notna()
        n_shared = shared_mask.sum()
        if n_shared < 2:
            continue
        rho, _ = spearmanr(scores_p[shared_mask], scores_q[shared_mask])
        correlations.append(rho)

    return np.nanmean(correlations) if correlations else np.nan


def average_pairwise_kendalls_tau(df: pd.DataFrame, measure: str) -> float:
    """Calculate average Kendall's tau across all participant pairs on shared items.

    Parameters
    ----------
    df : DataFrame
        Must contain 'subject_id', 'item_id', and measure column
    measure : str
        Column name to correlate on

    Returns
    -------
    float
        Average Kendall's tau across all pairs
    """
    pivot = df.pivot_table(index="subject_id", columns="item_id", values=measure)
    subjects = pivot.index.tolist()

    correlations = []
    for p, q in combinations(subjects, 2):
        scores_p = pivot.loc[p]
        scores_q = pivot.loc[q]
        shared_mask = scores_p.notna() & scores_q.notna()
        n_shared = shared_mask.sum()
        if n_shared < 2:
            continue
        tau, _ = kendalltau(scores_p[shared_mask], scores_q[shared_mask])
        correlations.append(tau)

    return np.nanmean(correlations) if correlations else np.nan


def average_pairwise_pearson_r(df: pd.DataFrame, measure: str) -> float:
    """Calculate average Pearson R correlation across all participant pairs on shared items.

    Parameters
    ----------
    df : DataFrame
        Must contain 'subject_id', 'item_id', and measure column
    measure : str
        Column name to correlate on

    Returns
    -------
    float
        Average Pearson R correlation across all pairs
    """
    from scipy.stats import pearsonr

    pivot = df.pivot_table(index="subject_id", columns="item_id", values=measure)
    subjects = pivot.index.tolist()

    correlations = []
    for p, q in combinations(subjects, 2):
        scores_p = pivot.loc[p]
        scores_q = pivot.loc[q]
        shared_mask = scores_p.notna() & scores_q.notna()
        n_shared = shared_mask.sum()
        if n_shared < 2:
            continue
        r, _ = pearsonr(scores_p[shared_mask], scores_q[shared_mask])
        correlations.append(r)

    return np.nanmean(correlations) if correlations else np.nan


def overall_agreement_summary_table() -> pd.DataFrame:
    """Compute overall agreement metrics (Kendall's W, and pairwise metrics) across all condition combinations.

    Returns a DataFrame with columns:
        l1_or_l2, batch, text_level, preview, version, measure,
        kendalls_w, kendalls_w_pvalue, avg_spearmans_rho, avg_kendalls_tau, avg_pearson_r

    Includes both versions (V0: even IDs, V1: odd IDs) for all conditions.
    """
    results = []

    for l1_or_l2 in ["L1", "L2", "L1_and_L2"]:
        for batch in [1, 2, 3]:
            for text_level in ["paragraph", "sentence"]:
                for preview in ["Gathering", "Hunting"]:
                    for measure in ["TF", "RR", "SR"]:
                        for version_0 in [True, False]:
                            try:
                                # Load data with SR inverted
                                df = load_agreement_data(
                                    l1_or_l2=l1_or_l2,
                                    text_level=text_level,
                                    preview=preview,
                                    repeated_Reading=False,
                                    eye_cols=[measure],
                                    invert_measures=["SR"] if measure == "SR" else None,
                                )

                                # Filter to batch and version
                                df = filter_data(df, batch=batch, version_0=version_0)

                                if df.empty or df['subject_id'].nunique() < 2:
                                    continue

                                # Determine which column to use
                                if measure == "TF":
                                    measure_col = "mean_TF"
                                elif measure == "RR":
                                    measure_col = "RegRateTotal"
                                elif measure == "SR":
                                    measure_col = "SkipRateTotal"

                                # Calculate metrics
                                kendalls_w_val, kendalls_w_pval = kendalls_w(df, measure_col)
                                avg_spearmans_rho = average_pairwise_spearmans_rho(df, measure_col)
                                avg_kendalls_tau = average_pairwise_kendalls_tau(df, measure_col)
                                avg_pearson_r = average_pairwise_pearson_r(df, measure_col)

                                version_label = "V0" if version_0 else "V1"
                                results.append({
                                    "l1_or_l2": l1_or_l2,
                                    "batch": batch,
                                    "text_level": text_level,
                                    "preview": preview,
                                    "version": version_label,
                                    "measure": measure,
                                    "n_raters": df['subject_id'].nunique(),
                                    "n_items": df['item_id'].nunique(),
                                    "kendalls_w": kendalls_w_val,
                                    "kendalls_w_pvalue": kendalls_w_pval,
                                    "avg_spearmans_rho": avg_spearmans_rho,
                                    "avg_kendalls_tau": avg_kendalls_tau,
                                    "avg_pearson_r": avg_pearson_r,
                                })
                            except Exception as e:
                                print("Error processing combination:", l1_or_l2, batch, text_level, preview, measure, version_label, e)
                                continue

    return pd.DataFrame(results)


def load_agreement_data_from_path(text_level: Literal["paragraph", "article", "sentence"],
                                   data_subdir: str = "FirstReading",
                                   eye_cols = ["TF", "RR", "SR", "FD", "FF", "GD", "NF"],
                                   invert_measures: list[str] = None) -> pd.DataFrame:
    """Load and merge eye-tracking measures from a specific data directory.

    Parameters
    ----------
    text_level : str
        'paragraph', 'article', or 'sentence'
    data_subdir : str
        Subdirectory name (e.g., 'FirstReading', 'Gathering0', 'Hunting0')
    eye_cols : list
        Eye tracking measures to load
    invert_measures : list
        Measures to invert (e.g., ['SR'] for difficulty)

    Returns
    -------
    DataFrame with columns:
        subject_id, text_id, level, batch, item_id,
        mean_TF, RegRateTotal, SkipRateTotal, etc.
    """
    data_dir = DATA_DIR / "L1_and_L2" / data_subdir / "metric_tables"
    merge_cols = ["subject_id", "text_id", "level", "batch"]
    if text_level == "sentence":
        merge_cols.append("align_idx")

    df = None

    for col in eye_cols:
        new_df = pd.read_csv(data_dir / f"{text_level}_{col}_agg_by_subject_df.csv")
        if col == "RR":
            value_col = "RegRateTotal"
        elif col == "SR":
            value_col = "SkipRateTotal"
        else:
            value_col = f"mean_{col}"
        new_df = new_df[merge_cols + [value_col]]

        if df is not None:
            df = df.merge(new_df, on=merge_cols)
        else:
            df = new_df

    # Invert specified measures
    if invert_measures:
        for col in invert_measures:
            if col == "SR":
                df["SkipRateTotal"] = 1 - df["SkipRateTotal"]
            elif col == "RR":
                df["RegRateTotal"] = 1 - df["RegRateTotal"]
            elif col == "TF":
                df["mean_TF"] = -df["mean_TF"]
            elif col in ["FD", "FF", "GD", "NF"]:
                df[f"mean_{col}"] = -df[f"mean_{col}"]

    # Create item identifier
    df["item_id"] = df["text_id"] + "_" + df["level"]
    if text_level == "sentence":
        df["item_id"] += "_" + df["align_idx"].astype(str)

    return df


def average_across_versions(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Average agreement metrics across both versions (V0 and V1).

    Parameters
    ----------
    summary_df : DataFrame
        Must contain 'version' column and metric columns

    Returns
    -------
    DataFrame with averaged metrics (version column removed)
    """
    groupby_cols = ["text_level", "measure"]
    return summary_df.groupby(groupby_cols, as_index=False).mean(numeric_only=True)


def overall_agreement_summary_table_all_participants() -> pd.DataFrame:
    """Compute overall agreement metrics across all participants from FirstReading data.

    Returns a DataFrame with columns:
        batch, text_level, version, measure,
        kendalls_w, kendalls_w_pvalue, avg_spearmans_rho, avg_kendalls_tau, avg_pearson_r

    Loads all participants together from the FirstReading directory (no preview division).
    Includes both versions (V0: even IDs, V1: odd IDs) for all conditions.
    """
    results = []

    for batch in [1, 2, 3]:
        for text_level in ["paragraph", "sentence"]:
            for measure in ["TF", "RR", "SR"]:
                for version_0 in [True, False]:
                    try:
                        # Load data from FirstReading directory
                        df = load_agreement_data_from_path(
                            text_level=text_level,
                            data_subdir="FirstReading",
                            eye_cols=[measure],
                            invert_measures=["SR"] if measure == "SR" else None,
                        )

                        # Filter to batch and version
                        df = filter_data(df, batch=batch, version_0=version_0)

                        if df.empty or df['subject_id'].nunique() < 2:
                            continue

                        # Determine which column to use
                        if measure == "TF":
                            measure_col = "mean_TF"
                        elif measure == "RR":
                            measure_col = "RegRateTotal"
                        elif measure == "SR":
                            measure_col = "SkipRateTotal"

                        # Calculate metrics
                        kendalls_w_val, kendalls_w_pval = kendalls_w(df, measure_col)
                        avg_spearmans_rho = average_pairwise_spearmans_rho(df, measure_col)
                        avg_kendalls_tau = average_pairwise_kendalls_tau(df, measure_col)
                        avg_pearson_r = average_pairwise_pearson_r(df, measure_col)

                        version_label = "V0" if version_0 else "V1"
                        results.append({
                            "batch": batch,
                            "text_level": text_level,
                            "version": version_label,
                            "measure": measure,
                            "n_raters": df['subject_id'].nunique(),
                            "n_items": df['item_id'].nunique(),
                            "kendalls_w": kendalls_w_val,
                            "kendalls_w_pvalue": kendalls_w_pval,
                            "avg_spearmans_rho": avg_spearmans_rho,
                            "avg_kendalls_tau": avg_kendalls_tau,
                            "avg_pearson_r": avg_pearson_r,
                        })
                    except Exception as e:
                        print("Error processing combination:", batch, text_level, measure, version_label, e)
                        continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    export_participant_items(
        l1_or_l2="L1_and_L2",
        text_level="paragraph",
        preview="Gathering",
        output_path=Path(__file__).parent / "participant_items.csv"
    )

    # print("="*80)
    # print("OVERALL AGREEMENT ANALYSIS: Kendall's W and Pairwise Metrics")
    # print("="*80)

    # # Compute overall agreement summary
    # print("\nComputing agreement metrics across all conditions...")
    # print("(2 readers × 3 batches × 2 text levels × 2 previews × 3 measures = 72 combinations)")

    # summary = overall_agreement_summary_table()

    # print(f"\n✓ Computed {len(summary)} condition combinations")
    # print(f"\nOVERALL AGREEMENT SUMMARY")
    # print("="*80)
    # print(summary.to_string(index=False))

    # # Save summary
    # csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    # summary.to_csv(csv_path, index=False)
    # print(f"\n→ Saved to: {csv_path}")

    # # Show summary statistics
    # print("\n" + "="*80)
    # print("SUMMARY STATISTICS")
    # print("="*80)
    # print(f"\nKendall's W (concordance):")
    # print(f"  Mean: {summary['kendalls_w'].mean():.4f}")
    # print(f"  Median: {summary['kendalls_w'].median():.4f}")
    # print(f"  Std: {summary['kendalls_w'].std():.4f}")
    # print(f"  Range: [{summary['kendalls_w'].min():.4f}, {summary['kendalls_w'].max():.4f}]")

    # print(f"\nAverage Spearman's Rho:")
    # print(f"  Mean: {summary['avg_spearmans_rho'].mean():.4f}")
    # print(f"  Median: {summary['avg_spearmans_rho'].median():.4f}")
    # print(f"  Std: {summary['avg_spearmans_rho'].std():.4f}")
    # print(f"  Range: [{summary['avg_spearmans_rho'].min():.4f}, {summary['avg_spearmans_rho'].max():.4f}]")

    # print(f"\nAverage Kendall's Tau:")
    # print(f"  Mean: {summary['avg_kendalls_tau'].mean():.4f}")
    # print(f"  Median: {summary['avg_kendalls_tau'].median():.4f}")
    # print(f"  Std: {summary['avg_kendalls_tau'].std():.4f}")
    # print(f"  Range: [{summary['avg_kendalls_tau'].min():.4f}, {summary['avg_kendalls_tau'].max():.4f}]")

    # print(f"\nAverage Pearson R:")
    # print(f"  Mean: {summary['avg_pearson_r'].mean():.4f}")
    # print(f"  Median: {summary['avg_pearson_r'].median():.4f}")
    # print(f"  Std: {summary['avg_pearson_r'].std():.4f}")
    # print(f"  Range: [{summary['avg_pearson_r'].min():.4f}, {summary['avg_pearson_r'].max():.4f}]")

    # # Analyze agreement on complexity_ds_en
    # print("\n" + "="*80)
    # print("COMPLEXITY DATASET (complexity_ds_en.csv) AGREEMENT ANALYSIS")
    # print("="*80)

    # complexity_path = Path(__file__).parent / "data" / "complexity_ds_en.csv"
    # if complexity_path.exists():
    #     complexity_df = pd.read_csv(complexity_path)
    #     W, p_value = kendalls_w_from_judgements(complexity_df)

    #     judgement_cols = [col for col in complexity_df.columns if col.startswith('judgement')]
    #     print(f"\nDataset: {complexity_path.name}")
    #     print(f"  Items (sentences): {len(complexity_df)}")
    #     print(f"  Raters: {len(judgement_cols)}")
    #     print(f"  Rating range: {complexity_df[judgement_cols].values.min():.0f}-{complexity_df[judgement_cols].values.max():.0f}")

    #     print(f"\nResults:")
    #     print(f"  Kendall's W: {W:.6f}")
    #     print(f"  P-value: {p_value:.2e}")

    #     # Interpret agreement level
    #     if W < 0.1:
    #         agreement_level = "Negligible/Poor"
    #     elif W < 0.3:
    #         agreement_level = "Fair"
    #     elif W < 0.5:
    #         agreement_level = "Moderate"
    #     elif W < 0.7:
    #         agreement_level = "Good"
    #     else:
    #         agreement_level = "Very Good/Excellent"

    #     print(f"\nInterpretation:")
    #     print(f"  Agreement Level: {agreement_level}")
    #     if p_value < 0.001:
    #         print(f"  Significance: Highly significant (p < 0.001)")
    #     elif p_value < 0.05:
    #         print(f"  Significance: Significant (p < 0.05)")
    #     else:
    #         print(f"  Significance: Not significant (p ≥ 0.05)")
    # else:
    #     print(f"\n⚠ Could not find: {complexity_path}")

    # # Compute agreement across all participants from FirstReading data
    # print("\n" + "="*80)
    # print("OVERALL AGREEMENT ANALYSIS: All Participants (FirstReading)")
    # print("="*80)
    # print("\nComputing agreement metrics across all conditions...")
    # summary_all = overall_agreement_summary_table_all_participants()
    # print(f"\n✓ Computed {len(summary_all)} condition combinations")
    # print(f"\nAGREEMENT SUMMARY (All Participants)")
    # print("="*80)
    # print(summary_all.to_string(index=False))

    # # Save summary with versions
    csv_path = Path(__file__).parent / "overall_agreement_summary_all_participants.csv"
    # summary_all.to_csv(csv_path, index=False)
    # print(f"\n→ Saved to: {csv_path}")

    # LOAD SUMMARY (to demonstrate loading and averaging)
    summary_all = pd.read_csv(csv_path)
    print(f"\n✓ Loaded summary from: {csv_path}")

    # Average across versions
    summary_averaged = average_across_versions(summary_all)
    print(f"\n" + "="*80)
    print("AGREEMENT SUMMARY (Averaged across versions)")
    print("="*80)
    print(summary_averaged.to_string(index=False))

    # Save averaged summary
    csv_path_avg = Path(__file__).parent / "overall_agreement_summary_all_participants_averaged.csv"
    summary_averaged.to_csv(csv_path_avg, index=False)
    print(f"\n→ Saved to: {csv_path_avg}")
