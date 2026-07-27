"""
Visualize inter-rater agreement using Kendall's W (from overall_agreement_summary.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

from measure_agreement import load_agreement_data, filter_data

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Create subdirectories for each metric
KENDALLS_W_DIR = OUTPUT_DIR / "kendalls_w"
KENDALLS_W_DIR.mkdir(exist_ok=True)
SPEARMANS_RHO_DIR = OUTPUT_DIR / "spearmans_rho"
SPEARMANS_RHO_DIR.mkdir(exist_ok=True)
KENDALLS_TAU_DIR = OUTPUT_DIR / "kendalls_tau"
KENDALLS_TAU_DIR.mkdir(exist_ok=True)
PEARSONS_R_DIR = OUTPUT_DIR / "pearsons_r"
PEARSONS_R_DIR.mkdir(exist_ok=True)


def plot_l1_vs_l2_kendalls_w() -> None:
    """Compare Kendall's W between L1 and L2."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Pivot for comparison
    l1_data = df[df["l1_or_l2"] == "L1"].set_index(["batch", "preview", "measure"])["kendalls_w"]
    l2_data = df[df["l1_or_l2"] == "L2"].set_index(["batch", "preview", "measure"])["kendalls_w"]

    comparison = pd.DataFrame({
        "L1": l1_data,
        "L2": l2_data,
    }).reset_index()
    comparison["Difference"] = comparison["L1"] - comparison["L2"]

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("L1 vs L2 Agreement (Kendall's W)", fontsize=14, fontweight='bold')

    # Plot 1: Side-by-side bars
    ax = axes[0]
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(x - width/2, comparison["L1"], width, label="L1 Speakers", alpha=0.8)
    ax.bar(x + width/2, comparison["L2"], width, label="L2 Speakers", alpha=0.8)
    ax.set_ylabel("Kendall's W")
    ax.set_xlabel("Condition")
    ax.set_title("Mean Agreement by Reader Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{int(r['batch'])}-{r['preview'][:3]}-{r['measure']}" 
                         for _, r in comparison.iterrows()], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Difference heatmap (L1 - L2)
    ax = axes[1]
    pivot_diff = comparison.pivot_table(
        index=["batch", "preview"],
        columns="measure",
        values="Difference"
    )
    sns.heatmap(
        pivot_diff,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "L1 W - L2 W"},
        ax=ax,
        vmin=-0.1,
        vmax=0.1,
    )
    ax.set_title("Difference (L1 - L2)")
    ax.set_xlabel("Eye-Tracking Measure")
    ax.set_ylabel("Batch & Preview Condition")

    plt.tight_layout()
    output_path = KENDALLS_W_DIR / "l1_vs_l2_kendalls_w.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved L1 vs L2 Kendall's W to: {output_path}")


def plot_kendalls_w_aggregated_by_preview() -> None:
    """Visualize Kendall's W averaged across all batches and both versions, by preview type."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Average across batches and both versions for each (preview, measure, l1_or_l2) combination
    aggregated = df.groupby(["l1_or_l2", "preview", "measure"])["kendalls_w"].mean().reset_index()

    # Create heatmap for L1 and L2 separately
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Kendall's W Concordance (Averaged Across Batches)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = aggregated[aggregated["l1_or_l2"] == reader]

        # Create pivot table for heatmap
        pivot_data = data.pivot_table(
            index="preview",
            columns="measure",
            values="kendalls_w"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Mean Kendall's W"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure (TF, RR, SR)")
        ax.set_ylabel("Preview Condition")

    plt.tight_layout()
    output_path = KENDALLS_W_DIR / "kendalls_w_aggregated_by_preview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved aggregated Kendall's W to: {output_path}")


def plot_l1_vs_l2_aggregated() -> None:
    """Compare Kendall's W between L1 and L2, aggregated across batches."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Aggregate across batches
    l1_data = df[df["l1_or_l2"] == "L1"].groupby(["preview", "measure"])["kendalls_w"].mean()
    l2_data = df[df["l1_or_l2"] == "L2"].groupby(["preview", "measure"])["kendalls_w"].mean()

    comparison = pd.DataFrame({
        "L1": l1_data,
        "L2": l2_data,
    }).reset_index()
    comparison["Difference"] = comparison["L1"] - comparison["L2"]

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("L1 vs L2 Agreement (Kendall's W - Aggregated Across Batches)", fontsize=14, fontweight='bold')

    # Plot 1: Side-by-side bars
    ax = axes[0]
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(x - width/2, comparison["L1"], width, label="L1 Speakers", alpha=0.8)
    ax.bar(x + width/2, comparison["L2"], width, label="L2 Speakers", alpha=0.8)
    ax.set_ylabel("Kendall's W")
    ax.set_xlabel("Condition")
    ax.set_title("Mean Agreement by Reader Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['preview'][:3]}-{r['measure']}"
                         for _, r in comparison.iterrows()], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Difference heatmap (L1 - L2)
    ax = axes[1]
    pivot_diff = comparison.pivot_table(
        index="preview",
        columns="measure",
        values="Difference"
    )
    sns.heatmap(
        pivot_diff,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "L1 W - L2 W"},
        ax=ax,
        vmin=-0.1,
        vmax=0.1,
    )
    ax.set_title("Difference (L1 - L2)")
    ax.set_xlabel("Eye-Tracking Measure")
    ax.set_ylabel("Preview Condition")

    plt.tight_layout()
    output_path = KENDALLS_W_DIR / "l1_vs_l2_kendalls_w_aggregated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved aggregated L1 vs L2 comparison to: {output_path}")


def compute_both_versions_summary() -> pd.DataFrame:
    """Compute agreement metrics for both version_0=True and version_0=False."""
    from measure_agreement import (
        load_agreement_data,
        filter_data,
        kendalls_w,
        average_pairwise_spearmans_rho,
        average_pairwise_kendalls_tau,
        average_pairwise_pearson_r,
    )

    results = []

    for l1_or_l2 in ["L1", "L2"]:
        for batch in [1, 2, 3]:
            for preview in ["Gathering", "Hunting"]:
                for measure in ["TF", "RR", "SR"]:
                    for version_0 in [True, False]:
                        try:
                            df = load_agreement_data(
                                l1_or_l2=l1_or_l2,
                                text_level="paragraph",
                                preview=preview,
                                eye_cols=[measure],
                                invert_measures=["SR"] if measure == "SR" else None,
                            )

                            df = filter_data(df, batch=batch, version_0=version_0)

                            if df.empty or df['subject_id'].nunique() < 2:
                                continue

                            if measure == "TF":
                                measure_col = "mean_TF"
                            elif measure == "RR":
                                measure_col = "RegRateTotal"
                            else:
                                measure_col = "SkipRateTotal"

                            kendalls_w_val, kendalls_w_pval = kendalls_w(df, measure_col)
                            avg_spearmans_rho_val = average_pairwise_spearmans_rho(df, measure_col)
                            avg_kendalls_tau_val = average_pairwise_kendalls_tau(df, measure_col)
                            avg_pearson_r_val = average_pairwise_pearson_r(df, measure_col)

                            version_label = "V0" if version_0 else "V1"
                            results.append({
                                "l1_or_l2": l1_or_l2,
                                "batch": batch,
                                "preview": preview,
                                "measure": measure,
                                "version": version_label,
                                "version_0": version_0,
                                "n_raters": df['subject_id'].nunique(),
                                "kendalls_w": kendalls_w_val,
                                "avg_spearmans_rho": avg_spearmans_rho_val,
                                "avg_kendalls_tau": avg_kendalls_tau_val,
                                "avg_pearson_r": avg_pearson_r_val,
                            })
                        except Exception as e:
                            continue

    return pd.DataFrame(results)


def plot_both_versions_kendalls_w() -> None:
    """Plot Kendall's W comparing both version groups across all conditions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Create figure with subplots (one for L1, one for L2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Kendall's W: Version 0 vs Version 1 (All Conditions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = df[df["l1_or_l2"] == reader]

        # Create pivot showing all conditions with version as separate rows
        pivot_data = data.pivot_table(
            index=["batch", "preview", "version"],
            columns="measure",
            values="kendalls_w"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Kendall's W"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers (V0=even IDs, V1=odd IDs)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview & Version")

    plt.tight_layout()
    output_path = KENDALLS_W_DIR / "kendalls_w_both_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Kendall's W both versions to: {output_path}")


def plot_version_average_kendalls_w() -> None:
    """Plot average Kendall's W across both versions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Average across versions
    averaged = df.groupby(["l1_or_l2", "batch", "preview", "measure"])["kendalls_w"].mean().reset_index()

    # Create heatmap for L1 and L2 separately
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Kendall's W (Averaged Across Both Versions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = averaged[averaged["l1_or_l2"] == reader]

        # Create pivot table for heatmap
        pivot_data = data.pivot_table(
            index=["batch", "preview"],
            columns="measure",
            values="kendalls_w"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Mean Kendall's W"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview")

    plt.tight_layout()
    output_path = KENDALLS_W_DIR / "kendalls_w_averaged_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved averaged Kendall's W to: {output_path}")


def plot_both_versions_spearmans_rho() -> None:
    """Plot Spearman's rho comparing both version groups across all conditions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Create figure with subplots (one for L1, one for L2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Average Pairwise Spearman's Rho: Version 0 vs Version 1 (All Conditions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = df[df["l1_or_l2"] == reader]

        # Create pivot showing all conditions with version as separate rows
        pivot_data = data.pivot_table(
            index=["batch", "preview", "version"],
            columns="measure",
            values="avg_spearmans_rho"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Spearman's Rho"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers (V0=even IDs, V1=odd IDs)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview & Version")

    plt.tight_layout()
    output_path = SPEARMANS_RHO_DIR / "spearmans_rho_both_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Spearman's rho both versions to: {output_path}")


def plot_version_average_spearmans_rho() -> None:
    """Plot average Spearman's rho across both versions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Average across versions
    averaged = df.groupby(["l1_or_l2", "batch", "preview", "measure"])["avg_spearmans_rho"].mean().reset_index()

    # Create heatmap for L1 and L2 separately
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Average Pairwise Spearman's Rho (Averaged Across Both Versions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = averaged[averaged["l1_or_l2"] == reader]

        # Create pivot table for heatmap
        pivot_data = data.pivot_table(
            index=["batch", "preview"],
            columns="measure",
            values="avg_spearmans_rho"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Mean Spearman's Rho"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview")

    plt.tight_layout()
    output_path = SPEARMANS_RHO_DIR / "spearmans_rho_averaged_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved averaged Spearman's rho to: {output_path}")


def plot_both_versions_kendalls_tau() -> None:
    """Plot Kendall's tau comparing both version groups across all conditions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Create figure with subplots (one for L1, one for L2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Average Pairwise Kendall's Tau: Version 0 vs Version 1 (All Conditions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = df[df["l1_or_l2"] == reader]

        # Create pivot showing all conditions with version as separate rows
        pivot_data = data.pivot_table(
            index=["batch", "preview", "version"],
            columns="measure",
            values="avg_kendalls_tau"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Kendall's Tau"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers (V0=even IDs, V1=odd IDs)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview & Version")

    plt.tight_layout()
    output_path = KENDALLS_TAU_DIR / "kendalls_tau_both_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Kendall's tau both versions to: {output_path}")


def plot_version_average_kendalls_tau() -> None:
    """Plot average Kendall's tau across both versions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Average across versions
    averaged = df.groupby(["l1_or_l2", "batch", "preview", "measure"])["avg_kendalls_tau"].mean().reset_index()

    # Create heatmap for L1 and L2 separately
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Average Pairwise Kendall's Tau (Averaged Across Both Versions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = averaged[averaged["l1_or_l2"] == reader]

        # Create pivot table for heatmap
        pivot_data = data.pivot_table(
            index=["batch", "preview"],
            columns="measure",
            values="avg_kendalls_tau"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Mean Kendall's Tau"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview")

    plt.tight_layout()
    output_path = KENDALLS_TAU_DIR / "kendalls_tau_averaged_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved averaged Kendall's tau to: {output_path}")


def plot_both_versions_pearsons_r() -> None:
    """Plot Pearson's R comparing both version groups across all conditions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Create figure with subplots (one for L1, one for L2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Average Pairwise Pearson's R: Version 0 vs Version 1 (All Conditions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = df[df["l1_or_l2"] == reader]

        # Create pivot showing all conditions with version as separate rows
        pivot_data = data.pivot_table(
            index=["batch", "preview", "version"],
            columns="measure",
            values="avg_pearson_r"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Pearson's R"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers (V0=even IDs, V1=odd IDs)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview & Version")

    plt.tight_layout()
    output_path = PEARSONS_R_DIR / "pearsons_r_both_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Pearson's R both versions to: {output_path}")


def plot_version_average_pearsons_r() -> None:
    """Plot average Pearson's R across both versions."""
    df = compute_both_versions_summary()

    if df.empty:
        print("No data for version comparison")
        return

    # Average across versions
    averaged = df.groupby(["l1_or_l2", "batch", "preview", "measure"])["avg_pearson_r"].mean().reset_index()

    # Create heatmap for L1 and L2 separately
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Average Pairwise Pearson's R (Averaged Across Both Versions)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = averaged[averaged["l1_or_l2"] == reader]

        # Create pivot table for heatmap
        pivot_data = data.pivot_table(
            index=["batch", "preview"],
            columns="measure",
            values="avg_pearson_r"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Mean Pearson's R"},
            ax=ax,
            vmin=0,
            vmax=0.4,
        )
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Batch & Preview")

    plt.tight_layout()
    output_path = PEARSONS_R_DIR / "pearsons_r_averaged_versions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved averaged Pearson's R to: {output_path}")


def plot_spearmans_rho_aggregated() -> None:
    """Plot Spearman's rho averaged across batches and both versions."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Average across both batches and versions
    aggregated = df.groupby(["l1_or_l2", "preview", "measure"])["avg_spearmans_rho"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Spearman's Rho (Aggregated Across Batches)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = aggregated[aggregated["l1_or_l2"] == reader]
        pivot_data = data.pivot_table(index="preview", columns="measure", values="avg_spearmans_rho")

        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0, vmax=0.4,
                    cbar_kws={"label": "Mean Spearman's Rho"})
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Preview")

    plt.tight_layout()
    output_path = SPEARMANS_RHO_DIR / "spearmans_rho_aggregated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved aggregated Spearman's rho to: {output_path}")


def plot_kendalls_tau_aggregated() -> None:
    """Plot Kendall's tau averaged across batches and both versions."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Average across both batches and versions
    aggregated = df.groupby(["l1_or_l2", "preview", "measure"])["avg_kendalls_tau"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Kendall's Tau (Aggregated Across Batches)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = aggregated[aggregated["l1_or_l2"] == reader]
        pivot_data = data.pivot_table(index="preview", columns="measure", values="avg_kendalls_tau")

        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0, vmax=0.4,
                    cbar_kws={"label": "Mean Kendall's Tau"})
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Preview")

    plt.tight_layout()
    output_path = KENDALLS_TAU_DIR / "kendalls_tau_aggregated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved aggregated Kendall's tau to: {output_path}")


def plot_pearsons_r_aggregated() -> None:
    """Plot Pearson's R averaged across batches and both versions."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Average across both batches and versions
    aggregated = df.groupby(["l1_or_l2", "preview", "measure"])["avg_pearson_r"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Pearson's R (Aggregated Across Batches)", fontsize=14, fontweight='bold')

    for idx, (reader, ax) in enumerate(zip(["L1", "L2"], axes)):
        data = aggregated[aggregated["l1_or_l2"] == reader]
        pivot_data = data.pivot_table(index="preview", columns="measure", values="avg_pearson_r")

        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0, vmax=0.4,
                    cbar_kws={"label": "Mean Pearson's R"})
        ax.set_title(f"{reader} Speakers", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eye-Tracking Measure")
        ax.set_ylabel("Preview")

    plt.tight_layout()
    output_path = PEARSONS_R_DIR / "pearsons_r_aggregated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved aggregated Pearson's R to: {output_path}")


def plot_combined_kendalls_w_aggregated() -> None:
    """Plot Kendall's W for combined L1 and L2 speakers (all readers as one group)."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Get only combined data
    combined_data = df[df["l1_or_l2"] == "L1_and_L2"]

    if combined_data.empty:
        print("No combined L1_and_L2 data found. Run measure_agreement.py first.")
        return

    # Average across batches and versions
    aggregated = combined_data.groupby(["preview", "measure"])["kendalls_w"].mean().reset_index()

    # Create heatmap for combined speakers
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_data = aggregated.pivot_table(
        index="preview",
        columns="measure",
        values="kendalls_w"
    )

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        cbar_kws={"label": "Mean Kendall's W"},
        ax=ax,
        vmin=0,
        vmax=0.4,
    )
    ax.set_title("Kendall's W: All Readers Combined (L1 + L2)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Eye-Tracking Measure (TF, RR, SR)", fontsize=12)
    ax.set_ylabel("Preview Condition", fontsize=12)

    plt.tight_layout()
    output_path = KENDALLS_W_DIR / "kendalls_w_combined_all_readers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined Kendall's W to: {output_path}")


def plot_combined_spearmans_rho_aggregated() -> None:
    """Plot Spearman's rho for combined L1 and L2 speakers."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Get only combined data
    combined_data = df[df["l1_or_l2"] == "L1_and_L2"]

    if combined_data.empty:
        print("No combined L1_and_L2 data found. Run measure_agreement.py first.")
        return

    # Average across batches and versions
    aggregated = combined_data.groupby(["preview", "measure"])["avg_spearmans_rho"].mean().reset_index()

    # Create heatmap for combined speakers
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_data = aggregated.pivot_table(
        index="preview",
        columns="measure",
        values="avg_spearmans_rho"
    )

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        cbar_kws={"label": "Mean Spearman's Rho"},
        ax=ax,
        vmin=0,
        vmax=0.4,
    )
    ax.set_title("Average Pairwise Spearman's Rho: All Readers Combined (L1 + L2)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Eye-Tracking Measure (TF, RR, SR)", fontsize=12)
    ax.set_ylabel("Preview Condition", fontsize=12)

    plt.tight_layout()
    output_path = SPEARMANS_RHO_DIR / "spearmans_rho_combined_all_readers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined Spearman's rho to: {output_path}")


def plot_combined_kendalls_tau_aggregated() -> None:
    """Plot Kendall's tau for combined L1 and L2 speakers."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Get only combined data
    combined_data = df[df["l1_or_l2"] == "L1_and_L2"]

    if combined_data.empty:
        print("No combined L1_and_L2 data found. Run measure_agreement.py first.")
        return

    # Average across batches and versions
    aggregated = combined_data.groupby(["preview", "measure"])["avg_kendalls_tau"].mean().reset_index()

    # Create heatmap for combined speakers
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_data = aggregated.pivot_table(
        index="preview",
        columns="measure",
        values="avg_kendalls_tau"
    )

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        cbar_kws={"label": "Mean Kendall's Tau"},
        ax=ax,
        vmin=0,
        vmax=0.4,
    )
    ax.set_title("Average Pairwise Kendall's Tau: All Readers Combined (L1 + L2)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Eye-Tracking Measure (TF, RR, SR)", fontsize=12)
    ax.set_ylabel("Preview Condition", fontsize=12)

    plt.tight_layout()
    output_path = KENDALLS_TAU_DIR / "kendalls_tau_combined_all_readers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined Kendall's tau to: {output_path}")


def plot_combined_pearsons_r_aggregated() -> None:
    """Plot Pearson's R for combined L1 and L2 speakers."""
    csv_path = Path(__file__).parent / "overall_agreement_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Get only combined data
    combined_data = df[df["l1_or_l2"] == "L1_and_L2"]

    if combined_data.empty:
        print("No combined L1_and_L2 data found. Run measure_agreement.py first.")
        return

    # Average across batches and versions
    aggregated = combined_data.groupby(["preview", "measure"])["avg_pearson_r"].mean().reset_index()

    # Create heatmap for combined speakers
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_data = aggregated.pivot_table(
        index="preview",
        columns="measure",
        values="avg_pearson_r"
    )

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        cbar_kws={"label": "Mean Pearson's R"},
        ax=ax,
        vmin=0,
        vmax=0.4,
    )
    ax.set_title("Average Pairwise Pearson's R: All Readers Combined (L1 + L2)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Eye-Tracking Measure (TF, RR, SR)", fontsize=12)
    ax.set_ylabel("Preview Condition", fontsize=12)

    plt.tight_layout()
    output_path = PEARSONS_R_DIR / "pearsons_r_combined_all_readers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined Pearson's R to: {output_path}")

if __name__ == "__main__":
    print("="*80)
    print("VISUALIZING AGREEMENT METRICS (KENDALL'S W & PAIRWISE METRICS)")
    print("="*80)

    print("\n" + "─"*80)
    print("COMBINED (L1 + L2 as one group)")
    print("─"*80)

    print("\n1. Kendall's W for all readers combined...")
    plot_combined_kendalls_w_aggregated()

    print("\n2. Average Spearman's rho for all readers combined...")
    plot_combined_spearmans_rho_aggregated()

    print("\n3. Average Kendall's tau for all readers combined...")
    plot_combined_kendalls_tau_aggregated()

    print("\n4. Average Pearson's R for all readers combined...")
    plot_combined_pearsons_r_aggregated()

    print("\n" + "─"*80)
    print("KENDALL'S W (Concordance)")
    print("─"*80)

    print("\n5. L1 vs L2 Kendall's W comparison (by batch)...")
    plot_l1_vs_l2_kendalls_w()

    print("\n6. Kendall's W aggregated by preview (averaged across batches & versions)...")
    plot_kendalls_w_aggregated_by_preview()

    print("\n7. L1 vs L2 Kendall's W comparison (aggregated across batches)...")
    plot_l1_vs_l2_aggregated()

    print("\n" + "─"*80)
    print("VERSION COMPARISON (V0=even IDs vs V1=odd IDs)")
    print("─"*80)

    print("\n8. Kendall's W for both versions (all conditions)...")
    plot_both_versions_kendalls_w()

    print("\n9. Kendall's W averaged across versions...")
    plot_version_average_kendalls_w()

    print("\n" + "─"*80)
    print("PAIRWISE METRICS (Average Spearman's Rho & Kendall's Tau)")
    print("─"*80)

    print("\n10. Average Spearman's rho aggregated by preview (averaged across batches & versions)...")
    plot_spearmans_rho_aggregated()

    print("\n11. Average Kendall's tau aggregated by preview (averaged across batches & versions)...")
    plot_kendalls_tau_aggregated()

    print("\n12. Spearman's rho for both versions (all conditions)...")
    plot_both_versions_spearmans_rho()

    print("\n13. Kendall's tau for both versions (all conditions)...")
    plot_both_versions_kendalls_tau()

    print("\n14. Spearman's rho averaged across versions...")
    plot_version_average_spearmans_rho()

    print("\n15. Kendall's tau averaged across versions...")
    plot_version_average_kendalls_tau()

    print("\n" + "─"*80)
    print("PAIRWISE METRICS (Average Pearson's R)")
    print("─"*80)

    print("\n16. Average Pearson's R aggregated by preview (averaged across batches & versions)...")
    plot_pearsons_r_aggregated()

    print("\n17. Pearson's R for both versions (all conditions)...")
    plot_both_versions_pearsons_r()

    print("\n18. Pearson's R averaged across versions...")
    plot_version_average_pearsons_r()

    print("\n" + "="*80)
    print("✓ All visualizations saved to: src/Agreement/plots/")
    print("="*80)


