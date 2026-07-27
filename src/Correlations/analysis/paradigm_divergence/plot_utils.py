"""
Reusable plotting helpers for paradigm divergence scatter plots.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def format_p(p: float) -> str:
    """Format p-value: use plain decimal when readable, scientific for tiny values."""
    if np.isnan(p):
        return "p=NaN"
    if p >= 0.01:
        return f"p={p:.3f}"
    if p >= 0.001:
        return f"p={p:.4f}"
    return f"p={p:.2e}"


def scatter_with_labels(ax, x, y, text_ids, xlabel, ylabel, title,
                        n_label=5, show_zero_lines=True, title_bold=False):
    """Scatter plot with extreme-divergence points labeled.

    Returns (r, p) Pearson correlation.
    """
    ax.scatter(x, y, alpha=0.5, s=30, c='steelblue', edgecolors='white',
               linewidths=0.5)

    if show_zero_lines:
        ax.axhline(0, color='grey', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='grey', alpha=0.3, linewidth=0.5)

    # Correlation
    mask = ~(np.isnan(x) | np.isnan(y))
    r, p = np.nan, np.nan
    title_weight = 'bold' if title_bold else 'normal'
    if mask.sum() > 5:
        r, p = pearsonr(x[mask], y[mask])
        ax.set_title(f"{title}\nr={r:.3f}, {format_p(p)}", fontsize=11,
                     fontweight=title_weight)
    else:
        ax.set_title(title, fontsize=11, fontweight=title_weight)

    # Label most divergent: normalize to [0,1] for fair distance
    x_s = pd.Series(x.values if hasattr(x, 'values') else x, index=x.index if hasattr(x, 'index') else None)
    y_s = pd.Series(y.values if hasattr(y, 'values') else y, index=y.index if hasattr(y, 'index') else None)
    x_range = x_s.max() - x_s.min()
    y_range = y_s.max() - y_s.min()
    if x_range > 0 and y_range > 0:
        x_norm = (x_s - x_s.min()) / x_range
        y_norm = (y_s - y_s.min()) / y_range
        residual = (y_norm - x_norm).abs()
        top_idx = residual.nlargest(n_label).index
        for idx in top_idx:
            ax.annotate(
                text_ids.loc[idx],
                (x.loc[idx], y.loc[idx]),
                fontsize=7, alpha=0.8,
                textcoords="offset points", xytext=(5, 5),
                arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5),
            )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    return r, p


def add_quadrant_labels(ax, top_right=None, bottom_left=None,
                        top_left=None, bottom_right=None):
    """Add italic quadrant labels to corners of a scatter plot."""
    positions = {
        'top_left': (0.02, 0.98, 'top', 'left'),
        'top_right': (0.98, 0.98, 'top', 'right'),
        'bottom_left': (0.02, 0.02, 'bottom', 'left'),
        'bottom_right': (0.98, 0.02, 'bottom', 'right'),
    }
    labels = {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right,
    }
    for key, text in labels.items():
        if text:
            x, y, va, ha = positions[key]
            ax.text(x, y, text,
                    transform=ax.transAxes, fontsize=8, alpha=0.4,
                    va=va, ha=ha, style='italic')


def report_extremes(df_clean, col_x, col_y, name_x, name_y, r, p,
                    subplot_label=None, n=5):
    """Find texts in each divergence quadrant using normalized residuals.

    Args:
        subplot_label: e.g. "Raw diff (top-right subplot)" to clarify
                       which subplot these values correspond to.

    Returns list of report lines.
    """
    df_clean = df_clean.copy()
    x = df_clean[col_x]
    y = df_clean[col_y]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range > 0 and y_range > 0:
        x_norm = (x - x.min()) / x_range
        y_norm = (y - y.min()) / y_range
        df_clean["_resid"] = y_norm - x_norm
    else:
        df_clean["_resid"] = 0

    lines = []
    lines.append(f"  {name_x} vs {name_y}")
    if subplot_label:
        lines.append(f"  [{subplot_label}]")
    lines.append(f"  Pearson r={r:.3f}, {format_p(p)}")
    lines.append("")

    high_x_low_y = df_clean.nsmallest(n, "_resid")
    lines.append(f"  High {name_x}, Low {name_y} (top {n}):")
    for _, row in high_x_low_y.iterrows():
        lines.append(
            f"    {row['text_id_str']:>10}  "
            f"{name_x}={row[col_x]:+.4f}  "
            f"{name_y}={row[col_y]:+.4f}"
        )

    low_x_high_y = df_clean.nlargest(n, "_resid")
    lines.append(f"\n  Low {name_x}, High {name_y} (top {n}):")
    for _, row in low_x_high_y.iterrows():
        lines.append(
            f"    {row['text_id_str']:>10}  "
            f"{name_x}={row[col_x]:+.4f}  "
            f"{name_y}={row[col_y]:+.4f}"
        )
    lines.append("")
    return lines
