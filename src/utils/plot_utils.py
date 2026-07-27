import matplotlib.patches as mpatches
import numpy as np
import textwrap
from src.constants import (LEVEL_LABELS)

DELTA = '\u2206'

BASE_COLOR = "#4778BB"
BASE_COLOR_2 = "#D7527F"

# Constants for significance colors
SIGNIFICANCE_COLORS = {
    '***': '#cf4d40',  # Highly significant
    '**': '#e67e22',   # Moderately significant
    '*': '#f1c40f',    # Weakly significant
    'ns': '#707b7c',    # Not significant (default)
}

SIGNIFICANCE_LABELS = {
    '***': '(p < 0.001)',
    '**': '(p < 0.01)',
    '*': '(p < 0.05)',
    'ns': '(p >= 0.05)',
}

# ---------------------------------------
# Used for paper on L1 vs. L2 differences
# ---------------------------------------

L1_COLOR = "#DB4046"
L2_COLOR = "#2D72CC"
L1_COLOR_DARK = "#A60303"
L2_COLOR_DARK = "#203D80"

# Constants for significance colors
SIGNIFICANCE_COLORS_L1 = {
    '***': "#A60303",  # Highly significant AB293C
    '**': '#ED1313',   # Moderately significant
    '*': "#FF9898",    # Weakly significant
    'ns': '#C0C4CF',    # Not significant (default)
}

# Constants for significance colors
SIGNIFICANCE_COLORS_L2 = {
    '***': "#203D80",  # Highly significant 2C50A3
    '**': "#2E82F1",   # Moderately significant
    '*': "#A0C9FB",    # Weakly significant
    'ns': '#C0C4CF',    # Not significant (default)
}

# ---------------------------------------
# Used for paper on readability formulas
# ---------------------------------------

SIGNIFICANCE_SIGN_DIFF_COLORS = {
    '*** +': '#ba4a00',  # Highly significant, Positive Difference
    '** +': '#e67e22',   # Moderately significant, Positive Difference
    '* +': '#f0b27a',    # Weakly significant, Positive Difference
    'ns +': '#e5e8e8',    # Not significant (default), Positive Difference
    'ns -': '#e5e8e8',    # Not significant (default), Negative Difference
    '* -': '#85c1e9',    # Weakly significant, Negative Difference
    '** -': '#2e86c1',   # Moderately significant, Negative Difference
    '*** -': '#1f618d',  # Highly significant, Negative Difference
}

SIGNIFICANCE_SIGN_DIFF_LABELS = {
    '*** +': 'Positive Diff (p < 0.001)',
    '** +': 'Positive Diff (p < 0.01)',
    '* +': 'Positive Diff (p < 0.05)',
    'ns +': 'Positive Diff (p >= 0.05)',
    'ns -': 'Negative Diff (p >= 0.05)',
    '* -': 'Negative (p < 0.05)',
    '** -': 'Negative (p < 0.01)',
    '*** -': 'Negative (p < 0.001)',
}

SIGNIFICANCE_SIGN_SLOPE_LABELS = {
    '*** +': 'Positive Slope (p < 0.001)',
    '** +': 'Positive Slope (p < 0.01)',
    '* +': 'Positive Slope (p < 0.05)',
    'ns +': 'Positive Slope (p >= 0.05)',
    'ns -': 'Negative Slope (p >= 0.05)',
    '* -': 'Negative Slope (p < 0.05)',
    '** -': 'Negative Slope (p < 0.01)',
    '*** -': 'Negative Slope (p < 0.001)',
}

# For hatching to differentiate levels
LEVEL_HATCH = {
    'Adv':  '',   # no hatch
    'Ele':  '///' # diagonal hatch
}

LEVEL_COLORS = {'Ele': '#AEC6CF', 'Adv': '#FFB347'}
LEVEL_TEXT_COLORS = {'Ele': '#7A8B91', 'Adv': '#B37D32'}
SIGNIFICANCE_COLORS_BY_STR = {'Significant Positive': '#45b39d', 'Significant Negative': '#ec7063', 'Not Significant': '#85929e'}
SIGN_COLORS_BY_STR = {'Positive': '#45b39d', 'Negative': '#ec7063', 'Neutral': '#85929e'}

# Constants for effect color
EFFECT_COLOR = '#16a085'
EFFECT_COLOR_L1 = L1_COLOR
EFFECT_COLOR_L2 = L2_COLOR

EFFECT_COLOR_L1_CONTRAST = L1_COLOR
EFFECT_COLOR_L2_CONTRAST = L2_COLOR
NEUTRAL_COLOR = '#59168B'  # purple, not assigned to L1 or L2

SIGNIFICANCE_MARKERSIZE = {
    '***': 8,
    '**': 7,
    '*': 6,
    'ns': 4
}

MARKER_SHAPES_BY_FAMILY = {
    "Pythia":   "s",  # square
    "GPT-2":    "o",  # circle
    "GPT-J":    "d",  # thin diamond
    "GPT-Neo":  "D",  # large diamond
    "Llama-2":  "^",  # triangle up
    "OPT":      "h",  # hexagon
    "Mistral":  "v",  # triangle down
    "Gemma":    ">",  # triangle right
    "RWKV":     "<",  # triangle left
    "Mamba":    "p",  # pentagon
}

SIGNIFICANCE_SHAPES = {
    '***': "o",  # circle
    '**': "o",  # circle
    '*': "o",  # circle
    'ns': '^',
}

MARKER_COLORS_BY_FAMILY = {
    "Pythia":   "#f4d03f",  # yellow
    "GPT-2":    "#a9cce3",  # light blue
    "GPT-J":    "#2980b9",  # blue
    "GPT-Neo":  "#229954",  # green
    "Llama-2":  "#9b59b6",  # purple
    "OPT":      "#616a6b",  # dark grey
    "Mistral":  "#c0392b",  # red
    "Gemma":    "#e67e22",  # orange
    "RWKV":     "#ea98e0",  # pink
    "Mamba":    "#a9dfbf",  # light green
}


def format_p_value(p: float) -> str:
    """Format p-value into discrete significance level string."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return "p > 0.05"


def add_significance_legend(
    fig, 
    with_colors=True, 
    handlelength=0.5, 
    bbox_to_anchor=(0.5, 0.0), 
    add_base_colors_legend=False,
    add_level_hatch_legend=False,
    add_L1_L2_legend=False,
    add_L1_L2_significance_legend=False,
    increase_fontsize=False,
    significance_colors: dict = SIGNIFICANCE_COLORS,
    significance_labels: dict = SIGNIFICANCE_LABELS,
    fontsize_legend_text=9,
    ):
    # Build a single legend for significance colors
    handles = {}
    
    if not with_colors:
        # create a legend without colors
        for sig_symbol in significance_labels.keys():
            if sig_symbol is None:
                continue
            label = f"{sig_symbol} {significance_labels[sig_symbol]}"
            patch = mpatches.Patch(color='white', label=label)
            handles[sig_symbol] = patch
    else:
        for sig_symbol, color in significance_colors.items():
            if sig_symbol is None:
                continue
            label = f"{sig_symbol} {significance_labels[sig_symbol]}"
            patch = mpatches.Patch(color=color, label=label)
            handles[sig_symbol] = patch
    
    if '* +' in significance_colors:
        symbols_order = list(significance_colors.keys())
    else:
        symbols_order = ['ns', '*', '**', '***']
    handles_list = [handles[sig_symbol] for sig_symbol in symbols_order]
    
    if add_base_colors_legend:
        # add BASE_COLOR, BASE_COLOR_2 that represents Reading Fluency Metrics and Reading Comprehension Metrics
        patch = mpatches.Patch(color=BASE_COLOR, label="Online Measures")
        handles['Online'] = patch
        patch = mpatches.Patch(color=BASE_COLOR_2, label="Offline Measures")
        handles['Offline'] = patch

        # handles_list order: ['ns', 'Online' '*', 'Offline' '**', '***']
        handles_list = [handles[sig_symbol] for sig_symbol in ['ns', 'Online', '*', 'Offline', '**', '***']]
    
    if add_level_hatch_legend:
        # add level hatch legend
        adv_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            hatch='',      # no hatch for 'Adv'
            label=LEVEL_LABELS['Adv'],
        )
        ele_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            hatch='///',   # diagonal hatch for 'Ele'
            label=LEVEL_LABELS['Ele'],
        )
        handles['Adv'] = adv_patch
        handles['Ele'] = ele_patch
        
        handles_list = [handles[sig_symbol] for sig_symbol in ['ns', '*', '**', '***', 'Online', 'Offline', 'Adv', 'Ele']]

    if add_L1_L2_legend:
        adv_patch = mpatches.Patch(
            facecolor='white', edgecolor='black', hatch='',
            label=LEVEL_LABELS['Adv'],
        )
        ele_patch = mpatches.Patch(
            facecolor='white', edgecolor='black', hatch='///',
            label=LEVEL_LABELS['Ele'],
        )
        l1_patch = mpatches.Patch(
            facecolor=L1_COLOR, edgecolor='black',
            label='L1',
        )
        l2_patch = mpatches.Patch(
            facecolor=L2_COLOR, edgecolor='black',
            label='L2',
        )
        handles['Adv'] = adv_patch
        handles['Ele'] = ele_patch
        handles['L1'] = l1_patch
        handles['L2'] = l2_patch
        # Double-row layout with ncol=8, 2 spacer columns between each group:
        # Row 1: ns,  *,   sp, sp, L1, sp, sp, Adv
        # Row 2: **, ***, sp, sp, L2, sp, sp, Ele
        spacers = [mpatches.Patch(facecolor='none', edgecolor='none', label=' ') for _ in range(8)]
        handles_list = [
            handles['ns'], handles['*'],
            handles['**'], handles['***'], 
            spacers[2], spacers[3], spacers[0], spacers[1], 
            handles['L1'], handles['L2'], 
            spacers[4], spacers[5], 
            spacers[6], spacers[7],
            handles['Adv'], handles['Ele']
        ]

    if add_L1_L2_significance_legend:
        # Build legend with separate L1 and L2 significance color scales
        handles = {}
        handles_list = []
        for sig_symbol in ['***', '**', '*']:
            label = f"L1 {sig_symbol} {significance_labels[sig_symbol]}"
            patch = mpatches.Patch(color=SIGNIFICANCE_COLORS_L1[sig_symbol], label=label)
            handles[f'L1_{sig_symbol}'] = patch
            handles_list.append(patch)
        # shared ns
        ns_patch = mpatches.Patch(color=SIGNIFICANCE_COLORS_L1['ns'], label=f"ns {significance_labels['ns']}")
        handles['ns'] = ns_patch
        handles_list.append(ns_patch)
        for sig_symbol in ['***', '**', '*']:
            label = f"L2 {sig_symbol} {significance_labels[sig_symbol]}"
            patch = mpatches.Patch(color=SIGNIFICANCE_COLORS_L2[sig_symbol], label=label)
            handles[f'L2_{sig_symbol}'] = patch
            handles_list.append(patch)

    if increase_fontsize:
        fontsize_legend_text = 14

    legend_columnspacing = 1.3
    if add_L1_L2_significance_legend:
        legend_ncol = 4
    elif add_L1_L2_legend:
        legend_ncol = 8
        legend_columnspacing = 0.3
    else:
        legend_ncol = 4

    leg = fig.legend(
        handles=handles_list, 
        loc='lower center', 
        bbox_to_anchor=bbox_to_anchor, 
        ncol=legend_ncol, 
        fontsize=fontsize_legend_text, 
        frameon=False,
        handlelength=handlelength,
        handletextpad=0.5,
        columnspacing=legend_columnspacing,
        borderaxespad=0.3
    )

    # Make spacer handles invisible
    if add_L1_L2_legend:
        for handle in leg.legend_handles:
            if handle.get_label() == ' ':
                handle.set_visible(False)
    # Optionally customize the frame’s appearance
    # leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_facecolor('white')
    # leg.get_frame().set_linewidth(0.3)
    return fig

def add_significance_bracket(ax, x1, x2, y, text, bracket_height=0.02, text_below=None, text_below_fontsize=9, text_fontsize=10):
    """
    Draw a significance bracket from x1 to x2 at vertical level y,
    with a small bracket "height" above y, and put the text (stars) in the center.
    """
    y_bracket = y + bracket_height
    ax.plot([x1, x1, x2, x2], [y, y_bracket, y_bracket, y], 
            color='black', linewidth=1)
    ax.text((x1 + x2) / 2.0, y_bracket, text,
            ha='center', va='bottom', color='black', fontsize=text_fontsize)
    if text_below is not None:
        ax.text((x1 + x2) / 2.0, y - bracket_height, text_below,
                ha='center', va='top', color='black', fontsize=text_below_fontsize)
        
        
def add_level_legend_to_subplot(ax, loc='upper right'):
    """
    Example of how to add an 'Adv' vs. 'Ele' legend in a single subplot 'ax'.
    """
    # Create the patch handles for level legend
    adv_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch='',      # no hatch for 'Adv'
        label=LEVEL_LABELS['Adv']
    )
    ele_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch='///',   # diagonal hatch for 'Ele'
        label=LEVEL_LABELS['Ele']
    )
    handles = [adv_patch, ele_patch]

    # Add the legend to this specific axes
    ax.legend(
        handles=handles,
        loc=loc,   # or wherever you prefer
        fontsize=8,
        frameon=False
    )
    
def _wrap_label(label_text, line_width=10):
    wrapped = textwrap.wrap(label_text, width=line_width)
    return "\n".join(wrapped)


def custom_raincloud_plot(merged_df, cols_to_plot, display_names, axes,
                          group_colors, group_labels, group_col='level'):
    """
    Create custom raincloud plots with half-violin, boxplot, and scatter for each column.

    Parameters:
    - merged_df: pd.DataFrame with data to plot
    - cols_to_plot: list of column names to plot
    - display_names: dict mapping column names to pretty names
    - axes: list of matplotlib axes objects (e.g., from plt.subplots)
    - group_colors: dict mapping group values to colors
    - group_labels: dict mapping group values to display labels
    - group_col: name of the column in merged_df that defines groups (default: 'level')
    """
    groups = list(group_colors.keys())
    colors = list(group_colors.values())

    for ax, col in zip(axes, cols_to_plot):
        data_x = [merged_df[merged_df[group_col] == grp][col].dropna() for grp in groups]

        # Boxplot
        bp = ax.boxplot(data_x, patch_artist=True, vert=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # Half Violin plot
        vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False)
        for idx, b in enumerate(vp['bodies']):
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
            b.set_color(colors[idx])
            b.set_alpha(0.5)

        # Scatter plot with jitter
        for idx, features in enumerate(data_x):
            y = np.full(len(features), idx + .8)
            y += np.random.uniform(low=-.05, high=.05, size=len(y))
            ax.scatter(features, y, s=5, c=colors[idx], alpha=0.6)

        # Y-axis ticks and labels
        ax.set_yticks(np.arange(1, len(groups)+1, 1))
        ax.set_yticklabels([group_labels[grp] for grp in groups])

        pretty = display_names[col]
        ax.set_xlabel(pretty)
        ax.margins(x=0.02)

    return axes