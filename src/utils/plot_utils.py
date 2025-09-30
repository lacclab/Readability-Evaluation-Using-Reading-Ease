import matplotlib.patches as mpatches

DELTA = '\u2206'

BASE_COLOR = "#4778BB"
BASE_COLOR_2 = "#43976e"

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


def add_significance_legend(
    fig, 
    with_colors=True, 
    handlelength=0.5, 
    bbox_to_anchor=(0.5, 0.0), 
    add_base_colors_legend=False,
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
    
    if increase_fontsize:
        fontsize_legend_text = 14

    _ = fig.legend(
        handles=handles_list, 
        loc='lower center', 
        bbox_to_anchor=bbox_to_anchor, 
        ncol=4, 
        fontsize=fontsize_legend_text, 
        frameon=False,  # Turn on the rectangular frame,
        handlelength=handlelength,      # length of the legend handle
        handletextpad=0.5,     # space between handle and label
        columnspacing=0.8,     # space between columns
        borderaxespad=0.3      # space between legend box & axes
    )
    # Optionally customize the frame’s appearance
    # leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_facecolor('white')
    # leg.get_frame().set_linewidth(0.3)
    return fig

def add_significance_bracket(ax, x1, x2, y, text, bracket_height=0.02):
    """
    Draw a significance bracket from x1 to x2 at vertical level y,
    with a small bracket "height" above y, and put the text (stars) in the center.
    """
    y_bracket = y + bracket_height
    ax.plot([x1, x1, x2, x2], [y, y_bracket, y_bracket, y], 
            color='black', linewidth=1)
    ax.text((x1 + x2) / 2.0, y_bracket, text,
            ha='center', va='bottom', color='black', fontsize=10)