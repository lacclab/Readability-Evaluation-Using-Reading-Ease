"""
Display names, direction mappings, and column group definitions
for paradigm divergence analysis.
"""

from src.Correlations.define_cols import (
    MAIN_RT_COLS, SM_RT_COLS,
    TRADITIONAL_MEASURES, MODERN_MEASURES, SYSTEMS_MEASURES,
    PSYCHOLINGUISTIC_MEASURES, MAIN_PROMPT_COLS,
    READING_COMPREHENSION_COLS,
)

# ---- Column groups ----

ET_COLS = MAIN_RT_COLS
FORMULA_COLS = TRADITIONAL_MEASURES + MODERN_MEASURES + SYSTEMS_MEASURES
PSYCHOLING_COLS = PSYCHOLINGUISTIC_MEASURES
LLM_COLS = MAIN_PROMPT_COLS
COMPREHENSION_COLS = READING_COMPREHENSION_COLS
ALL_RT_COLS = MAIN_RT_COLS + SM_RT_COLS

# ---- Direction mappings ----

# RT cols where higher value = easier to read (opposite to default)
OPPOSITE_RT_COLS = {
    'SkipRateTotal', 'SkipRateFirstPass',
    'reading_speed',
}

# Comprehension cols where higher value = harder (opposite to default)
OPPOSITE_COMP_COLS = {
    'QA_RT',  # higher response time = harder
}

# ---- Display names ----

RT_DISPLAY = {
    'mean_nonzero_TF': 'Total Fixation Time',
    'SkipRateTotal': 'Skip Rate',
    'RegRateTotal': 'Regression Rate',
    'mean_nonzero_FF': 'First Fixation Duration',
    'mean_FD': 'Fixation Duration',
    'mean_NF': 'Number of Fixations',
    'mean_FirstPassGD': 'First Pass Gaze Duration',
    'SkipRateFirstPass': 'First Pass Skip Rate',
    'RegRateFirstPass': 'First Pass Regression Rate',
    'mean_GD': 'Gaze Duration',
    'mean_HigherPassFixation': 'Higher Pass Fixation',
    'reading_speed': 'Reading Speed',
}

COMP_DISPLAY = {
    'comprehension_score': 'Comprehension Accuracy',
    'QA_RT': 'QA Response Time',
}


def rt_difficulty_direction(rt_col):
    """Return (sign, high_label, low_label) for an RT col.
    sign: +1 if higher value = harder, -1 if higher value = easier.
    """
    if rt_col in OPPOSITE_RT_COLS:
        return -1, "Easy to read", "Hard to read"
    return +1, "Hard to read", "Easy to read"


def comp_difficulty_direction(comp_col):
    """Return (sign, high_label, low_label) for a comp col.
    sign: +1 if higher value = better comprehension, -1 if higher = worse.
    """
    if comp_col in OPPOSITE_COMP_COLS:
        return -1, "Hard to comprehend", "Easy to comprehend"
    return +1, "Easy to comprehend", "Hard to comprehend"
