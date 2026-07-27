from pathlib import Path

DEFAULT_RANDOM_STATE = 42

# Readability project paths
READABILITY_BASE_PATH = Path("/data/home/shared/projects/readability")
# simplifaction effects results folder
EFFECTS_RESULTS_PATH = Path("Simplification_Effects/results")

LN_SHARED_READING_HABITS_PATH = "/data/home/shared/data/onestop_survey/reading_habits.csv"

# L1 paths
EYE_DF_L1_VERSION_DATE = "20250126"
EYE_DF_L1_VERSION = f"OneStop_v1_{EYE_DF_L1_VERSION_DATE}"
EYE_BY_WORD_DF_L1_PATH = f"/data/home/shared/data/onestop_L1/reports/archive/{EYE_DF_L1_VERSION}/lacclab_processed/ia_Paragraph.csv"
EYE_BY_WORD_DF_L1_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/{EYE_DF_L1_VERSION}_ia_P_aligned.csv"
EYE_BY_FIXATION_DF_L1_PATH = f"/data/home/shared/data/onestop_L1/reports/archive/{EYE_DF_L1_VERSION}/lacclab_processed/fixations_Paragraph.csv"
EYE_BY_FIXATION_DF_L1_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/{EYE_DF_L1_VERSION}_fix_P_aligned.csv"
PROCESSED_L1_EYE_METRICS_PATH = Path(f"{READABILITY_BASE_PATH}/{EYE_DF_L1_VERSION}")

# L2 Paths
EYE_DF_L2_VERSION = "20260610"
EYE_DF_L2_SPLIT = "splits/keep" # not full - we want only partcipants we reviewed and approved for analysis

ONESTOP_L2_REPORTS_BASE_PATH = "/data/home/shared/data/onestop_L2/reports/lacclab"
ONESTOP_L2_METADATA_BASE_PATH = "/data/home/shared/data/onestop_L2/metadata"
EYE_BY_WORD_DF_L2_PATH = f"{ONESTOP_L2_REPORTS_BASE_PATH}/{EYE_DF_L2_VERSION}/{EYE_DF_L2_SPLIT}/ia_Paragraph.csv.zip"
EYE_BY_WORD_DF_L2_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/ia_P_L2_{EYE_DF_L2_VERSION}_aligned.csv"
EYE_BY_FIXATION_DF_L2_PATH = f"{ONESTOP_L2_REPORTS_BASE_PATH}/{EYE_DF_L2_VERSION}/{EYE_DF_L2_SPLIT}/fixations_Paragraph.csv.zip"
EYE_BY_FIXATION_DF_L2_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/fix_P_L2_{EYE_DF_L2_VERSION}_aligned.csv"
PROCESSED_L2_EYE_METRICS_PATH = Path(f"{READABILITY_BASE_PATH}/{EYE_DF_L2_VERSION}")
L2_METADATA_PATH = f"{ONESTOP_L2_METADATA_BASE_PATH}/metadata.csv"
L2_MICHIGEN_TEST_PATH = f"{ONESTOP_L2_METADATA_BASE_PATH}/michtest_results.csv"
LN_SHARED_L2_SESSION_SUMMARY_PATH = f"{ONESTOP_L2_METADATA_BASE_PATH}/session_summary.csv"
L2_SESSION_SUMMARY_PATH = f"{ONESTOP_L2_METADATA_BASE_PATH}/session_summary.csv"
L2_FULL_SESSION_SUMMARY_PATH = f"{ONESTOP_L2_METADATA_BASE_PATH}/full_session_summary.csv"

# L1_and_L2 Paths
EYE_BY_WORD_DF_L1_AND_L2_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/ia_P_L1_and_L2_{EYE_DF_L1_VERSION}_{EYE_DF_L2_VERSION}_aligned.csv"
EYE_BY_FIXATION_DF_L1_AND_L2_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/fix_P_L1_and_L2_{EYE_DF_L1_VERSION}_{EYE_DF_L2_VERSION}_aligned.csv"
PROCESSED_L1_AND_L2_EYE_METRICS_PATH = Path(f"{READABILITY_BASE_PATH}/L1_and_L2_{EYE_DF_L1_VERSION}_{EYE_DF_L2_VERSION}")

# EZ Reader paths
EZ_READER_VERSION_DATE = "20250122"
EZ_READER_EYE_BY_WORD_DF_PATH = f"/data/home/shared/data/onestop_L1/reports/archive/{EYE_DF_L1_VERSION}/ez_reader/ezreader_n1000_{EZ_READER_VERSION_DATE}.csv"
EZ_READER_EYE_BY_WORD_DF_ALIGNED_PATH = f"{READABILITY_BASE_PATH}/ezreader_n1000s_{EZ_READER_VERSION_DATE}.csv"
PROCESSED_EZ_EYE_METRICS_PATH = Path(f"{READABILITY_BASE_PATH}/ezreader_{EZ_READER_VERSION_DATE}")

# PAths Dicts
PROCESSED_EYE_METRICS_PATHS = {
    'L1': PROCESSED_L1_EYE_METRICS_PATH,
    'L2': PROCESSED_L2_EYE_METRICS_PATH,
    'L1_and_L2': PROCESSED_L1_AND_L2_EYE_METRICS_PATH,
    'EZ': PROCESSED_EZ_EYE_METRICS_PATH,
    'other': Path(f"{READABILITY_BASE_PATH}/other")
}
EYE_BY_WORD_PATHS = {
    'L1': EYE_BY_WORD_DF_L1_PATH,
    'L2': EYE_BY_WORD_DF_L2_PATH,
    'EZ': EZ_READER_EYE_BY_WORD_DF_PATH
}
EYE_BY_WORD_ALIGNED_PATHS = {
    'L1': EYE_BY_WORD_DF_L1_ALIGNED_PATH,
    'L2': EYE_BY_WORD_DF_L2_ALIGNED_PATH,
    'L1_and_L2': EYE_BY_WORD_DF_L1_AND_L2_ALIGNED_PATH,
    'EZ': EZ_READER_EYE_BY_WORD_DF_ALIGNED_PATH
}
EYE_BY_FIXATION_PATHS = {
    'L1': EYE_BY_FIXATION_DF_L1_PATH,
    'L2': EYE_BY_FIXATION_DF_L2_PATH
}
EYE_BY_FIXATION_ALIGNED_PATHS = {
    'L1': EYE_BY_FIXATION_DF_L1_ALIGNED_PATH,
    'L2': EYE_BY_FIXATION_DF_L2_ALIGNED_PATH,
    'L1_and_L2': EYE_BY_FIXATION_DF_L1_AND_L2_ALIGNED_PATH
}

# ---------------
# Eye Measures
# ---------------

TF_THRESHOLD = 3000

EYE_METRICS = {
    "FF": "IA_FIRST_FIXATION_DURATION",
    "GD": "IA_FIRST_RUN_DWELL_TIME",
    "TF": "IA_DWELL_TIME",
    "FirstFixProg": "IA_FIRST_FIX_PROGRESSIVE",
    "RegPD": "IA_REGRESSION_PATH_DURATION",
    'NF': "IA_FIXATION_COUNT",
    "FirstPassFF": "FirstPassFFD"
}

EYE_METRICS_INVERTED = {value: key for key, value in EYE_METRICS.items()}

# ---------------
# Other
# ---------------

SIMILARITY_COL = "all-MiniLM-L6-v2_cosine_similarity"

# ---------------------
# Readability Measures
# ---------------------

READABILITY_FORMULAS = [
    'coleman_liau_index',
    'dale_chall_score',
    'ARI',
    'flesch_reading_ease',
    'flesch_kincaid_grade_score', 
    'gunning_fog_index',     
    'linsear_formula',
    'smog_index',
]

ARTE_METRICS = [
    'CARES',
    'CAREC',
    'CAREC_M',
    'CML2RI'
]

PSYCHOLINGUISTIC_METRICS = [
    "cpidr_density", 
    "depid_density",
    "max_embedding_depth", 
    "avg_embedding_depth",
]

LEN_FREQ_SURP_METRICS = [
    "surprisal",
    "word_length",
    "text_length",
    "wordFreq_frequency",
    "mean_entropy",
    "mean_entropy_sents"
]

LANGUAGES_METRICS = [
    "fernansex_huerta",
    "szigriszt_pazos",
    "gutierrez_polini",
    "crawford"
]

EXTRA_TEXTSTATS = [
    "difficult_words_count",
    "syllable_count",
    "char_count",
    "letter_count",
    "polysyllabcount",
    "monosyllabcount"
]


# Words to replace -> short alias
SIMPLIFICATION_TYPES_SHORT_LABELS = {
    "addition": "add",
    "deletion": "del",
    "deleted_sentence": "s_del",
    "paraphrasing": "par",
    "split": "spl"
}

PRED_COLS_SHORT_LABELS = {
    'mean_TF': 'Total Fixation (ms)\ninclude 0s',
    'mean_nonzero_TF': 'Total Fixation (ms)',
    'TF': 'Total Fixation (ms) \ninclude 0s',
    'GD': 'Gaze Duration (ms) \ninclude 0s',
    'FF': 'First Fixation (ms) \ninclude 0s',
    'NF': 'Fixation Count',
    'nonzero_TF': 'Total Fixation (ms)',
    'nonzero_GD': 'Gaze Duration (ms)',
    'nonzero_FF': 'First Fixation (ms)',
    'FirstPassGD': 'First Pass \nGaze Duration (ms)',
    'FirstPassFF': 'First Pass \nFirst Fixation (ms)',
    'HigherPassFixation': 'Higher Pass \nFixation Duration (ms)',
    'SkipTotal': 'Skip Probability',
    'SkipRateTotal': 'Skip Rate',
    'SkipFirstPass': 'First Pass \nSkip Rate',
    'SkipRateFirstPass': 'First Pass \nSkip Probability',
    'RegRateTotal': 'Regression Rate',
    'RegCountTotal': 'Regression Rate',
    'RegRateFirstPass': 'First Pass \nRegression Rate',
    'RegCountFirstPass': 'First Pass \nRegression Rate',    
    'IsReg': 'Regression Probability',
    'is_correct': 'QA Accuracy',
    'QA_RT': 'QA Time (sec)',
    'norm_QA_RT': 'QA Time (ms per word)',
    'words_per_sec_based_P_RT': 'Reading Speed \n(words per second)',
    'is_student': 'Student',
    'is_secondary': 'Secondary/High School',
    'is_undergrad': 'Undergraduate',
    'is_postgrad': 'Postgraduate',
    'comprehension_score': 'Comprehension Accuracy',
    'reading_speed': 'Reading Speed (words per second)',
}

# dict to map RT_col to RT_col_label
PRED_COLS_FULL_LABELS = {
    'TF': 'Total Fixation (ms) include 0s',
    'GD': 'Gaze Duration (ms) include 0s',
    'FF': 'First Fixation (ms) include 0s',
    'NF': 'Fixation Count',
    
    'mean_TF': 'Total Fixation (ms)',
    'mean_GD': 'Gaze Duration (ms)',
    'mean_FF': 'First Fixation (ms)',
    'mean_NF': 'Fixation Count',
    'mean_FD': 'Fixation Duration (ms)',
    
    'nonzero_TF': 'Total Fixation (ms)',
    'nonzero_GD': 'Gaze Duration (ms)',
    'nonzero_FF': 'First Fixation (ms)',
    
    'mean_nonzero_TF': 'Total Fixation (ms)',
    'mean_nonzero_GD': 'Gaze Duration (ms)',
    'mean_nonzero_FF': 'First Fixation (ms)',
    
    'HigherPassFixation': 'Higher Pass \nFixation Duration (ms)',
    'FirstPassGD': 'First Pass \nGaze Duration (ms)',
    'FirstPassFF': 'First Pass \nFirst Fixation (ms)',
    
    'mean_HigherPassFixation': 'Higher Pass \nFixation Duration (ms)',
    'mean_FirstPassGD': 'First Pass \nGaze Duration (ms)',
    'mean_FirstPassFF': 'First Pass \nFirst Fixation (ms)',
    
    'SkipTotal': 'Skip Probability',
    'SkipRateTotal': 'Skip Rate',
    'SkipFirstPass': 'First Pass \nSkip Rate',
    'SkipRateFirstPass': 'First Pass \nSkip Rate',
    
    'RegRateTotal': 'Regression Rate',
    'RegCountTotal': 'Regression Rate',
    'RegRateFirstPass': 'First Pass \nRegression Rate',
    'RegCountFirstPass': 'First Pass \nRegression Rate',
    'IsReg': 'Regression Probability',
    
    'is_correct': 'QA Accuracy',
    'QA_RT': 'Question Answering \nTime (sec)',
    'norm_QA_RT': 'Question Answering \nTime (ms per word)',
    
    'reading_speed': 'Reading Speed (words per second)',
    'words_per_sec_based_P_RT': 'Reading Speed \n(words per second)',
    
    'comprehension_score': 'Comprehension Accuracy',
}

REGIME_LABELS = {
    'Hunting': 'Information Seeking',
    'Hunting0': 'Information Seeking',
    'Gathering': 'Ordinary Reading',
    'Gathering0': 'Ordinary Reading',
}

READING_REGIMES_VALUES = {
    'Gathering0': {'has_preview': ['Gathering'], 'reread': [0]},
    'Gathering1': {'has_preview': ['Gathering'], 'reread': [1]},
    'Hunting0': {'has_preview': ['Hunting'], 'reread': [0]},
    'Hunting1': {'has_preview': ['Hunting'], 'reread': [1]},
    'FirstReading': {'has_preview': ['Gathering', 'Hunting'], 'reread': [0]},
    'RepeatedReading': {'has_preview': ['Gathering', 'Hunting'], 'reread': [1]},
    'All': {'has_preview': ['Gathering', 'Hunting'], 'reread': [0, 1]}
    }

SUBJECT_TEXT_LABELS = {
        "subject": "Participant",
        "text": "Paragraph"
    }

SELF_REPORTED_STR = "Reading Hours Per Week - "

LEVEL_LABELS = {
    "Ele": "Simplified",
    "Adv": "Original",
}
LEVEL_LABELS_SHORT = {
    "Ele": "Simp.",
    "Adv": "Orig.",
}

LEXTALE_BINS_NAMES = ['0_60', '60_70', '70_80', '80_90', '90_100']
ADV_COMP_BINS_NAMES = ['0*0_0*6', '0*6_0*7', '0*7_0*8', '0*8_0*9', '0*9_1*0']

BY_COL_LABELS = {
    "TF_per_word": "Total Fixation Time (ms) per word",
    "TF_per_word_adv": f"Total Fixation Time (ms) per word on {LEVEL_LABELS['Adv']} level",
    "P_RT_per_word": "Total Reading Time (include Saccades) (ms) per word",
    "P_RT_per_word_adv": f"Total Reading Time (include Saccades) (ms) per word on {LEVEL_LABELS['Adv']} level",
    "SaccT_per_word": "Total Saccades Time (ms) per word",
    "SaccT_per_word_adv": f"Total Saccades Time (ms) per word on {LEVEL_LABELS['Adv']} level",
    "words_per_sec_based_TF": "Reading Speed (words per second, based on TF)",
    "words_per_sec_based_TF_adv": f"Reading Speed (words per second, based on TF) on {LEVEL_LABELS['Adv']} level",
    "words_per_sec_based_SaccT": "Reading Speed (words per second, based on SaccT)",
    "words_per_sec_based_SaccT_adv": f"Reading Speed (words per second, based on SaccT) on {LEVEL_LABELS['Adv']} level",
    "words_per_sec_based_P_RT": "Reading Speed (words per second)",
    "words_per_sec_based_P_RT_adv": f"Reading Speed (words per second) on {LEVEL_LABELS['Adv']} level",
    "comprehension_score": "Comprehension Accuracy",
    "comprehension_score_adv": f"Comprehension Accuracy on {LEVEL_LABELS['Adv']} level",
    "n_total_reading": "Reading Hours per Week",
    "n_textbooks": f"{SELF_REPORTED_STR} Textbooks",
    "n_academic": f"{SELF_REPORTED_STR} Academic",
    "n_magazines": f"{SELF_REPORTED_STR} Magazines",
    "n_newspapers": f"{SELF_REPORTED_STR} Newspapers",
    "n_email": f"{SELF_REPORTED_STR} Email",
    "n_fiction": f"{SELF_REPORTED_STR} Fiction",
    "n_nonfiction": f"{SELF_REPORTED_STR} Nonfiction",
    "n_other_reading": f"{SELF_REPORTED_STR} Other",
    "years_secondary": "Years in Secondary/High School",
    "years_undergrad": "Years in Undergraduate",
    "years_postgrad": "Years in Postgraduate",
    "years_education": "Total Years of Education",
    "age": "Age",
    "text_length": "N words",
    "word_length": "Word Length",
    "wordFreq_frequency": "Word Frequency",
    "flesch_kincaid_grade_score": "Flesch-Kincaid\nGrade Score",
    "syllable_count": "Syllable Count",
    "deletion": "Deletion",
    "addition": "Addition",
    "paraphrasing": "Paraphrasing",
    "split": "Split",
    "deleted_sentence": "Deleted Sentence",
    "ratio_sen_with_del": "Ratio of Sentences\nwith Deletion",
    "ratio_sen_with_add": "Ratio of Sentences\nwith Addition",
    "ratio_sen_with_para": "Ratio of Sentences\nwith Paraphrasing",
    "ratio_sen_with_split": "Ratio of Sentences\nwith Split",
    "ratio_sen_with_deleted_sentence": "Ratio of\nDeleted Sentences",
    "mean_gpt2_surprisal": "Mean GPT-2\nSurprisal",
    "mean_pythia70m_surprisal": "Mean Pythia-70m\nSurprisal",
    "max_gpt2_surprisal": "Max GPT-2\nSurprisal",
    "max_pythia70m_surprisal": "Max Pythia-70m Surprisal",
    "mean_pythia70m_surprisal_adv": f"Surprisal\n({LEVEL_LABELS['Adv']}\nlevel)",
    "mean_gpt2_surprisal_adv": f"GPT-2 Surprisal ({LEVEL_LABELS['Adv']} level)",
    "word_length_adv": f"Length\n({LEVEL_LABELS['Adv']}\nlevel)",
    "wordFreq_frequency_adv": f"Frequency\n({LEVEL_LABELS['Adv']}\nlevel)",
    "diff_mean_pythia70m_surprisal": f"Surprisal\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "diff_wordFreq_frequency": f"Frequency\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "diff_word_length": f"Length\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "diff_sentence_length": f"N Words\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "multivar_diff": "Original\nComplexity",
    "multivar_adv": "Complexity\nReduction",
    "multivar_all": "Combined",
    'is_student': 'Student',
    'is_secondary': 'Secondary/High School',
    'is_undergrad': 'Undergraduate',
    'is_postgrad': 'Postgraduate',
    "all-MiniLM-L6-v2_cosine_similarity": "Cosine Similarity\n(all-MiniLM-L6-v2)",
    "all-MiniLM-L6-v2_manhattan_similarity": "Manhattan Similarity\n(all-MiniLM-L6-v2)",
    "cosine_similarity_tfidf": "Cosine Similarity\n(TF-IDF)",
    "jaccard_similarity": "Jaccard Similarity",
    "levenshtein_similarity": "Levenshtein Similarity",
    "fuzz_ratio": "Fuzzy\nRatio",
    "fuzz_partial_ratio": "Partial Fuzzy\nRatio",
    # Linguistic (original-level and difference variants)
    "syllable_count_textstats": "Syllable Count",
    "syllable_count_textstats_adv": f"Syllable Count\n({LEVEL_LABELS['Adv']} level)",
    "text_length_adv": f"N words\n({LEVEL_LABELS['Adv']} level)",
    "flesch_kincaid_grade_score_adv": f"Flesch-Kincaid Grade\n({LEVEL_LABELS['Adv']} level)",
    "diff_text_length": f"N words\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    # Word diff
    "ratio_added": "Ratio of\nAdded Words",
    "ratio_deleted": "Ratio of\nDeleted Words",
    "n_added_to_ele": "N Added Words",
    "n_deleted_from_adv": "N Deleted Words",
    "avg_change_ratio": "Avg Word\nChange Ratio",
    # Psycholinguistic (idea density + syntactic embedding depth)
    "cpidr_density": "CPIDR Idea Density",
    "depid_density": "DEPID Idea Density",
    "max_embedding_depth": "Max Embedding Depth",
    "avg_embedding_depth": "Avg Embedding Depth",
    "cpidr_density_adv": f"CPIDR Idea Density\n({LEVEL_LABELS['Adv']} level)",
    "depid_density_adv": f"DEPID Idea Density\n({LEVEL_LABELS['Adv']} level)",
    "max_embedding_depth_adv": f"Max Embedding Depth\n({LEVEL_LABELS['Adv']} level)",
    "avg_embedding_depth_adv": f"Avg Embedding Depth\n({LEVEL_LABELS['Adv']} level)",
    "diff_cpidr_density": f"CPIDR Idea Density\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "diff_depid_density": f"DEPID Idea Density\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "diff_max_embedding_depth": f"Max Embedding Depth\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    "diff_avg_embedding_depth": f"Avg Embedding Depth\nDiff: ({LEVEL_LABELS['Adv']} - {LEVEL_LABELS['Ele']})",
    # Item difficulty (comprehension accuracy per item; Acc.=accuracy, Hunt.=Hunting regime)
    "item_comp_L1": "Item Comp.\nAcc. (L1)",
    "item_comp_L1_adv": f"Item Comp. Acc.\n(L1, {LEVEL_LABELS['Adv']})",
    "item_comp_L1_adv_hunting": f"Item Comp. Acc.\n(L1, {LEVEL_LABELS['Adv']}, Hunt.)",
    "item_comp_L1_and_L2": "Item Comp.\nAcc. (L1+L2)",
    "item_comp_L1_and_L2_adv": f"Item Comp. Acc.\n(L1+L2, {LEVEL_LABELS['Adv']})",
    "item_comp_L1_and_L2_adv_hunting": f"Item Comp. Acc.\n(L1+L2, {LEVEL_LABELS['Adv']}, Hunt.)",
    "diff_item_comp_L1": "Item Comp. Acc.\nDiff (L1)",
    "diff_item_comp_L1_hunting": "Item Comp. Acc.\nDiff (L1, Hunt.)",
    "diff_item_comp_L1_and_L2": "Item Comp. Acc.\nDiff (L1+L2)",
    "diff_item_comp_L1_and_L2_hunting": "Item Comp. Acc.\nDiff (L1+L2, Hunt.)",
}

 # comprehension cols
BY_COMPREHENSION_COLS = ["comprehension_score", "comprehension_score_adv"]
# speed cols
BY_SPEED_COLS = ['TF_per_word', 'SaccT_per_word', 'P_RT_per_word', 'words_per_sec_based_TF', 'words_per_sec_based_SaccT', 'words_per_sec_based_P_RT']
BY_ADV_SPEED_COLS = [f"{speed_col}_adv" for speed_col in BY_SPEED_COLS]
BY_SPEED_COLS += BY_ADV_SPEED_COLS
# metadata cols
BY_METADATA_COLS = [
    "n_total_reading",
    "n_textbooks",
    "n_academic",
    "n_magazines",
    "n_newspapers",
    "n_email",
    "n_fiction",
    "n_nonfiction",
    "n_other_reading",
    "years_secondary",
    "years_undergrad",
    "years_postgrad",
    "years_education",
    "age",
    "is_student",
    "is_secondary",
    "is_undergrad",
    "is_postgrad",
    "edu_level_num"
]
# text cols
BY_CONTEXT_COLS = ["mean_gpt2_surprisal", "mean_pythia70m_surprisal"]
# BY_CONTEXT_COLS += ["max_gpt2_surprisal", "max_pythia70m_surprisal"]
BY_CONTEXT_COLS += [f"{col}_adv" for col in BY_CONTEXT_COLS]
BY_CONTEXT_COLS += ["diff_mean_pythia70m_surprisal"]
BY_LINGUISTIC_COLS = ["text_length", "word_length", "wordFreq_frequency", "flesch_kincaid_grade_score", "syllable_count_textstats"]
BY_LINGUISTIC_COLS += [f"{col}_adv" for col in BY_LINGUISTIC_COLS]
BY_LINGUISTIC_COLS += ["diff_wordFreq_frequency", "diff_word_length", "diff_text_length"]
BY_SIMPLIFICATION_COLS = ["deletion", "addition", "paraphrasing", "split", "deleted_sentence"]
# Per-text ratio versions: count of sentences with the operation / n sentences in the text
# (denominator = all Adv sentences, incl. unaligned/deleted rows). Built in process_results.py.
BY_SIMPLIFICATION_COLS += [
    "ratio_sen_with_del", "ratio_sen_with_add", "ratio_sen_with_para",
    "ratio_sen_with_split", "ratio_sen_with_deleted_sentence",
]
BY_WORD_DIFF_COLS = ["ratio_added", "ratio_deleted", "n_added_to_ele", "n_deleted_from_adv", "avg_change_ratio"]
BY_SIMILARITY_COLS = [
    "fuzz_partial_ratio", "fuzz_ratio",
    "all-MiniLM-L6-v2_cosine_similarity", "all-MiniLM-L6-v2_manhattan_similarity",
    "cosine_similarity_tfidf", "jaccard_similarity",
    "levenshtein_similarity",
]
BY_ITEM_DIFFICULTY_COLS = [
    "item_comp_L1", "item_comp_L1_adv", "item_comp_L1_adv_hunting",
    "item_comp_L1_and_L2", "item_comp_L1_and_L2_adv", "item_comp_L1_and_L2_adv_hunting",
    "diff_item_comp_L1", "diff_item_comp_L1_hunting",
    "diff_item_comp_L1_and_L2", "diff_item_comp_L1_and_L2_hunting",
]
BY_PSYCHOLINGUISTIC_COLS = ["cpidr_density", "depid_density", "max_embedding_depth", "avg_embedding_depth"]
BY_PSYCHOLINGUISTIC_COLS += [f"{col}_adv" for col in BY_PSYCHOLINGUISTIC_COLS]
BY_PSYCHOLINGUISTIC_COLS += [f"diff_{col}" for col in ["cpidr_density", "depid_density", "max_embedding_depth", "avg_embedding_depth"]]

def _get_by_col_type(by_col):
    if by_col in BY_COMPREHENSION_COLS:
        by_col_type = "comprehension"
    elif by_col in BY_METADATA_COLS:
        by_col_type = "metadata"
    elif by_col in BY_SPEED_COLS:
        by_col_type = "speed"
    elif by_col in BY_LINGUISTIC_COLS:
        by_col_type = "linguistic"
    elif by_col in BY_SIMPLIFICATION_COLS:
        by_col_type = "simplification"
    elif by_col in BY_WORD_DIFF_COLS:
        by_col_type = "word_diff"
    elif by_col in BY_SIMILARITY_COLS:
        by_col_type = "similarity"
    elif by_col in BY_ITEM_DIFFICULTY_COLS:
        by_col_type = "item_difficulty"
    elif by_col in BY_PSYCHOLINGUISTIC_COLS:
        by_col_type = "psycholinguistic"
    elif by_col in BY_CONTEXT_COLS:
        by_col_type = "context"
    elif 'multivar' in by_col:
        by_col_type = "multivar"
    else:
        raise ValueError(f"Invalid by_col: {by_col}")
    return by_col_type

def _get_per_type(per):
    if per == "subject":
        per_type = "individual"
    elif per == "text":
        per_type = "text"
    else:
        raise ValueError(f"Invalid per: {per}")
    return per_type


BY_COL_TYPE_LABELS = {
    "comprehension": "Reading Comprehension",
    "metadata": "Participant Metadata",
    "speed": "Participant Reading Speed",
    "linguistic": "Linguistic",
    "simplification": "Text Simplification",
    "context": "Linguistic"
}

def _get_label_linguistic_col(col):
    # replace surp len freq
    col = col.replace('_', ' ')
    full_name = col
    full_name = full_name.replace('surp', 'Surprisal')
    full_name = full_name.replace('len', 'Length')
    full_name = full_name.replace('freq', 'Frequency')
    if len(full_name) > 15:
        short_name = col
        short_name = short_name.replace('surp', 'Surp')
        short_name = short_name.replace('len', 'Len')
        short_name = short_name.replace('freq', 'Freq')
        return short_name
    else:
        return full_name
    
LEXTALE_BIN_COLORS = {'0_60': '#C11007', '60_70': '#FF692A', '70_80': '#FFD230', '80_90': '#5EA529', '90_100': '#016630'}
ADV_COMP_BIN_COLORS = {'0*0_0*6': '#C11007', '0*6_0*7': '#FF692A', '0*7_0*8': '#FFD230', '0*8_0*9': '#5EA529', '0*9_1*0': '#016630'}

LEXTALE_BIN_LABELS = {'0_60': '0-60', '60_70': '60-70', '70_80': '70-80', '80_90': '80-90', '90_100': '90-100'}
ADV_COMP_BIN_LABELS = {'0*0_0*6': '0-60', '0*6_0*7': '60-70', '0*7_0*8': '70-80', '0*8_0*9': '80-90', '0*9_1*0': '90-100'}

BIN_TO_CODE = {
    '0_60': 30,
    '60_70': 65,
    '70_80': 75,
    '80_90': 85,
    '90_100': 95,
    '0*0_0*6': 30,
    '0*6_0*7': 65 ,
    '0*7_0*8': 75,
    '0*8_0*9': 85,
    '0*9_1*0': 95
}

CODES = [30, 65, 75, 85, 95]