import pandas as pd
from pathlib import Path
import string
from wordfreq import word_frequency
from src.utils.textual_utils import count_sentences, split_into_sentences
import numpy as np
from scipy.stats import sem, t, ttest_rel, ttest_ind

from src.constants import EYE_BY_WORD_ALIGNED_PATHS
EYE_BY_WORD_DF_L1_ALIGNED_PATH = EYE_BY_WORD_ALIGNED_PATHS['L1']

# ----------------------------- std and 95% CI stats -----------------------------


# Compute sample standard deviations and 95% confidence intervals for the means
def mean_std_ci(data, confidence=0.95):
    clean = pd.Series(data).dropna()
    n = len(clean)
    m = np.mean(clean)
    if n <= 1:
        return m, np.nan, np.nan
    std = np.std(clean, ddof=1)
    ci95 = sem(clean) * t.ppf((1 + confidence) / 2, n - 1)
    return m, std, ci95


def pval_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

# ------------------------ functions to compute stats (paired) ---------------


def paired_ttest_from_data(data, text_id_col, level_col, value, label):
    # Step 1: Pivot to align matching paragraphs (or documents) by their align_index
    pivoted = data.pivot(index=text_id_col, columns=level_col, values=value)

    # Step 2: Drop rows with any missing values (in case there are unmatched pairs)
    pivoted = pivoted.dropna(subset=["Adv", "Ele"])

    # Step 3: Extract aligned arrays for Adv and Ele
    adv_vals = pivoted["Adv"]
    ele_vals = pivoted["Ele"]

    # Step 4: Perform paired t-test
    return perform_paired_ttest(adv_vals, ele_vals, label=label)

def perform_paired_ttest(adv_vals, ele_vals, label):
    stat, pval = ttest_rel(adv_vals, ele_vals)
    mean_adv, std_adv, ci95_adv = mean_std_ci(adv_vals)
    mean_ele, std_ele, ci95_ele = mean_std_ci(ele_vals)
    stars = pval_stars(pval)

    return {
        "mean_adv":mean_adv,
        "std_adv":std_adv,
        "ci95_adv":ci95_adv,
        "mean_ele":mean_ele,
        "std_ele":std_ele,
        "ci95_ele":ci95_ele,
        "pval":pval,
        "stars":stars
    }

# ------------------------ functions to compute stats (unpaired) ---------------
def extract_sent_lens(text):
    sentences = split_into_sentences(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    return lengths

def get_sentence_lens_for_level(df, level):
    lengths_all = []
    for text in df[df['level'] == level]['text']:
        lengths = extract_sent_lens(text)
        lengths_all.extend(lengths)

    return np.array(lengths_all)

def unpaired_ttest(data_adv, data_ele, label):

    stat, pval = ttest_ind(data_adv, data_ele)
    mean_adv, std_adv, ci95_adv = mean_std_ci(data_adv)
    mean_ele, std_ele, ci95_ele = mean_std_ci(data_ele)
    stars = pval_stars(pval)

    return {
        "mean_adv":mean_adv,
        "std_adv":std_adv,
        "ci95_adv":ci95_adv,
        "mean_ele":mean_ele,
        "std_ele":std_ele,
        "ci95_ele":ci95_ele,
        "pval":pval,
        "stars":stars
    }


# -------------------- load paragraph metrics data -------------------
src_path = Path.cwd().parents[0]
parag_metrics_path = src_path / Path("Readability/src/readability_metrics/data/paragraphs_metrics_cleaned.csv")

data_parags = pd.read_csv(parag_metrics_path)

# --------------------- extract stats sentences and paragraphs -------------------
data_parags['n_sentences'] = data_parags['text'].apply(count_sentences)

data_parags['n_words_no_punct'] = data_parags['text'].apply(lambda x: len(''.join(char for char in x if char not in string.punctuation).split()))
data_parags['n_words'] = data_parags['text'].apply(lambda x: len((x).split()))

data_parags['sum_word_length'] = data_parags['text'].apply(lambda x: sum([len(word.strip(string.punctuation)) for word in x.split()]))
data_parags['sum_raw_wordfreq'] = data_parags['text'].apply(lambda x: sum([word_frequency(word, lang="en") * 1e4 for word in x.split()]))


# ----------------------- per paragraph stats (paired) ---------------------------
# run paired t-test on number of sentences per paragraph
number_of_sentences_per_parag_dict = paired_ttest_from_data(data_parags, text_id_col="text_id", level_col="level", value="n_sentences", label="sentences per paragraph")
# run paired t-test on number of words per paragraph
number_of_words_per_parag_dict = paired_ttest_from_data(data_parags, text_id_col="text_id", level_col="level", value="n_words", label="words per paragraph")


# ----------------------- per sentence stats (unpaired) ---------------------------
sentence_lengths_adv = get_sentence_lens_for_level(data_parags, level="Adv")
sentence_lengths_ele = get_sentence_lens_for_level(data_parags, level="Ele")

sentence_length_dict = unpaired_ttest(sentence_lengths_adv, sentence_lengths_ele, label="sentence length")

# ----------------------- extract word length, word frequency and surprisal stats (unpaired) -------------------

# -------------------- load eye-tracking data -------------------
eye_data = pd.read_csv(EYE_BY_WORD_DF_L1_ALIGNED_PATH)
eye_data_one_version = eye_data.drop_duplicates(subset=["unique_paragraph_id", "level", "text_spacing_version","gpt2_surprisal","pythia70m_surprisal","word_length", "word_length_no_punctuation", "wordfreq_frequency", "text_spacing_version", "IA_LABEL"])
eye_data_one_version = eye_data_one_version[eye_data_one_version["text_spacing_version"]==0]

agg_eye = eye_data_one_version.groupby(['text_id', 'level']).agg({
    'wordfreq_frequency': 'mean',
    'word_length': 'mean',
    'gpt2_surprisal': 'mean',
    'pythia70m_surprisal': 'mean'
}).reset_index()

# calc unpaired t test on word length, word frequency and surprisal using eye_data_one_version:

word_length_adv = eye_data_one_version[eye_data_one_version['level'] == 'Adv']['word_length']
word_length_ele = eye_data_one_version[eye_data_one_version['level'] == 'Ele']['word_length']
word_length_dict = unpaired_ttest(word_length_adv, word_length_ele, label="word length")

wordfreq_adv = eye_data_one_version[eye_data_one_version['level'] == 'Adv']['wordfreq_frequency']
wordfreq_ele = eye_data_one_version[eye_data_one_version['level'] == 'Ele']['wordfreq_frequency']
word_freq_dict = unpaired_ttest(wordfreq_adv, wordfreq_ele, label="word frequency")

pythia_surprisal_adv = eye_data_one_version[eye_data_one_version['level'] == 'Adv']['pythia70m_surprisal']
pythia_surprisal_ele = eye_data_one_version[eye_data_one_version['level'] == 'Ele']['pythia70m_surprisal']
pythia70m_surprisal_dict = unpaired_ttest(pythia_surprisal_adv, pythia_surprisal_ele, label="surprisal")


########################################################3
# add number of paragraphs and sentences per level to the results (unpaired since its overall, not per paragraph)
#######################################################


# calculate number of paragraphs per level
num_parags_adv = len(data_parags[data_parags['level'] == 'Adv'])
num_parags_ele = len(data_parags[data_parags['level'] == 'Ele'])

# calculate number of sentences per level
num_sentences_adv = data_parags[data_parags['level'] == 'Adv']['n_sentences'].sum()
num_sentences_ele = data_parags[data_parags['level'] == 'Ele']['n_sentences'].sum()

rows = []

rows.insert(0, {
    "metric": "number_of_paragraphs",
    "group": "original",
    "n": num_parags_adv,
    "mean": float(num_parags_adv),
    "std": "",
    "ci95": "",
    "mean_std": f"{num_parags_adv}",
    "mean_ci": f"{num_parags_adv}",
    "pval": "NA",
    "stars": "NA"
})
rows.insert(1, {
    "metric": "number_of_paragraphs",
    "group": "simplified",
    "n": num_parags_ele,
    "mean": float(num_parags_ele),
    "std": "",
    "ci95": "",
    "mean_std": f"{num_parags_ele}",
    "mean_ci": f"{num_parags_ele}",
    "pval": "NA",
    "stars": "NA"
})
rows.insert(2, {
    "metric": "number_of_sentences",
    "group": "original",
    "n": num_parags_adv,
    "mean": float(num_sentences_adv),
    "std": "",
    "ci95": "",
    "mean_std": f"{num_sentences_adv}",
    "mean_ci": f"{num_sentences_adv}",
    "pval": "NA",
    "stars": "NA"
})
rows.insert(3, {
    "metric": "number_of_sentences",
    "group": "simplified",
    "n": num_parags_ele,
    "mean": float(num_sentences_ele),
    "std": "",
    "ci95": "",
    "mean_std": f"{num_sentences_ele}",
    "mean_ci": f"{num_sentences_ele}",
    "pval": "NA",
    "stars": "NA"
})


# create csv of results
all_relevant_dicts = {
    "Sentences per passage": number_of_sentences_per_parag_dict,
    "Words per passage": number_of_words_per_parag_dict,
    "Sentence length (words)": sentence_length_dict,
    "Word length (characters)": word_length_dict,
    "Word frequency (Wordfreq)": word_freq_dict,
    "Word surprisal (Pythia-70m)": pythia70m_surprisal_dict
}



# --- Reformat and write CSV in requested format ---
group_map = {"Adv": "original", "Ele": "simplified"}
metric_map = {
    "Sentences per passage": "sentences_per_passage",
    "Words per passage": "words_per_passage",
    "Sentence length (words)": "sentence_length_words",
    "Word length (characters)": "word_length_characters",
    "Word frequency (Wordfreq)": "word_frequency_wordfreq",
    "Word surprisal (Pythia-70m)": "word_surprisal_pythia70m"
}

for metric, stats in all_relevant_dicts.items():
    for group, group_label in zip(["mean_adv", "mean_ele"], ["original", "simplified"]):
        n = None
        if metric in ["Sentences per passage", "Words per passage"]:
            n = int(stats["mean_adv"] if group == "mean_adv" else stats["mean_ele"])
        elif metric == "Sentence length (words)":
            n = int(stats["mean_adv"] if group == "mean_adv" else stats["mean_ele"])
        else:
            n = None
        mean = stats[group]
        std = stats["std_adv"] if group == "mean_adv" else stats["std_ele"]
        ci95 = stats["ci95_adv"] if group == "mean_adv" else stats["ci95_ele"]
        mean_std = f"{mean:.2f} ± {std:.2f}" if std != 0 else f"{mean:.2f}"
        mean_ci = f"{mean:.2f} ± {ci95:.2f}" if ci95 != 0 else f"{mean:.2f}"
        rows.append({
            "metric": metric_map[metric],
            "group": group_label,
            "n": n,
            "mean": mean,
            "std": std,
            "ci95": ci95,
            "mean_std": mean_std,
            "mean_ci": mean_ci,
            "pval": stats["pval"],
            "stars": stats["stars"]
        })

results_df = pd.DataFrame(rows)
out_dir = src_path / Path("Readability/src/data/OneStop")
results_df.to_csv(out_dir / "OneStop_mean_ci_stats.csv", index=False)
results_df.to_csv(out_dir / "OneStop_mean_std_stats.csv", index=False)
