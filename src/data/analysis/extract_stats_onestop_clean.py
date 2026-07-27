import pandas as pd
from pathlib import Path
import string
from wordfreq import word_frequency
from src.utils.textual_utils import count_sentences, split_into_sentences
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, sem, t
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.constants import EYE_BY_WORD_ALIGNED_PATHS
EYE_BY_WORD_DF_L1_ALIGNED_PATH = EYE_BY_WORD_ALIGNED_PATHS['L1']

# ------------------------ functions to compute stats (paired) ---------------


# Compute 95% confidence intervals for the means
def mean_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return m, h  # Return mean and margin of error


def pval_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

def paired_ttest_from_data(data, text_id_col, level_col, value, label):
    # Step 1: Pivot to align matching paragraphs (or documents) by their align_index
    pivoted = data.pivot(index=text_id_col, columns=level_col, values=value)

    # Step 2: Drop rows with any missing values (in case there are unmatched pairs)
    pivoted = pivoted.dropna(subset=["Adv", "Ele"])

    # Step 3: Extract aligned arrays for Adv and Ele
    adv_vals = pivoted["Adv"]
    ele_vals = pivoted["Ele"]

    # Step 4: Perform paired t-test
    perform_paired_ttest(adv_vals, ele_vals, label=label)

def perform_paired_ttest(adv_vals, ele_vals, label):
    stat, pval = ttest_rel(adv_vals, ele_vals)
    stat, pval_un = ttest_ind(adv_vals, ele_vals)
    mean_adv, moe_adv = mean_ci(adv_vals)
    mean_ele, moe_ele = mean_ci(ele_vals)
    print(f"{label} - Adv mean: {mean_adv:.2f} ± {moe_adv:.2f} (95% CI)")
    print(f"{label} - Ele mean: {mean_ele:.2f} ± {moe_ele:.2f} (95% CI)")
    print(f"{label} - Paired t-test p = {pval:.4f} {pval_stars(pval)}")
    # print(f"{label} - unPaired t-test p = {pval_un:.4f} {pval_stars(pval)}")


# ------------------------ functions to compute stats (unpaired) ---------------
def extract_sent_lens(text):
    sentences = split_into_sentences(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    return lengths

def extract_word_lens(text):
    lengths = [len(word) for word in text.split()]
    return lengths

def extract_wordfreqs(text):
    freqs = [-np.log2(word_frequency(word, lang="en", minimum=1e-11)) for word in text.split()]
    return freqs

def get_sentence_lens_for_level(df, level):
    lengths_all = []
    for text in df[df['level'] == level]['text']:
        lengths = extract_sent_lens(text)
        lengths_all.extend(lengths)

    return np.array(lengths_all)

def get_word_lens_for_level(df, level):
    lengths_all = []
    for text in df[df['level'] == level]['text']:
        lengths = extract_word_lens(text)
        lengths_all.extend(lengths)

    return np.array(lengths_all)

def get_wordfreqs_for_level(df, level):
    freqs_all = []
    for text in df[df['level'] == level]['text']:
        freqs = extract_wordfreqs(text)
        freqs_all.extend(freqs)

    return np.array(freqs_all)

def unpaired_ttest(data_adv, data_ele, label):

    stat, pval = ttest_ind(data_adv, data_ele)
    mean_adv, moe_adv = mean_ci(data_adv)
    mean_ele, moe_ele = mean_ci(data_ele)

    print(f"{label} - Adv mean: {mean_adv:.2f} ± {moe_adv:.2f} (95% CI)")
    print(f"{label} - Ele mean: {mean_ele:.2f} ± {moe_ele:.2f} (95% CI)")
    print(f"{label} - Unpaired t-test p = {pval:.4f} {pval_stars(pval)}")


# -------------------- load paragraph metrics data -------------------
src_path = Path.cwd().parents[0]
parag_metrics_path = src_path / Path("Readability/src/readability_metrics/data/paragraphs_metrics_cleaned.csv")

data_parags = pd.read_csv(parag_metrics_path)


#####################################################################################################
# ------------------------- compute paired t tests ------------------
# --------------------(first average in each paragraph, then compare) ---------------------
#############################################################################################################


# --------------------- extract stats number of sentences per paragraph -------------------
data_parags['n_sentences'] = data_parags['text'].apply(count_sentences)

data_parags['n_words_no_punct'] = data_parags['text'].apply(lambda x: len(''.join(char for char in x if char not in string.punctuation).split()))
data_parags['n_words'] = data_parags['text'].apply(lambda x: len((x).split()))

data_parags['sum_word_length'] = data_parags['text'].apply(lambda x: sum([len(word.strip(string.punctuation)) for word in x.split()]))
data_parags['sum_raw_wordfreq'] = data_parags['text'].apply(lambda x: sum([word_frequency(word, lang="en") * 1e4 for word in x.split()]))

print()
print("number of sentences per paragraph (data_parags):")
# run paired t-test on number of sentences per paragraph
paired_ttest_from_data(data_parags, text_id_col="text_id", level_col="level", value="n_sentences", label="sentences per paragraph")
print()
# ---------------------- extract stats number of words per paragraph -------------------
print()
print("number of words per paragraph (data_parags):")
# run paired t-test on number of words per paragraph
paired_ttest_from_data(data_parags, text_id_col="text_id", level_col="level", value="n_words", label="words per paragraph")


############################################
# special case: sentence length
############################################

# ----------------------- extract sentence length stats -------------------
# sentence length is a bit trickier since we cant really do it paired - its average sentence length overall, not per paragraph, so we need to calculate seperatly 
# i think the paired should be paired per sentences - but this is tricky cause of the alignment - chceck what we decided to do last time

# for now - unpaired t test on the total average 
print()
print("sentence length (data_parags):")
# run paired t-test on sentence length
sentence_lengths_adv = get_sentence_lens_for_level(data_parags, level="Adv")
sentence_lengths_ele = get_sentence_lens_for_level(data_parags, level="Ele")

unpaired_ttest(sentence_lengths_adv, sentence_lengths_ele, label="sentence length")


############################################
# calculate unpaired (word length, word frequency, surprisal) - using data_parags instead of eye_data_one_version 
############################################

# ----- word length according to text ---------------

word_length_adv = get_word_lens_for_level(data_parags, level="Adv")
word_length_ele = get_word_lens_for_level(data_parags, level="Ele")

unpaired_ttest(word_length_adv, word_length_ele, label="word length")


# # ------ word frequency according to text ---------------

wordfreq_adv = get_wordfreqs_for_level(data_parags, level="Adv")
wordfreq_ele = get_wordfreqs_for_level(data_parags, level="Ele")
unpaired_ttest(wordfreq_adv, wordfreq_ele, label="word frequency according to text")

# surprisal is trickier to calculate since we need model for it, so using eye_data is good. validated both word length and frequency 
# with eye data against the text data and they are consistent, so we can be confident in the results for surprisal using eye data.

#############################################
# general stats
############################################

# calculate number of paragraphs per level
num_parags_adv = len(data_parags[data_parags['level'] == 'Adv'])
num_parags_ele = len(data_parags[data_parags['level'] == 'Ele'])
print(f"number of paragraphs - Adv: {num_parags_adv}, Ele: {num_parags_ele}")

# calculate number of sentences per level
num_sentences_adv = data_parags[data_parags['level'] == 'Adv']['n_sentences'].sum()
num_sentences_ele = data_parags[data_parags['level'] == 'Ele']['n_sentences'].sum()
print(f"number of sentences - Adv: {num_sentences_adv}, Ele: {num_sentences_ele}")

############################################
# calculate unpaired (word length, word frequency, surprisal) with eye_data_one_version
############################################

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
unpaired_ttest(word_length_adv, word_length_ele, label="word length")

wordfreq_adv = eye_data_one_version[eye_data_one_version['level'] == 'Adv']['wordfreq_frequency']
wordfreq_ele = eye_data_one_version[eye_data_one_version['level'] == 'Ele']['wordfreq_frequency']
unpaired_ttest(wordfreq_adv, wordfreq_ele, label="word frequency")

pythia_surprisal_adv = eye_data_one_version[eye_data_one_version['level'] == 'Adv']['pythia70m_surprisal']
pythia_surprisal_ele = eye_data_one_version[eye_data_one_version['level'] == 'Ele']['pythia70m_surprisal']
unpaired_ttest(pythia_surprisal_adv, pythia_surprisal_ele, label="surprisal")

# calc paired t test on word length, word frequency and surprisal using eye_data_one_version:

# ---------------------- extract word length stats (paired) -------------------
print()
print("word length (data_parags):")
# run paired t-test on word length
data_parags['avg_word_len'] = data_parags['sum_word_length'] / data_parags['n_words']
paired_ttest_from_data(data_parags, text_id_col="text_id", level_col="level", value="avg_word_len", label="word length")


# ---------------------- extract word frequency stats (paired) -------------------
# need to use eye df for this one since we need to calculate word frequency per word and then average, not sum
print()
print("word frequency (eye_data_one_version):")
# run paired t-test on word frequency
paired_ttest_from_data(agg_eye, text_id_col="text_id", level_col="level", value="wordfreq_frequency", label="word frequency")

# ---------------------- extract surprisal stats (paired) -------------------
print()
print("surprisal (eye_data_one_version):")
# run paired t-test on surprisal
paired_ttest_from_data(agg_eye, text_id_col="text_id", level_col="level", value="pythia70m_surprisal", label="surprisal")

