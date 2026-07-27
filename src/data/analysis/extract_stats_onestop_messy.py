import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk
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


# Download tokenizer (if not already done)
nltk.download('punkt')


src_path = Path.cwd().parents[0]
parag_metrics_path = src_path / Path("Readability/src/readability_metrics/data/paragraphs_metrics_cleaned.csv")

data_parags = pd.read_csv(parag_metrics_path)


# trying counting sentences with nltk
data_parags['sentence_count_nltk'] = data_parags['text'].apply(lambda x: len(sent_tokenize(x)))
data_parags['n_sentences'] = data_parags['text'].apply(count_sentences)

data_parags['n_words_no_punct'] = data_parags['text'].apply(lambda x: len(''.join(char for char in x if char not in string.punctuation).split()))
data_parags['n_words'] = data_parags['text'].apply(lambda x: len((x).split()))

data_parags['sum_word_length'] = data_parags['text'].apply(lambda x: sum([len(word.strip(string.punctuation)) for word in x.split()]))
data_parags['sum_raw_wordfreq'] = data_parags['text'].apply(lambda x: sum([word_frequency(word, lang="en") * 1e4 for word in x.split()]))


print(f"number of sentences per paragraph adv:{data_parags[data_parags['level'] == 'Adv']["n_sentences"].sum()/len(data_parags[data_parags['level'] == 'Adv'])}")
print(f"number of sentences per paragraph ele:{data_parags[data_parags['level'] == 'Ele']["n_sentences"].sum()/len(data_parags[data_parags['level'] == 'Ele'])}")

# paired t test for number of sentences

# Step 1: Pivot to align matching paragraphs (or documents) by their align_index
pivoted = data_parags.pivot(index="text_id", columns="level", values="n_sentences")

# Step 2: Drop rows with any missing values (in case there are unmatched pairs)
pivoted = pivoted.dropna(subset=["Adv", "Ele"])

# Step 3: Extract aligned arrays for Adv and Ele
adv_vals = pivoted["Adv"]
ele_vals = pivoted["Ele"]

# Step 4: Perform Paired t-test (within the same documents)
stat_paired, pval_paired = ttest_rel(adv_vals, ele_vals)

# Step 5: Perform Regular (Unpaired) t-test (treating the groups as independent)
stat_unpaired, pval_unpaired = ttest_ind(adv_vals, ele_vals)

# Step 6: Compute means
mean_adv = adv_vals.mean()
mean_ele = ele_vals.mean()

# Step 7: Compute 95% confidence intervals for the means
def mean_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return m - h, m + h

ci_adv = mean_ci(adv_vals)
ci_ele = mean_ci(ele_vals)

# Step 8: Optional p-value stars (for both t-tests)
def pval_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# Final printout for both tests
print(f"Avg sentences per paragraph (Adv): {mean_adv:.2f} +- {mean_adv-ci_adv[0]:.2f} (95% CI)")
print(f"Avg sentences per paragraph (Ele): {mean_ele:.2f} +- {mean_ele-ci_ele[0]:.2f} (95% CI)")
print(f"Paired t-test p = {pval_paired:.4f} {pval_stars(pval_paired)}")
print(f"Unpaired t-test p = {pval_unpaired:.4f} {pval_stars(pval_unpaired)}")

# other statisticssss 

# Helper function for confidence intervals
def mean_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return m, h  # Return mean and margin of error

# Perform paired t-test and CI for each statistic
def perform_paired_ttest(adv_vals, ele_vals, label):
    stat, pval = ttest_rel(adv_vals, ele_vals)
    stat, pval_un = ttest_ind(adv_vals, ele_vals)
    mean_adv, moe_adv = mean_ci(adv_vals)
    mean_ele, moe_ele = mean_ci(ele_vals)
    print(f"{label} - Adv mean: {mean_adv:.2f} ± {moe_adv:.2f} (95% CI)")
    print(f"{label} - Ele mean: {mean_ele:.2f} ± {moe_ele:.2f} (95% CI)")
    print(f"{label} - Paired t-test p = {pval:.4f} {pval_stars(pval)}")
    print(f"{label} - unPaired t-test p = {pval_un:.4f} {pval_stars(pval)}")

# Function to add stars based on p-value
def pval_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# Group by 'level' and calculate the averages for each statistic
averages = data_parags.groupby('level')[['sentence_count', 'text_length', 'word_length', 'n_words', 
                                         'wordFreq_frequency', 'flesch_kincaid_grade_score', 
                                         'smog_index', 'sentence_count_nltk', 'raw_wordFreq_frequency', ]].mean()

print("now added")
for level, row in averages.iterrows():
    print(f"{level}:")
    for column, value in row.items():
        print(f"{column}: {value}")
    print() 


# Calculate individual word length and word frequency for Adv and Ele
word_length_adv = data_parags[data_parags['level'] == 'Adv']['sum_word_length'].sum() / data_parags[data_parags['level'] == 'Adv']['n_words'].sum()
word_length_ele = data_parags[data_parags['level'] == 'Ele']['sum_word_length'].sum() / data_parags[data_parags['level'] == 'Ele']['n_words'].sum()

word_freq_adv = data_parags[data_parags['level'] == 'Adv']['sum_raw_wordfreq'].sum() / data_parags[data_parags['level'] == 'Adv']['n_words'].sum()
word_freq_ele = data_parags[data_parags['level'] == 'Ele']['sum_raw_wordfreq'].sum() / data_parags[data_parags['level'] == 'Ele']['n_words'].sum()

print("-----------------------------------------------------")
print("counting words including punctuations (n_words is text.split meaning that stand alone punctuations count)")

# Print word count for each level
print(f"total words adv: {data_parags[data_parags['level'] == 'Adv']['n_words'].sum()}")
print(f"total words ele: {data_parags[data_parags['level'] == 'Ele']['n_words'].sum()}")

# Sentence length (words)
print(f"sentence length (words) adv: {data_parags[data_parags['level'] == 'Adv']['n_words'].sum() / data_parags[data_parags['level'] == 'Adv']['n_sentences'].sum()}")
print(f"sentence length (words) ele: {data_parags[data_parags['level'] == 'Ele']['n_words'].sum() / data_parags[data_parags['level'] == 'Ele']['n_sentences'].sum()}")


data_parags['avg_sent_len'] = data_parags['n_words']/data_parags['n_sentences']
# Sentence length 2 agg (words)
print(f"sentence length (words) 2 agg adv: {data_parags[data_parags['level'] == 'Adv']['avg_sent_len'].sum() / len(data_parags[data_parags['level'] == 'Adv'])}")
print(f"sentence length (words) 2 agg ele: {data_parags[data_parags['level'] == 'Ele']['avg_sent_len'].sum() / len(data_parags[data_parags['level'] == 'Ele'])}")


# Print calculated word frequency and word length
print(f"word freq advanced: {word_freq_adv}")
print(f"word freq elementary: {word_freq_ele}")
print(f"word length advanced: {word_length_adv}")
print(f"word length elementary: {word_length_ele}")


def compute_mean_and_ci(numerator_col, denominator_col, data, level):
    # Create per-word values
    subset = data[data['level'] == level]
    per_word_values = subset[numerator_col] / subset[denominator_col]

    mean = per_word_values.mean()
    se = stats.sem(per_word_values, nan_policy='omit')
    ci_range = stats.t.ppf(0.975, len(per_word_values) - 1) * se  # 95% CI

    return mean, ci_range

# Word length (sum_word_length / n_words)
wl_adv_mean, wl_adv_ci = compute_mean_and_ci('sum_word_length', 'n_words', data_parags, 'Adv')
wl_ele_mean, wl_ele_ci = compute_mean_and_ci('sum_word_length', 'n_words', data_parags, 'Ele')

# Word frequency (sum_raw_wordfreq / n_words)
wf_adv_mean, wf_adv_ci = compute_mean_and_ci('sum_raw_wordfreq', 'n_words', data_parags, 'Adv')
wf_ele_mean, wf_ele_ci = compute_mean_and_ci('sum_raw_wordfreq', 'n_words', data_parags, 'Ele')

# Print nicely
print("means and ci's for word measures")
print(f"Word length (Adv): {wl_adv_mean:.3f} ± {wl_adv_ci:.2f} (95% CI)")
print(f"Word length (Ele): {wl_ele_mean:.3f} ± {wl_ele_ci:.2f} (95% CI)")

print(f"Word frequency (Adv): {wf_adv_mean:.3f} ± {wf_adv_ci:.2f} (95% CI)")
print(f"Word frequency (Ele): {wf_ele_mean:.3f} ± {wf_ele_ci:.2f} (95% CI)")


# Words per paragraph
print(f"words per paragraph adv = {data_parags[data_parags['level'] == 'Adv']['n_words'].sum() / len(data_parags[data_parags['level'] == 'Adv'])}")
print(f"words per paragraph ele = {data_parags[data_parags['level'] == 'Ele']['n_words'].sum() / len(data_parags[data_parags['level'] == 'Ele'])}")

# ---- Calculations for paired t-tests ----
# Prepare data for t-test
pivoted_words = data_parags.pivot(index="text_id", columns="level", values="n_words")
pivoted_sentences = data_parags.pivot(index="text_id", columns="level", values="n_sentences")
pivoted_word_length = data_parags.pivot(index="text_id", columns="level", values="sum_word_length")
pivoted_word_freq = data_parags.pivot(index="text_id", columns="level", values="sum_raw_wordfreq")
pivoted_wordfreq_frequency = data_parags.pivot(index="text_id", columns="level", values="wordFreq_frequency")
pivoted_avg_surprisal = data_parags.pivot(index="text_id", columns="level", values="mean_surprisal_pythia")

# Drop rows with any missing values for matched pairs
pivoted_words = pivoted_words.dropna(subset=["Adv", "Ele"])
pivoted_sentences = pivoted_sentences.dropna(subset=["Adv", "Ele"])
pivoted_word_length = pivoted_word_length.dropna(subset=["Adv", "Ele"])
pivoted_word_freq = pivoted_word_freq.dropna(subset=["Adv", "Ele"])
pivoted_wordfreq_frequency = pivoted_wordfreq_frequency.dropna(subset=["Adv", "Ele"])
pivoted_avg_surprisal = pivoted_avg_surprisal.dropna(subset=["Adv", "Ele"])


# Extract aligned values for comparison
adv_words = pivoted_words["Adv"]
ele_words = pivoted_words["Ele"]
adv_sentences = pivoted_sentences["Adv"]
ele_sentences = pivoted_sentences["Ele"]
adv_word_length = pivoted_word_length["Adv"]
ele_word_length = pivoted_word_length["Ele"]
adv_word_freq = pivoted_word_freq["Adv"]
ele_word_freq = pivoted_word_freq["Ele"]
adv_wordfreq_frequency = pivoted_wordfreq_frequency["Adv"]
ele_wordfreq_frequency = pivoted_wordfreq_frequency["Ele"]
adv_pythia_surprisal = pivoted_avg_surprisal["Adv"]
ele_pythia_surprisal = pivoted_avg_surprisal["Ele"]

# Perform paired t-tests and calculate CIs for each statistic
print("-----------------------------------------------------")
print("Paired t-tests and CIs:")

# Perform and print results
perform_paired_ttest(adv_words, ele_words, 'Words per Paragraph')
perform_paired_ttest(adv_sentences, ele_sentences, 'Sentence per paragraph')
perform_paired_ttest(adv_wordfreq_frequency, ele_wordfreq_frequency, 'Word Frequency (sum_raw_wordfreq) jjjrj')
perform_paired_ttest(adv_pythia_surprisal, ele_pythia_surprisal, 'Avg Surprisal (pythia)')
print("------------------------------------------------------------------------------")


#counting words including punctuations
averages = data_parags.groupby('level')[['sentence_count', 'text_length', 'word_length','n_words', 
                                'wordFreq_frequency', 'flesch_kincaid_grade_score', 'smog_index', 'sentence_count_nltk', 'raw_wordFreq_frequency']].mean()
word_length_adv = data_parags[data_parags['level'] == 'Adv']['sum_word_length'].sum()/data_parags[data_parags['level'] == 'Adv']['n_words'].sum()
word_length_ele = data_parags[data_parags['level'] == 'Ele']['sum_word_length'].sum()/data_parags[data_parags['level'] == 'Ele']['n_words'].sum()
word_freq_adv = data_parags[data_parags['level'] == 'Adv']['sum_raw_wordfreq'].sum()/data_parags[data_parags['level'] == 'Adv']['n_words'].sum()
word_freq_ele = data_parags[data_parags['level'] == 'Ele']['sum_raw_wordfreq'].sum()/data_parags[data_parags['level'] == 'Ele']['n_words'].sum()
print("-----------------------------------------------------")
print("counting words including punctuations (n_words is text.split meaning that stand alone punctuations count)")

print(f"total words adv: {data_parags[data_parags['level'] == 'Adv']["n_words"].sum()}")
print(f"total words ele: {data_parags[data_parags['level'] == 'Ele']["n_words"].sum()}")
print(f"sentence length (words) adv: {data_parags[data_parags['level'] == 'Adv']["n_words"].sum()/data_parags[data_parags['level'] == 'Adv']["n_sentences"].sum()}")
print(f"sentence length (words) ele: {data_parags[data_parags['level'] == 'Ele']["n_words"].sum()/data_parags[data_parags['level'] == 'Ele']["n_sentences"].sum()}")


print(f"word freq advanced: {word_freq_adv}")
print(f"word freq elementry: {word_freq_ele}")
print(f"word length advanced: {word_length_adv}")
print(f"word length elementry: {word_length_ele}")

print(f"words per paragrpah adv = {data_parags[data_parags['level'] == 'Adv']['n_words'].sum()/len(data_parags[data_parags['level'] == 'Adv'])}")
print(f"words per paragrpah ele = {data_parags[data_parags['level'] == 'Ele']['n_words'].sum()/len(data_parags[data_parags['level'] == 'Ele'])}")


# counting words not including punctuations
word_length_adv = data_parags[data_parags['level'] == 'Adv']['sum_word_length'].sum()/data_parags[data_parags['level'] == 'Adv']['n_words_no_punct'].sum()
word_length_ele = data_parags[data_parags['level'] == 'Ele']['sum_word_length'].sum()/data_parags[data_parags['level'] == 'Ele']['n_words_no_punct'].sum()
word_freq_adv = data_parags[data_parags['level'] == 'Adv']['sum_raw_wordfreq'].sum()/data_parags[data_parags['level'] == 'Adv']['n_words_no_punct'].sum()
word_freq_ele = data_parags[data_parags['level'] == 'Ele']['sum_raw_wordfreq'].sum()/data_parags[data_parags['level'] == 'Ele']['n_words_no_punct'].sum()


print("-----------------------------------------------------")
print("counting words not including punctuations (n_words is number of words by split, minus stand alone punctiations )")
print(f"total words adv: {data_parags[data_parags['level'] == 'Adv']["n_words_no_punct"].sum()}")
print(f"total words ele: {data_parags[data_parags['level'] == 'Ele']["n_words_no_punct"].sum()}")
print(f"sentence length (words) adv: {data_parags[data_parags['level'] == 'Adv']["n_words_no_punct"].sum()/data_parags[data_parags['level'] == 'Adv']["n_sentences"].sum()}")
print(f"sentence length (words) ele: {data_parags[data_parags['level'] == 'Ele']["n_words_no_punct"].sum()/data_parags[data_parags['level'] == 'Ele']["n_sentences"].sum()}")

print(f"word freq advanced: {word_freq_adv}")
print(f"word freq elementry: {word_freq_ele}")
print(f"word length advanced: {word_length_adv}")
print(f"word length elementry: {word_length_ele}")


print(f"words per paragrpah adv = {data_parags[data_parags['level'] == 'Adv']['n_words_no_punct'].sum()/len(data_parags[data_parags['level'] == 'Adv'])}")
print(f"words per paragrpah ele = {data_parags[data_parags['level'] == 'Ele']['n_words_no_punct'].sum()/len(data_parags[data_parags['level'] == 'Ele'])}")



print("prev_marked")
for level, row in averages.iterrows():
    print(f"{level}:")
    for column, value in row.items():
        print(f"{column}: {value}")
    print() 





def summarize(data):
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.ppf(0.975, len(data) - 1) * se
    return mean, ci

print("stats from eye_df, including surprisal")
# trying the same with eye_df

src_path = Path.cwd().parents[0]
parag_metrics_path = EYE_BY_WORD_DF_L1_ALIGNED_PATH

eye_data = pd.read_csv(parag_metrics_path, engine='pyarrow') #[["unique_paragraph_id", "level", "align_idx", "subject_id", "text_spacing_version", "gpt2_surprisal", "word_length_no_punctuation", "wordfreq_frequency"]]


eye_data_one_version = eye_data.drop_duplicates(subset=["unique_paragraph_id", "level", "text_spacing_version","gpt2_surprisal","pythia70m_surprisal","word_length", "word_length_no_punctuation", "wordfreq_frequency", "text_spacing_version", "IA_LABEL"])
eye_data_one_version = eye_data_one_version[eye_data_one_version["text_spacing_version"]==0]

eye_data_one_version['raw_wordfreq'] = eye_data_one_version['IA_LABEL'].apply(lambda word: word_frequency(word.strip(string.punctuation), lang="en") * 1e4)
eye_data_one_version['wordfreq_frequency_2'] = eye_data_one_version['IA_LABEL'].apply(lambda word: -np.log2(word_frequency(word, lang="en", minimum=1e-11)))
eye_data_one_version['new_len'] = eye_data_one_version['IA_LABEL'].apply(lambda word: len(word.strip(string.punctuation)))


averages = eye_data_one_version[eye_data_one_version["text_spacing_version"]==0].groupby(["level"])[["gpt2_surprisal", "pythia70m_surprisal", "word_length","prev_wordfreq_frequency", "word_length_no_punctuation", "wordfreq_frequency", "wordfreq_frequency_2","raw_wordfreq", "new_len"]].mean()


# Print the results in the desired format
for level, row in averages.iterrows():
    print(f"Level: {level}, Text Spacing Version:0")
    for column, value in row.items():
        print(f"{column}: {value}")
    print()  # Add a newline between different combinations

# Print the results in the desired format
columns = [
    "gpt2_surprisal", "pythia70m_surprisal", "word_length",
    "prev_wordfreq_frequency", "word_length_no_punctuation",
    "wordfreq_frequency", "wordfreq_frequency_2", "raw_wordfreq", "new_len"
]

for col in columns:
    adv_ele = []
    for level in ["Adv", "Ele"]:
        subset = eye_data_one_version[eye_data_one_version["level"] == level][col].dropna()
        mean, ci = summarize(subset)
        adv_ele.append(subset)
        print(f"Level: {level}, Text Spacing Version: 0")
        print(f"{col}: {mean:.2f} ± {ci:.2f} (95% CI)")
        print()
    stat, pval = ttest_ind(adv_ele[0], adv_ele[1])
    print(f"Unpaired t-test p {col}= {pval:.4f} {pval_stars(pval)}")

#checking word_frequancy
col = 'raw_wordfreq'
for level in ["Adv", "Ele"]:
    subset = eye_data_one_version[eye_data_one_version["level"] == level][col].dropna()
    mean, ci = summarize(subset)
    adv_ele.append(subset)
    print(f"Level: {level}, Text Spacing Version: 0")
    print(f"{col}: {mean:.2f} ± {ci:.2f} (95% CI)")
    print()
stat, pval = ttest_ind(adv_ele[0], adv_ele[1])
print(f"Unpaired t-test p {col}= {pval:.4f} {pval_stars(pval)}")

levels = ["Adv", "Ele"]


import pandas as pd

# Prepare the data
data_frames = []
for i, level in enumerate(levels):
    df = pd.DataFrame({
        'Word Frequency': adv_ele[i],
        'Level': level
    })
    data_frames.append(df)

plot_data = pd.concat(data_frames, ignore_index=True)



plt.figure(figsize=(10, 6))
sns.histplot(data=plot_data, x='Word Frequency', hue='Level', element='poly',bins=30, stat='density', common_norm=False)

plt.title('Raw Word Frequency Distribution by Level')
plt.xlabel('Word Frequency')
plt.ylabel('Density')
plt.legend(title='Level', labels=levels)
plt.savefig("freq_plot_sns.png")
plt.show()

adv_ele = []
## not raw -  word_freq

#checking word_frequancy
col = 'wordfreq_frequency_2'
for level in ["Adv", "Ele"]:
    subset = eye_data_one_version[eye_data_one_version["level"] == level][col].dropna()
    mean, ci = summarize(subset)
    adv_ele.append(subset)
    print(f"Level: {level}, Text Spacing Version: 0")
    print(f"{col}: {mean:.2f} ± {ci:.2f} (95% CI)")
    print()
stat, pval = ttest_ind(adv_ele[0], adv_ele[1])
print(f"Unpaired t-test p {col}= {pval:.4f} {pval_stars(pval)}")

levels = ["Adv", "Ele"]


# Prepare the data
data_frames = []
for i, level in enumerate(levels):
    df = pd.DataFrame({
        'Word Frequency': adv_ele[i],
        'Level': level
    })
    data_frames.append(df)

plot_data = pd.concat(data_frames, ignore_index=True)



plt.figure(figsize=(10, 6))
sns.histplot(data=plot_data, x='Word Frequency', hue='Level', element='poly',bins=30, stat='density', common_norm=False)

plt.title('Word Frequency with figure skating Distribution by Level')
plt.xlabel('Word Frequency')
plt.ylabel('Density')
plt.legend(title='Level', labels=levels)
plt.savefig("freq_plot_2_sns.png")
plt.show()




# Load your paragraph-level data
df = pd.read_csv("src/Alignment_Sentences/data/paragraphs_df.csv")  # Replace with your actual filename
df = df.drop_duplicates()
def extract_word_stats(text):
    words = text.split()
    words = [word.strip(string.punctuation) for word in words]
    lengths = [len(word.strip(string.punctuation)) for word in words if word.strip(string.punctuation)]
    freqs = [word_frequency(word.strip(string.punctuation), lang="en") * 1e4 for word in words if word.strip(string.punctuation)]
    return lengths, freqs

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

def get_stats_for_level(df, level):
    lengths_all = []
    freqs_all = []

    for text in df[df['level'] == level]['text']:
        lengths, freqs = extract_word_stats(text)
        lengths_all.extend(lengths)
        freqs_all.extend(freqs)

    return np.array(lengths_all), np.array(freqs_all)

def summarize(data):
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.ppf(0.975, len(data) - 1) * se
    return mean, ci


compare = {"Adv":[], "Ele": []}
# Process for both levels
for level in ['Adv', 'Ele']:
    lengths, freqs = get_stats_for_level(df, level)
    sent_lengths = get_sentence_lens_for_level(df, level)
    mean_len, ci_len = summarize(lengths)
    mean_freq, ci_freq = summarize(freqs)
    mean_sent_len, ci_sent_len = summarize(sent_lengths)
    compare[level]= [lengths, freqs, sent_lengths]

    print(f"\nLevel: {level}")
    print(f"  Word Length     : {mean_len:.2f} ± {ci_len:.2f} (95% CI)")
    print(f"  Word Frequency  : {mean_freq:.2f} ± {ci_freq:.2f} (95% CI)")
    print(f"  Sentence Length     : {mean_sent_len:.2f} ± {ci_sent_len:.2f} (95% CI)")

for i in range(3):
    stat, pval = ttest_ind(compare["Adv"][i], compare["Ele"][i])
    print(f"Unpaired t-test p {i}= {pval:.4f} {pval_stars(pval)}")

compare = {"Adv":[], "Ele": []}
# Process for both levels
for level in ['Adv', 'Ele']:
    lengths, freqs = get_stats_for_level(data_parags, level)
    sent_lengths = get_sentence_lens_for_level(data_parags, level)
    mean_len, ci_len = summarize(lengths)
    mean_freq, ci_freq = summarize(freqs)
    mean_sent_len, ci_sent_len = summarize(sent_lengths)
    compare[level] = [lengths, freqs, sent_lengths]

    print(f"\nLevel: {level}")
    print(f"  Word Length     : {mean_len:.2f} ± {ci_len:.2f} (95% CI)")
    print(f"  Word Frequency  : {mean_freq:.2f} ± {ci_freq:.2f} (95% CI)")
    print(f"  Sentence Length     : {mean_sent_len:.2f} ± {ci_sent_len:.2f} (95% CI)")

for i in range(3):
    stat, pval = ttest_ind(compare["Adv"][i], compare["Ele"][i])
    print(f"Unpaired t-test p {i}= {pval:.4f} {pval_stars(pval)}")


print("---------------------new-------------------")

# print(eye_data_one_version.columns.tolist())
# Aggregate per-paragraph metrics
agg_eye = eye_data_one_version.groupby(['text_id', 'level']).agg({
    'wordfreq_frequency': 'mean',
    'word_length': 'mean',
    'gpt2_surprisal': 'mean',
    'pythia70m_surprisal': 'mean'
}).reset_index()

print(agg_eye.head())

# Pivot for paired comparison
pivoted_eye_surp = agg_eye.pivot(index='text_id', columns='level', values='pythia70m_surprisal')  # Change to desired metric for comparison
pivoted_eye_freq = agg_eye.pivot(index='text_id', columns='level', values='wordfreq_frequency')
pivoted_eye_len = agg_eye.pivot(index='text_id', columns='level', values   ='word_length')
# Drop unmatched pairs
pivoted_eye_surp = pivoted_eye_surp.dropna(subset=["Adv", "Ele"])
pivoted_eye_freq = pivoted_eye_freq.dropna(subset=["Adv", "Ele"])
pivoted_eye_len = pivoted_eye_len.dropna(subset=["Adv", "Ele"])
# Extract aligned values
adv_surp = pivoted_eye_surp['Adv']
ele_surp = pivoted_eye_surp['Ele']
adv_word_freq = pivoted_eye_freq['Adv']
ele_word_freq = pivoted_eye_freq['Ele']
adv_word_len = pivoted_eye_len['Adv']
ele_word_len = pivoted_eye_len['Ele']
# Paired t-test
perform_paired_ttest(adv_surp, ele_surp, 'Pythia Surprisal')
perform_paired_ttest(adv_word_freq, ele_word_freq, 'Word Frequency')
perform_paired_ttest(adv_word_len, ele_word_len, 'Word Length')
