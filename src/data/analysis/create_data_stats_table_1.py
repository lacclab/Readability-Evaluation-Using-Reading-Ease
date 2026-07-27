import pandas as pd

# Load the CSV data
csv_path = "src/data/OneStop/OneStop_mean_ci_stats.csv"
df = pd.read_csv(csv_path)

# Prepare mapping for metrics and order
metric_order = [
    ("number_of_paragraphs", "Number of passages"),
    ("number_of_sentences", "Number of sentences"),
    ("number_of_questions", "Number of questions"),  # Not in CSV, placeholder
    ("words_per_passage", "Words per passage"),
    ("sentences_per_passage", "Sentences per passage"),
    ("sentence_length_words", "Sentence length (words)"),
    ("word_length_characters", "Word length (characters)"),
    ("word_frequency_wordfreq", "Word frequency (Wordfreq)"),
    ("word_surprisal_pythia70m", "Word surprisal (Pythia-70m)"),
]

# Optionally, add number_of_questions manually
num_questions = 486

latex_rows = []
for metric, label in metric_order:
    if metric == "number_of_questions":
        latex_rows.append(f"{label}  &  {num_questions} & {num_questions} & NA")
        continue
    orig = df[(df.metric == metric) & (df.group == "original")]
    simp = df[(df.metric == metric) & (df.group == "simplified")]
    if not orig.empty and not simp.empty:
        orig_val = orig.iloc[0]["mean_ci"]
        simp_val = simp.iloc[0]["mean_ci"]
        stars = orig.iloc[0]["stars"]  # Assuming p-value is the same for both groups
        # Replace nan with 'NA' for display
        if pd.isna(stars):
            stars = "NA"
        if pd.isna(stars):
            stars = "NA"
        latex_rows.append(f"{label}  &  {orig_val} & {simp_val} & {stars}")

# Build LaTeX table using the exact format requested
tabular_lines = [
    "\\begin{tabular}{@{}llll@{}}",
    "\\toprule",
    "      & \\textbf{Original} & \\textbf{Simplified} & \\textbf{p} \\\\ \\midrule",
    latex_rows[0] + " \\\\ ",  # Number of passages
    latex_rows[1] + " \\\\",  # Number of sentences
    latex_rows[2] + " \\\\ \\midrule",  # Number of questions
    latex_rows[3] + " \\\\",  # Words per passage
    latex_rows[4] + " \\\\",  # Sentences per passage
    latex_rows[5] + " \\\\ \\midrule",  # Sentence length
    latex_rows[6] + " \\\\",  # Word length
    latex_rows[7] + " \\\\",  # Word frequency
    latex_rows[8] + " \\\\",  # Word surprisal
    "\\bottomrule",
    "\\end{tabular}"
]

latex_table = "\n".join(tabular_lines)

with open("src/data/OneStop/stats_table_1.tex", "w") as f:
    f.write(latex_table)