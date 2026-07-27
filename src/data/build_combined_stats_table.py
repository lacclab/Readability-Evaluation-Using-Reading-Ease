from pathlib import Path

import pandas as pd


ROW_ORDER = [
    ("Number of paragraphs", "number_of_passages"),
    ("Number of sentences", "number_of_sentences"),
    ("Words per paragraph", "words_per_passage"),
    ("Sentences per paragraph", "sentences_per_passage"),
    ("Sentence length (words)", "sentence_length_words"),
    ("Word length (characters)", "word_length_characters"),
    ("Word frequency (Wordfreq)", "word_frequency_wordfreq"),
    ("Word surprisal (Pythia-70m)", "word_surprisal_pythia70m"),
]

ONESTOP_METRIC_RENAMES = {
    "number_of_paragraphs": "number_of_passages",
}


def format_count(value: int | str) -> str:
    if isinstance(value, str):
        value = int(value.replace(",", ""))
    return f"{value:,}"


def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"metric", "group", "mean_std"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def get_group_value(df: pd.DataFrame, metric: str, group: str) -> str:
    match = df[(df["metric"] == metric) & (df["group"] == group)]
    if match.empty:
        raise ValueError(f"Missing value for metric={metric}, group={group}")
    return str(match["mean_std"].iloc[0])


def load_onestop_stats(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "OneStop" / "OneStop_mean_std_stats.csv"
    df = load_stats(path)
    df["metric"] = df["metric"].replace(ONESTOP_METRIC_RENAMES)
    return df


def load_or_compute_pwkp(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "PWKP" / "PWKP_mean_std_stats.csv"
    if path.exists():
        return load_stats(path)
    from src.data.PWKP.PWKP_statistics_display import compute_stats

    df = compute_stats(base_dir / "PWKP")
    df.to_csv(path, index=False)
    return df


def load_or_compute_clear(base_dir: Path) -> dict[str, str]:
    path = base_dir / "CLEAR" / "CLEAR_mean_std_stats.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        from src.data.CLEAR.CLEAR_statistics import compute_whole_corpus_stats

        text_df = pd.read_csv(base_dir / "CLEAR" / "CLEAR_corpus_final.csv")
        metrics_df = pd.read_csv(base_dir / "CLEAR" / "CLEAR_corpus_metrics.csv")
        df = compute_whole_corpus_stats(text_df, metrics_df)
        df.to_csv(path, index=False)
    return dict(zip(df["metric"], df["mean_std"]))


def build_table(
    onestop_df: pd.DataFrame,
    pwkp_df: pd.DataFrame,
    clear_values: dict[str, str],
) -> list[dict]:
    rows = []
    for label, metric_key in ROW_ORDER:
        onestop_original = get_group_value(onestop_df, metric_key, "original")
        onestop_simplified = get_group_value(onestop_df, metric_key, "simplified")

        if metric_key == "number_of_sentences":
            onestop_original = format_count(onestop_original)
            onestop_simplified = format_count(onestop_simplified)
            pwkp_original = format_count(
                get_group_value(pwkp_df, "number_of_sentences", "original")
            )
            pwkp_simplified = format_count(
                get_group_value(pwkp_df, "number_of_sentences", "simplified")
            )
        elif metric_key == "number_of_passages":
            onestop_original = format_count(onestop_original)
            onestop_simplified = format_count(onestop_simplified)
            pwkp_original = "NA"
            pwkp_simplified = "NA"
        elif metric_key in {"words_per_passage", "sentences_per_passage"}:
            pwkp_original = "NA"
            pwkp_simplified = "NA"
        else:
            pwkp_original = get_group_value(pwkp_df, metric_key, "original")
            pwkp_simplified = get_group_value(pwkp_df, metric_key, "simplified")

        clear_value = clear_values[metric_key]
        if metric_key in {"number_of_sentences", "number_of_passages"}:
            clear_value = format_count(clear_value)

        rows.append(
            {
                "Metric": label,
                "OneStop Original": onestop_original,
                "OneStop Simplified": onestop_simplified,
                "PWKP Original": pwkp_original,
                "PWKP Simplified": pwkp_simplified,
                "CLEAR All texts": clear_value,
            }
        )
    return rows


def build_multiindex_df(rows: list[dict]) -> pd.DataFrame:
    flat_df = pd.DataFrame(rows)
    ordered_flat_cols = [
        "Metric",
        "OneStop Original",
        "OneStop Simplified",
        "PWKP Original",
        "PWKP Simplified",
        "CLEAR All texts",
    ]
    flat_df = flat_df[ordered_flat_cols]
    flat_df.columns = pd.MultiIndex.from_tuples(
        [
            ("Metric", ""),
            ("OneStop", "Original"),
            ("OneStop", "Simplified"),
            ("PWKP", "Original"),
            ("PWKP", "Simplified"),
            ("CLEAR", "CLEAR"),
        ]
    )
    return flat_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    out_csv_path = base_dir / "combined_readability_stats.csv"
    out_tex_path = base_dir / "combined_readability_stats.tex"

    onestop_df = load_onestop_stats(base_dir)
    pwkp_df = load_or_compute_pwkp(base_dir)
    clear_values = load_or_compute_clear(base_dir)
    rows = build_table(onestop_df, pwkp_df, clear_values)
    combined_df = build_multiindex_df(rows)

    combined_df.to_csv(out_csv_path, index=False)

    body_lines = []
    for row in rows:
        body_lines.append(
            " & ".join(
                [
                    row["Metric"],
                    row["OneStop Original"],
                    row["OneStop Simplified"],
                    row["PWKP Original"],
                    row["PWKP Simplified"],
                    row["CLEAR All texts"],
                ]
            )
            + " \\\\"
        )
    tabular = "\n".join(
        [
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            " & \\multicolumn{2}{c}{OneStop} & \\multicolumn{2}{c}{PWKP} & CLEAR \\\\",
            "Metric & Original & Simplified & Original & Simplified &  \\\\",
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    caption = "Corpus statistics for the textual materials of OneStop \\citep{starc2020}, PWKP \\citep{zhu-etal-2010-monolingual} and CLEAR \\citep{crossley2021commonlit}. Means are reported with standard deviation. Word statistics are based on whitespace tokenization. Word length excludes punctuation. For CLEAR, sentence boundaries were obtained using NLTK."
    latex_table = (
        "\\begin{table}[ht]\n"
        f"\\caption{{{caption}}}\n"
        "\\centering\n"
        "\\small\n"
        "\\resizebox{1\\columnwidth}{!}{%\n"
        f"{tabular}"
        "}\n"
        "\\label{table:text-stats-app}\n"
        "\\end{table}\n"
    )
    out_tex_path.write_text(latex_table, encoding="utf-8")

    print(f"Saved {out_csv_path}")
    print(f"Saved {out_tex_path}")
