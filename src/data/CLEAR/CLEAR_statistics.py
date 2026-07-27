from pathlib import Path

import pandas as pd

from src.data.stats_utils import compute_group_stats
from src.utils.textual_utils import count_sentences


def compute_whole_corpus_stats(
    text_df: pd.DataFrame, metrics_df: pd.DataFrame
) -> pd.DataFrame:
    passage_df = text_df.copy()
    passage_df["word_count"] = passage_df["Google WC"] / passage_df["Paragraphs"]
    passage_df["num_sentences"] = passage_df["Sentence Count"] / passage_df["Paragraphs"]
    passage_df["sentence_length_words"] = (
        passage_df["word_count"] / passage_df["num_sentences"]
    )
    passage_df = passage_df.loc[
        passage_df.index.repeat(passage_df["Paragraphs"])
    ].reset_index(drop=True)

    rows = compute_group_stats(passage_df, metrics_df, "all")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    text_df = pd.read_csv(base_dir / "CLEAR_corpus_final.csv")
    metrics_df = pd.read_csv(base_dir / "CLEAR_corpus_metrics.csv")

    out_df = compute_whole_corpus_stats(text_df, metrics_df)
    out_path = base_dir / "CLEAR_mean_std_stats.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
