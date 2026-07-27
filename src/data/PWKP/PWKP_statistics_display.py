from pathlib import Path

import pandas as pd

from src.data.stats_utils import compute_group_stats


def build_group_rows(
    texts: pd.DataFrame,
    token_metrics: pd.DataFrame,
    group_name: str,
    text_col: str,
) -> list[dict]:
    word_col = f"word_count_{text_col}"
    sentence_col = f"num_sentences_{text_col}"
    sent_len_col = f"sentence_length_words_{text_col}"
    missing_cols = [
        col for col in [word_col, sentence_col, sent_len_col] if col not in texts.columns
    ]
    if missing_cols:
        raise ValueError(
            "Missing columns in PWKP_texts.csv: "
            f"{missing_cols}. Re-run PWKP_statistics_calculation.py first."
        )

    passage_df = texts[[word_col, sentence_col, sent_len_col]].rename(
        columns={
            word_col: "word_count",
            sentence_col: "num_sentences",
            sent_len_col: "sentence_length_words",
        },
    )

    return compute_group_stats(passage_df, token_metrics, group_name)


def compute_stats(base_dir: Path) -> pd.DataFrame:
    texts = pd.read_csv(base_dir / "PWKP_texts.csv").reset_index(drop=True)
    original_metrics = pd.read_csv(base_dir / "PWKP_metrics_original.csv")
    simplified_metrics = pd.read_csv(base_dir / "PWKP_metrics_simplified.csv")

    all_rows = []
    all_rows.extend(build_group_rows(texts, original_metrics, "original", "original"))
    all_rows.extend(
        build_group_rows(texts, simplified_metrics, "simplified", "simplified")
    )
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    out_df = compute_stats(base_dir)
    out_path = base_dir / "PWKP_mean_std_stats.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
