import nltk
import numpy as np
import pandas as pd


def mean_std(data: pd.Series) -> tuple[float, float]:
    clean = pd.Series(data).dropna()
    n = len(clean)
    m = float(np.mean(clean))
    if n <= 1:
        return m, np.nan
    std = float(np.std(clean, ddof=1))
    return m, std


def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if pd.isna(std):
        return f"{mean:.0f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def compute_group_stats(
    passage_df: pd.DataFrame,
    token_metrics_df: pd.DataFrame,
    group_name: str,
) -> list[dict]:
    """Compute descriptive statistics for one group of texts.

    Parameters
    ----------
    passage_df : DataFrame
        Must contain columns ``word_count``, ``num_sentences``,
        ``sentence_length_words``.
    token_metrics_df : DataFrame
        Must contain columns ``Length``, ``Wordfreq_Frequency``,
        ``EleutherAI/pythia-70m_Surprisal``.
    group_name : str
        Label stored in the ``group`` column (e.g. "original").
    """
    rows: list[dict] = []
    n_passages = len(passage_df)

    rows.append(
        {
            "metric": "number_of_passages",
            "group": group_name,
            "n": n_passages,
            "mean": float(n_passages),
            "std": np.nan,
            "mean_std": str(n_passages),
        }
    )
    rows.append(
        {
            "metric": "number_of_sentences",
            "group": group_name,
            "n": n_passages,
            "mean": float(passage_df["num_sentences"].sum()),
            "std": np.nan,
            "mean_std": str(int(passage_df["num_sentences"].sum())),
        }
    )

    metric_configs = [
        ("words_per_passage", "word_count", passage_df),
        ("sentences_per_passage", "num_sentences", passage_df),
        ("sentence_length_words", "sentence_length_words", passage_df),
        ("word_length_characters", "Length", token_metrics_df),
        ("word_frequency_wordfreq", "Wordfreq_Frequency", token_metrics_df),
        ("word_surprisal_pythia70m", "EleutherAI/pythia-70m_Surprisal", token_metrics_df),
    ]

    for metric_name, col_name, df in metric_configs:
        m, std = mean_std(df[col_name])
        rows.append(
            {
                "metric": metric_name,
                "group": group_name,
                "n": int(df[col_name].dropna().shape[0]),
                "mean": m,
                "std": std,
                "mean_std": format_mean_std(m, std),
            }
        )

    return rows


def ensure_sentence_tokenizer() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
