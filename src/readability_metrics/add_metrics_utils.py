"""Shared path configs and helpers for adding metrics to readability dataframes."""
from pathlib import Path
from typing import Union, Literal
import pandas as pd

# Resolve paths from this file's location so the code works regardless of
# the current working directory. This file lives at <src>/readability_metrics/.
SRC_PATH = Path(__file__).resolve().parents[1]
INPUT_DATA_DIR = SRC_PATH / "Alignment_Sentences/data"
OUTPUT_DATA_DIR = SRC_PATH / "readability_metrics/data"

Resolution = Union[Literal["paragraph"], Literal["sentence"], Literal["article"]]

RESOLUTION_PATHS = {
    "paragraph": {
        "text_input_path": INPUT_DATA_DIR / "paragraphs_df_cleaned.csv",
        "save_path": OUTPUT_DATA_DIR / "paragraphs_metrics_cleaned.csv",
        "text_col": "text",
        "merge_on": ["unique_paragraph_id"],
    },
    "sentence": {
        "text_input_path": INPUT_DATA_DIR / "sentences_df_cleaned.csv",
        "save_path": OUTPUT_DATA_DIR / "sentences_metrics_cleaned.csv",
        "text_col": "sentence",
        "merge_on": ["unique_paragraph_id", "align_idx"],
    },
    "article": {
        "text_input_path": INPUT_DATA_DIR / "articles_df_cleaned.csv",
        "save_path": OUTPUT_DATA_DIR / "articles_metrics_cleaned.csv",
        "text_col": "text",
        "merge_on": ["unique_article_id"],
    },
}

ALIGNED_RESOLUTION_PATHS = {
    "paragraph": {
        "text_input_path": INPUT_DATA_DIR / "aligned_paragraphs_cleaned.csv",
        "save_path": OUTPUT_DATA_DIR / "paragraphs_metrics_diff_cleaned.csv",
        "merge_on": ["text_id"],
    },
    "sentence": {
        "text_input_path": INPUT_DATA_DIR / "aligned_sentences_no_NA_cleaned.csv",
        "save_path": OUTPUT_DATA_DIR / "sentences_metrics_diff_cleaned.csv",
        "merge_on": ["text_id", "align_idx"],
    },
    "article": {
        "text_input_path": INPUT_DATA_DIR / "aligned_articles_df_cleaned.csv",
        "save_path": OUTPUT_DATA_DIR / "articles_metrics_diff_cleaned.csv",
        "merge_on": ["text_id"],
    },
}


def merge_and_save(new_df: pd.DataFrame, save_path: Path, merge_on: list[str]):
    """Drop overlapping columns from existing metrics, merge with new data, and save.

    Preserves the existing file's column order: columns already in the file keep
    their position (with values from new_df when overlapping), and any new columns
    from new_df are appended at the end.
    """
    existing_df = pd.read_csv(save_path)
    existing_df = existing_df.loc[:, ~existing_df.columns.str.startswith("Unnamed")]
    # also strip any leaked index column from new_df so it never reaches the output
    new_df = new_df.loc[:, ~new_df.columns.str.startswith("Unnamed")]
    cols_to_drop = [c for c in existing_df.columns if c in new_df.columns and c not in merge_on]
    existing_df = existing_df.drop(columns=cols_to_drop)
    merged = new_df.merge(existing_df, on=merge_on, how="inner").reset_index(drop=True)

    # preserve existing column order: existing columns first (in their original order),
    # then any new columns not previously in the file
    existing_cols = [c for c in existing_df.columns if c in merged.columns]
    new_cols = [c for c in merged.columns if c not in existing_cols]
    merged = merged[existing_cols + new_cols]

    merged.to_csv(save_path, index=False)
