import pandas as pd
#  pip install git+https://github.com/lacclab/text-metrics.git@v1.1.12 
from src.utils.textual_utils import count_sentences
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from tqdm import tqdm
import tarfile
import os


if __name__ == "__main__":
    # Extract the tar.gz file
    tar_path = 'PWKP_108016.tar.gz'
    extract_dir = '.'

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

    # Read the extracted file
    extracted_file = 'PWKP_108016'

    # Read the file and parse it
    with open(extracted_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse the file: original text, then simplified text(s) until empty line
    original_texts = []
    simplified_texts = []
    original_sentence_counts = []
    simplified_sentence_counts = []

    i = 0
    while i < len(lines):
        # Read original text
        if lines[i].strip():
            original = lines[i].strip()
            i += 1
            
            # Read simplified text (can be multiple lines)
            simplified_parts = []
            while i < len(lines) and lines[i].strip():
                simplified_parts.append(lines[i].strip())
                i += 1
            
            if simplified_parts:
                original_texts.append(original)
                simplified_texts.append(' '.join(simplified_parts))
                original_sentence_counts.append(count_sentences(original))
                simplified_sentence_counts.append(count_sentences(' '.join(simplified_parts)))
            
        # Skip empty line
        i += 1

    # Create dataframe
    df = pd.DataFrame({
        'original': original_texts,
        'simplified': simplified_texts,
        'num_sentences_original': original_sentence_counts,
        'num_sentences_simplified': simplified_sentence_counts,
    })

    print(f"Extracted and loaded {len(df)} text pairs from {tar_path}")

    for text_col in ['original', 'simplified']:
        df[f"word_count_{text_col}"] = df[text_col].str.split().str.len()
        df[f"sentence_length_words_{text_col}"] = (
            df[f"word_count_{text_col}"] / df[f"num_sentences_{text_col}"]
        )

    df.to_csv('PWKP_texts.csv', index=False)
    df.describe().to_csv('PWKP_text_stats.csv')

    if os.getenv("PWKP_SKIP_TOKEN_METRICS", "0") == "1":
        print("PWKP_SKIP_TOKEN_METRICS=1 -> skipped token-level metric extraction.")
        raise SystemExit(0)

    extractor = get_surp_extractor(
        extractor_type=SurpExtractorType.CAT_CTX_LEFT,
        model_name='EleutherAI/pythia-70m',
        model_target_device='cuda:0',
    )
    for text_col in ['original', 'simplified']:
        metric_dfs = []
        for _, row in tqdm(
            iterable=df.iterrows(), total=len(df), desc='Processing texts'
        ):
            try:
                metrics = get_metrics(
                    target_text=' '.join(row[text_col].strip().split()), # normalize whitespace
                    language='en',
                    surp_extractor=extractor,
                    parsing_model=None,
                    add_parsing_features=False,
                    disregard_punctuation=True,
                )
            except Exception as e:
                print(f"Error processing text:{row[text_col]}: {e}")
                continue
            metric_dfs.append(metrics)
        metrics_df = pd.concat(metric_dfs, ignore_index=True)
        metrics_df.to_csv(f'PWKP_metrics_{text_col}.csv', index=False)
