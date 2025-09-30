import pandas as pd
from typing import Literal

def agg_col_by_level_subject(eye_df: pd.DataFrame, col: str, agg_func: str, resolution: Literal["sentence", "paragraph", "article"]) -> pd.DataFrame:
    text_id_cols = get_text_id_cols(resolution)
    group_by_cols = list(set(text_id_cols + ['level', 'subject_id', 'batch']))
        
    return eye_df.groupby(group_by_cols)[col].agg(agg_func).reset_index()

def agg_col_by_level(eye_df: pd.DataFrame, col: str, agg_func: str, resolution: Literal["sentence", "paragraph", "article"]) -> pd.DataFrame:
    text_id_cols = get_text_id_cols(resolution)
    group_by_cols = text_id_cols + ['level']
        
    return eye_df.groupby(group_by_cols)[col].agg(agg_func).reset_index()

def separate_adv(data: pd.DataFrame):
    """
    The original data contains the paragrah ID in a x_x_adv_x format. This function splits this column into a binary
    'is_Adv' column and a paragraph id column.
    :param data: pandas df containing the eye tracking data.
    :return: the given dataframe with the "unique_paragraph_id" column replaced with a "is_adv" column and a
    "paragraph_id" column.
    """
    data["is_adv"] = data["unique_paragraph_id"].apply(lambda x: 1 if "Adv" in x else 0)
    data["paragraph_id"] = data["unique_paragraph_id"].apply(lambda x: x.replace("_Adv", "").replace("_Ele", ""))
    return data.drop(columns=['unique_paragraph_id'])

def separate_align_idx(df: pd.DataFrame):
    """
    from unique_sentence id to unique paragraph id+align idx
    """
    df[['unique_paragraph_id', 'align_idx']] = df['unique_sentence_id'].str.rsplit('_', n=1, expand=True)
    df['align_idx'] = df['align_idx'].astype(int)
    df = df.drop(columns={"unique_sentence_id"})
    return df

def get_text_id_cols(resolution: Literal["sentence", "paragraph", "article"]):
    if resolution == "sentence" or resolution == "sentences":
        text_id_cols = ['text_id', 'align_idx']
    elif resolution == "paragraph" or resolution == "paragraphs":
        text_id_cols = ['text_id']
    elif resolution == "article" or resolution == "articles":
        text_id_cols = ['batch', 'article_id']
    else:
        raise ValueError("resolution must be 'sentence' or 'paragraph'")
    return text_id_cols

def _check_batch(df, origin=False):
    if origin == "unique_article_id":
        df['batch'] = df['unique_article_id'].str.split("_").str[0].astype(int)
        return df
    elif origin == "text_id":
        df['batch'] = df['text_id'].str.split("_").str[0].astype(int)
        return df
    elif origin == "unique_paragraph_id":
        df['batch'] = df['unique_paragraph_id'].str.split("_").str[0].astype(int)
        return df
    
    if 'batch' not in df.columns:
        if 'unique_paragraph_id' in df.columns:
            df['batch'] = df['unique_paragraph_id'].str.split("_").str[0].astype(int)
        elif 'text_id' in df.columns:
            df['batch'] = df['text_id'].str.split("_").str[0].astype(int)
    return df

def _check_article_id(df, origin=False):
    if origin == "unique_article_id":
        df['article_id'] = df['unique_article_id'].str.split("_").str[1].astype(int)
        return df
    elif origin == "text_id":
        df['article_id'] = df['text_id'].str.split("_").str[1].astype(int)
        return df
    elif origin == "unique_paragraph_id":
        df['article_id'] = df['unique_paragraph_id'].str.split("_").str[1].astype(int)
        return df
    
    if 'article_id' not in df.columns:
        if 'unique_paragraph_id' in df.columns:
            df['article_id'] = df['unique_paragraph_id'].str.split("_").str[1].astype(int)
        elif 'text_id' in df.columns:
            df['article_id'] = df['text_id'].str.split("_").str[1].astype(int)
    return df

def _check_level(df, origin=False):
    if origin == "unique_article_id":
        df['level'] = df['unique_article_id'].str.split("_").str[2]
        return df
    elif origin == "unique_paragraph_id":
        df['level'] = df['unique_paragraph_id'].str.split("_").str[2]
        return df
        
    if 'level' not in df.columns:
        if 'unique_paragraph_id' in df.columns:
            df['level'] = df['unique_paragraph_id'].str.split("_").str[2]
    return df

def _check_paragraph_id(df):
    if 'paragraph_id' not in df.columns:
        if 'unique_paragraph_id' in df.columns:
            df['paragraph_id'] = df['unique_paragraph_id'].str.split("_").str[3].astype(int)
        elif 'text_id' in df.columns:
            df['level'] = df['text_id'].str.split("_").str[2]
    return df

def _add_text_id(df):
    if 'text_id' not in df.columns:
        df['text_id'] = df['batch'].astype(str) + "_" + df['article_id'].astype(str) + "_" + df['paragraph_id'].astype(str)
    return df

def _add_unique_paragraph_id(df):
    if 'unique_paragraph_id' not in df.columns:
        df['unique_paragraph_id'] = df['batch'].astype(str) + "_" + df['article_id'].astype(str) + "_" + df['level'] + "_" + df['paragraph_id'].astype(str)
    return df

def _add_unique_article_id(df):
    if 'unique_article_id' not in df.columns:
        df['unique_article_id'] = df['batch'].astype(str) + "_" + df['article_id'].astype(str) + "_" + df['level']
    return df

def _add_batch_article_id(df):
    if 'batch_article_id' not in df.columns:
        df['batch_article_id'] = df['batch'].astype(str) + "_" + df['article_id'].astype(str)
    return df

def add_id_cols(
    df, 
    text_id=False, 
    unique_paragraph_id=False,
    batch_article_id=False,
    unique_article_id=False,
    de_unique_article_id=False,
    de_unique_paragraph_id=False,
    de_text_id=False):
    if text_id:
        df = _check_batch(df)
        df = _check_article_id(df)
        df = _check_paragraph_id(df)
        df = _add_text_id(df)
    if unique_paragraph_id:
        df = _check_batch(df)
        df = _check_article_id(df)
        df = _check_level(df)
        df = _check_paragraph_id(df)
        df = _add_unique_paragraph_id(df)
    if unique_article_id:
        df = _check_batch(df)
        df = _check_article_id(df)
        df = _check_level(df)
        df = _add_unique_article_id(df)
    if batch_article_id:
        df = _check_batch(df)
        df = _check_article_id(df)
        df = _add_batch_article_id(df)
    if de_unique_article_id:
        df = _check_batch(df, origin="unique_article_id")
        df = _check_article_id(df, origin="unique_article_id")
        df = _check_level(df, origin="unique_article_id")
    if de_unique_paragraph_id:
        df = _check_batch(df, origin="unique_paragraph_id")
        df = _check_article_id(df, origin="unique_paragraph_id")
        df = _check_level(df, origin="unique_paragraph_id")
    if de_text_id:
        df = _check_batch(df, origin="text_id")
        df = _check_article_id(df, origin="text_id")
    return df

def add_reading_regime_col(df):
    df['reading_regime'] = df['has_preview'] + df['reread'].astype(str)
    return df
   