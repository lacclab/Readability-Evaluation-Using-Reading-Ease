import pandas as pd

def _get_models_data(src_path):
    perplexity_df = pd.read_csv(src_path / "Linguistic_Metrics/perplexity/models_names_with_ppl.csv")
    # rename col sentence_level_ppl to ppl
    perplexity_df = perplexity_df.rename(columns={'sentence_level_ppl': 'ppl'})
    # dict surp_col to perplexity
    surp_to_ppl = perplexity_df.set_index('surp_col')['ppl'].to_dict()
    # surp to family
    surp_to_family = perplexity_df.set_index('surp_col')['model_family'].to_dict()
    # surp to model_name_with_size
    surp_to_model_name_with_size = perplexity_df.set_index('surp_col')['model_name_with_size'].to_dict()
    return surp_to_ppl, surp_to_family, surp_to_model_name_with_size