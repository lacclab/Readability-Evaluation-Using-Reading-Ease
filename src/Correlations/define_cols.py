TRADITIONAL_MEASURES = [
    "flesch_reading_ease",# 1948
    "dale_chall_score", # 1948
    "gunning_fog_index", # 1952
    "ARI", # 1967
    "coleman_liau_index", # 1975
    "flesch_kincaid_grade_score", # 1975
]

MODERN_MEASURES = [
    "CML2RI", # 2008
    "CAREC", # 2019
    "CARES", # 2019
    "SBERT", # 2023
]

SYSTEMS_MEASURES = [
    "text_evaluator", # 2014
    "lexile", # 2022
]

TEXT_COL_TO_YEAR = {
    "flesch_reading_ease": 1948,
    "dale_chall_score": 1948,
    "gunning_fog_index": 1952,
    "ARI": 1967,
    "coleman_liau_index": 1975,
    "flesch_kincaid_grade_score": 1975,
    "CML2RI": 2008,
    "CAREC": 2019,
    "CARES": 2019,
    "SBERT": 2023,
}

LLMS_PROMPT_1 = [
    "llama-3.3-70b-versatile_simple_prompt",
    "gpt-4o_simple_prompt", 
    "gpt-5_simple_prompt",
    "gemini-2.0-flash_simple_prompt",
    "gemini-2.5-pro_simple_prompt",
    "claude-sonnet-4-0_simple_prompt"
]
LLMS_PROMPT_2 = [
    "llama-3.3-70b-versatile_simple_specific_prompt",
    "gpt-4o_simple_specific_prompt",
    "gpt-5_simple_specific_prompt",
    "gemini-2.0-flash_simple_specific_prompt",
    "gemini-2.5-pro_simple_specific_prompt",
    "claude-sonnet-4-0_simple_specific_prompt"
]
LLMS_PROMPT_3 = [
    "llama-3.3-70b-versatile_clear_prompt",
    "gpt-4o_clear_prompt",
    "gpt-5_clear_prompt",
    "gemini-2.0-flash_clear_prompt",
    "gemini-2.5-pro_clear_prompt",
    "claude-sonnet-4-0_clear_prompt"
]
LLMS_PROMPT_4 = [
    "llama-3.3-70b-versatile_clear_specific_prompt",
    "gpt-4o_clear_specific_prompt",
    "gpt-5_clear_specific_prompt",
    "gemini-2.0-flash_clear_specific_prompt",
    "gemini-2.5-pro_clear_specific_prompt",
    "claude-sonnet-4-0_clear_specific_prompt"
]
PROMPT_COLS_BY_NUMBER = {
    1: LLMS_PROMPT_1,
    2: LLMS_PROMPT_2,
    3: LLMS_PROMPT_3,
    4: LLMS_PROMPT_4
}
PROMPT_NAMES_BY_NUMBER = {
    1: "simple_prompt",
    2: "simple_specific_prompt",
    3: "clear_prompt",
    4: "clear_specific_prompt"
}
MAIN_LLM_PROMPT_NUMBER = 3

LLM_TO_YEAR = {
    "llama-3.3-70b-versatile": 2024,
    "gpt-4o": 2024,
    "gpt-5": 2025,
    "gemini-2.0-flash": 2025,
    "gemini-2.5-pro": 2025,
    "claude-sonnet-4-0": 2025,
}

LLM_COL_TO_YEAR = {
    "llama-3.3-70b-versatile_simple_prompt": 2024,
    "llama-3.3-70b-versatile_simple_specific_prompt": 2024,
    "llama-3.3-70b-versatile_clear_prompt": 2024,
    "llama-3.3-70b-versatile_clear_specific_prompt": 2024,
    "gpt-4o_simple_prompt": 2024,
    "gpt-4o_simple_specific_prompt": 2024,
    "gpt-4o_clear_prompt": 2024,
    "gpt-4o_clear_specific_prompt": 2024,
    "gpt-5_simple_prompt": 2025,
    "gpt-5_simple_specific_prompt": 2025,
    "gpt-5_clear_prompt": 2025,
    "gpt-5_clear_specific_prompt": 2025,
    "gemini-2.0-flash_simple_prompt": 2025,
    "gemini-2.0-flash_simple_specific_prompt": 2025,
    "gemini-2.0-flash_clear_prompt": 2025,
    "gemini-2.0-flash_clear_specific_prompt": 2025,
    "gemini-2.5-pro_simple_prompt": 2025,
    "gemini-2.5-pro_simple_specific_prompt": 2025,
    "gemini-2.5-pro_clear_prompt": 2025,
    "gemini-2.5-pro_clear_specific_prompt": 2025,
    "claude-sonnet-4-0_simple_prompt": 2025,
    "claude-sonnet-4-0_simple_specific_prompt": 2025,
    "claude-sonnet-4-0_clear_prompt": 2025,
    "claude-sonnet-4-0_clear_specific_prompt": 2025,
}

TEXT_COL_TO_YEAR.update(LLM_COL_TO_YEAR)

MAIN_PLL_COL = 'PLL_bert-base-uncased'
SM_PLL_COLS = [ 
    'PLL_roberta-base', 
    'PLL_roberta-large'
]

PSYCHOLINGUISTIC_MEASURES = [
    "cpidr_density", 
    "avg_integration_cost", 
    "avg_embedding_depth", 
    MAIN_PLL_COL,
    "mean_entropy_pythia",
    "word_length", "wordFreq_frequency", 
]


MAIN_PROMPT_COLS = PROMPT_COLS_BY_NUMBER[MAIN_LLM_PROMPT_NUMBER]
MAIN_LLM_PROMPT_NAME = PROMPT_NAMES_BY_NUMBER[MAIN_LLM_PROMPT_NUMBER]
SM_PROMPT_COLS = []
for num, cols in PROMPT_COLS_BY_NUMBER.items():
    if num != MAIN_LLM_PROMPT_NUMBER:
        SM_PROMPT_COLS += cols

MAIN_TEXT_COLS = (
    TRADITIONAL_MEASURES +
    MODERN_MEASURES +
    MAIN_PROMPT_COLS +
    SYSTEMS_MEASURES +
    PSYCHOLINGUISTIC_MEASURES
)

SM_TEXT_COLS = [
    "depid_density", 
    "max_embedding_depth",
    "max_integration_cost",
    "total_integration_cost", 
    "CAREC_M", 
    "mean_entropy_gpt2"
]
SM_TEXT_COLS += SM_PLL_COLS

MAIN_RT_COLS = ['mean_nonzero_TF', 'SkipRateTotal', 'RegRateTotal']

SM_RT_COLS_SET1 = ['mean_NF', 'mean_FirstPassGD', 'mean_GD']
SM_RT_COLS_SET2 = ['mean_nonzero_FF', 'mean_FD', 'mean_HigherPassFixation']
SM_RT_COLS_SET3 = ['reading_speed' ,'SkipRateFirstPass', 'RegRateFirstPass']
SM_RT_COLS = SM_RT_COLS_SET1 + SM_RT_COLS_SET2 + SM_RT_COLS_SET3

READING_COMPREHENSION_COLS = [
    "comprehension_score", "QA_RT"
]

OPPOSITE_DIRECTION_METRICS = [
    "flesch_reading_ease",
    "CML2RI",
    "monosyllabcount",
    "fernansex_huerta",
    "szigriszt_pazos",
    "gutierrez_polini",
    "wordFreq_frequency",
    "lexicon_count"
]

TEXT_COLS_FULL_LABELS = {
        "flesch_reading_ease": 'Flesch RE',
        "flesch_kincaid_grade_score": 'Flesch Kincaid',
        "dale_chall_score": 'Dale-Chall',
        "gunning_fog_index": 'Gunning Fog',
        "coleman_liau_index": 'Coleman-Liau',
        "ARI": 'ARI',
        
        "CAREC": 'CAREC',
        "CAREC_M": 'CAREC-M',
        "CARES": 'CARES',
        "CML2RI": 'CML2RI',
        
        "SBERT": 'SBERT', 
        "lexile": 'Lexile', 
        "text_evaluator": 'Text Evaluator',
        
        "word_length": 'Word Length',
        "wordFreq_frequency": 'Word Frequency',
        
        'pythia70m_surprisal': 'Pythia70m Surprisal',
        "mean_pythia70m_surprisal": 'Surprisal',
        'max_pythia70m_surprisal': 'Max Surprisal',
        'mean_gpt2_surprisal': 'GPT-2 Surprisal',
        'max_gpt2_surprisal': 'Max GPT-2 Surprisal',
        "mean_entropy_pythia": 'Entropy',
        "mean_entropy_gpt2": 'GPT-2 Entropy',
        "Pythia 70M Mean": 'Surprisal',
        'Pythia 70M Max': 'Max Surprisal',

        'cpidr_density': 'Idea Density',
        'depid_density': 'DEPID Idea Density',
        
        "avg_integration_cost": "Integration Cost",
        "max_integration_cost": "Max Integration Cost",
        "total_integration_cost": "Total Integration Cost",
        
        'max_embedding_depth': 'Max Emb Depth',
        'avg_embedding_depth': 'Embedding Depth',
        
        'n_words': 'N Words',
        'sentence_count': 'N Sentences',
        'syllable_count_textstats': 'N Syllables',
        'lexicon_count': 'Lexicon Count',
        'monosyllabcount': 'Monosyllable Count',
        'linsear_formula': 'Linsear',
        'fernansex_huerta': 'Fernández Huerta',
        'szigriszt_pazos': 'Szigriszt-Pazos',
        
        f"gpt-4o_{MAIN_LLM_PROMPT_NAME}": 'GPT-4o',
        f"gpt-5_{MAIN_LLM_PROMPT_NAME}": 'GPT-5',
        f"gemini-2.0-flash_{MAIN_LLM_PROMPT_NAME}": 'Gemini 2.0 Flash',
        f"gemini-2.5-pro_{MAIN_LLM_PROMPT_NAME}": 'Gemini 2.5 Pro',
        f'llama-3.3-70b-versatile_{MAIN_LLM_PROMPT_NAME}': 'Llama 3.3 70B',
        f'claude-sonnet-4-0_{MAIN_LLM_PROMPT_NAME}': 'Claude Sonnet 4.0',
        
        'PLL_bert-base-uncased': 'PLL',
        'PLL_bert-large-uncased': 'PLL BERT Large',
        'PLL_roberta-base': 'PLL RoBERTa Base',
        'PLL_roberta-large': 'PLL RoBERTa Large',
}

GRADE_PROMPT_LABEL = 'Grade'
GRADE_SPECIFIC_PROMPT_LABEL = 'Grade + Criteria'
SCORE_PROMPT_LABEL = 'Score'
SCORE_SPECIFIC_PROMPT_LABEL = 'Score + Criteria'
PROMPT_LABEL_ORDER = [GRADE_PROMPT_LABEL, GRADE_SPECIFIC_PROMPT_LABEL, SCORE_PROMPT_LABEL, SCORE_SPECIFIC_PROMPT_LABEL]
PROMPT_LABEL_ORDER_REVERSED = PROMPT_LABEL_ORDER[::-1]

PROMPT_COLS_FULL_LABELS = {
    "gpt-4o_simple_prompt": f'GPT-4o {GRADE_PROMPT_LABEL}',
    "gpt-5_simple_prompt": f'GPT-5 {GRADE_PROMPT_LABEL}',
    "o3-mini_simple_prompt": f'O3-Mini {GRADE_PROMPT_LABEL}',
    "gemini-2.0-flash_simple_prompt": f'Gemini 2.0 Flash {GRADE_PROMPT_LABEL}',
    "gemini-2.5-pro_simple_prompt": f'Gemini 2.5 Pro {GRADE_PROMPT_LABEL}',
    "llama-3.3-70b-versatile_simple_prompt": f'Llama 3.3 70B {GRADE_PROMPT_LABEL}',
    "claude-sonnet-4-0_simple_prompt": f'Claude Sonnet 4.0 {GRADE_PROMPT_LABEL}',


    "gpt-4o_simple_specific_prompt": f'GPT-4o {GRADE_SPECIFIC_PROMPT_LABEL}',
    "gpt-5_simple_specific_prompt": f'GPT-5 {GRADE_SPECIFIC_PROMPT_LABEL}',
    "gemini-2.0-flash_simple_specific_prompt": f'Gemini 2.0 Flash {GRADE_SPECIFIC_PROMPT_LABEL}',
    "o3-mini_simple_specific_prompt": f'O3-Mini {GRADE_SPECIFIC_PROMPT_LABEL}',
    "gemini-2.5-pro_simple_specific_prompt": f'Gemini 2.5 Pro {GRADE_SPECIFIC_PROMPT_LABEL}',
    "llama-3.3-70b-versatile_simple_specific_prompt": f'Llama 3.3 70B {GRADE_SPECIFIC_PROMPT_LABEL}',
    "claude-sonnet-4-0_simple_specific_prompt": f'Claude Sonnet 4.0 {GRADE_SPECIFIC_PROMPT_LABEL}',

    "gpt-4o_clear_prompt": f'GPT-4o {SCORE_PROMPT_LABEL}',
    "gpt-5_clear_prompt": f'GPT-5 {SCORE_PROMPT_LABEL}',
    "gemini-2.0-flash_clear_prompt": f'Gemini 2.0 Flash {SCORE_PROMPT_LABEL}',
    "o3-mini_clear_prompt": f'O3-Mini {SCORE_PROMPT_LABEL}',
    "gemini-2.5-pro_clear_prompt": f'Gemini 2.5 Pro {SCORE_PROMPT_LABEL}',
    "llama-3.3-70b-versatile_clear_prompt": f'Llama 3.3 70B {SCORE_PROMPT_LABEL}',
    "claude-sonnet-4-0_clear_prompt": f'Claude Sonnet 4.0 {SCORE_PROMPT_LABEL}',

    "gpt-4o_clear_specific_prompt": f'GPT-4o {SCORE_SPECIFIC_PROMPT_LABEL}',
    "gpt-5_clear_specific_prompt": f'GPT-5 {SCORE_SPECIFIC_PROMPT_LABEL}',
    "o3-mini_clear_specific_prompt": f'O3-Mini {SCORE_SPECIFIC_PROMPT_LABEL}',
    "gemini-2.0-flash_clear_specific_prompt": f'Gemini 2.0 Flash {SCORE_SPECIFIC_PROMPT_LABEL}',
    "gemini-2.5-pro_clear_specific_prompt": f'Gemini 2.5 Pro {SCORE_SPECIFIC_PROMPT_LABEL}',
    "llama-3.3-70b-versatile_clear_specific_prompt": f'Llama 3.3 70B {SCORE_SPECIFIC_PROMPT_LABEL}',
    "claude-sonnet-4-0_clear_specific_prompt": f'Claude Sonnet 4.0 {SCORE_SPECIFIC_PROMPT_LABEL}',
}

SURP_COLS_TO_SHORT_NAMES = {
    'EleutherAI/pythia-70m': 'Pythia 70M', 
    'EleutherAI/pythia-160m': 'Pythia 160M', 
    'EleutherAI/pythia-410m': 'Pythia 410M', 
    'EleutherAI/pythia-1b': 'Pythia 1B', 
    'EleutherAI/pythia-1.4b': 'Pythia 1.4B', 
    'EleutherAI/pythia-2.8b': 'Pythia 2.8B', 
    'EleutherAI/pythia-6.9b': 'Pythia 6.9B', 
    'gpt2': 'GPT-2 117M', 
    'gpt2-medium': 'GPT-2 345M', 
    'gpt2-large': 'GPT-2 774M', 
    'gpt2-xl': 'GPT-2 1558M', 
    'EleutherAI/gpt-j-6B': 'GPT-J 6B', 
    'EleutherAI/gpt-neo-125M': 'GPT-Neo 125M', 
    'EleutherAI/gpt-neo-1.3B': 'GPT-Neo 1.3B', 
    'EleutherAI/gpt-neo-2.7B': 'GPT-Neo 2.7B', 
    'meta-llama/Llama-2-7b-hf': 'Llama-2 7B', 
    'meta-llama/Llama-2-13b-hf': 'Llama-2 13B',
    'facebook/opt-350m': 'OPT 350M', 
    'facebook/opt-1.3b': 'OPT 1.3B',
    'facebook/opt-2.7b': 'OPT 2.7B', 
    'facebook/opt-6.7b': 'OPT 6.7B',
    'mistralai/Mistral-7B-v0.1': 'Mistral-v0.1 7B', 
    'mistralai/Mistral-7B-v0.3': 'Mistral-v0.3 7B', 
    'google/gemma-7b': 'Gemma 7B', 
    'google/gemma-2-9b': 'Gemma-2 9B',  
    'google/recurrentgemma-9b': 'Recurrent-Gemma 9B', 
    'RWKV/rwkv-4-169m-pile': 'RWKV-4 169M', 
    'RWKV/rwkv-4-430m-pile': 'RWKV-4 430M', 
    'state-spaces/mamba-370m-hf': 'Mamba 370M', 
    'state-spaces/mamba-790m-hf': 'Mamba 790M', 
    'state-spaces/mamba-1.4b-hf': 'Mamba 1.4B', 
    'state-spaces/mamba-2.8b-hf': 'Mamba 2.8B', 
}
# surp cols dict wih name f"{origin_name}_Surprisal"
SURP_COLS_RENAME_DICT = {f"{key}_Surprisal": value for key, value in SURP_COLS_TO_SHORT_NAMES.items()}

# ------------------------
# Choose Surprisal Columns
# ------------------------

MAIN_SURP_COLS = ['Pythia 70M Mean']
SM_SURP_COLS = []
ALL_SURP_COLS = [f"{col} Mean" for col in SURP_COLS_TO_SHORT_NAMES.values()] + ['Pythia 70M Max']