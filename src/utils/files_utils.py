import pandas as pd
import os
import re
from loguru import logger
 
def replace_results_in_file(out_path, new_results_df, second_col=None):
    # Check which pred_col (and optionally second_col) combinations are in new results
    if second_col:
        new_keys = new_results_df[['pred_col', second_col]].drop_duplicates()
    else:
        new_keys = new_results_df[['pred_col']].drop_duplicates()

    # Load old results
    if os.path.exists(out_path):
        old_results_df = pd.read_csv(out_path)
        if 'pred_col' not in old_results_df.columns:
            old_results_df.rename(columns={'RT_col': 'pred_col'}, inplace=True)

        # Remove only rows whose keys match the new data; keep everything else
        for _, row in new_keys.iterrows():
            pred_col_val = row['pred_col']
            if second_col:
                second_col_val = row[second_col]
                condition = ~((old_results_df['pred_col'] == pred_col_val) &
                              (old_results_df[second_col] == second_col_val))
                old_results_df = old_results_df[condition]
            else:
                old_results_df = old_results_df[old_results_df['pred_col'] != pred_col_val]

        # Logging summary and preview
        num_keys = len(new_keys)
        preview_keys = new_keys.head(5).to_dict(orient='records')
        logger.warning(f"Replacing {num_keys} row combinations. Preview: {preview_keys}{' ...' if num_keys > 5 else ''}")

        # Concatenate preserved old rows with new results
        new_results_df = pd.concat([old_results_df, new_results_df], ignore_index=True)

    # Save final results
    new_results_df.to_csv(out_path, index=False)

def clean_paper_repo_from_unused_plots(paper_main_file, paper_SM_file, repo_dir, plots_dir="Plots"):

    # Define the paths to the main LaTeX files
    main_tex_files = [paper_main_file, paper_SM_file]

    # Regex pattern to match \input{path/to/file.tex}
    input_pattern = re.compile(r"\\input\{([^}]+)\}")

    # Set to store referenced plot files
    referenced_files = set()

    # Function to extract referenced plot files from LaTeX files
    def extract_referenced_files(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            matches = input_pattern.findall(content)
            for match in matches:
                if match.startswith("Plots/"):
                    referenced_files.add(match + ".tex" if not match.endswith(".tex") else match)

    # Extract referenced files from main LaTeX files
    for tex_file in main_tex_files:
        if os.path.exists(tex_file):
            extract_referenced_files(tex_file)

    # Find all .tex .pdf files in the Plots directory
    for root, _, files in os.walk(repo_dir / plots_dir):
        for file in files:
            if file.endswith(".tex"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path)  # Get relative path

                # Delete files not appearing in the referenced list
                if rel_path not in referenced_files:
                    print(f"Deleting: {rel_path}")
                    os.remove(full_path)

    
