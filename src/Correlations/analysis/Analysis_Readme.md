Analysis scripts for the Correlations module.

Folders:
- paradigm_divergence/    Find text_ids where readability paradigms (ET, formulas, psycholinguistic, LLM, comprehension) diverge.
- opposite_direction_check/  Check which metrics should be in OPPOSITE_DIRECTION_METRICS based on diff(Adv-Ele) stats.
- ppl_trend/              Perplexity trend analysis and model PPL comparisons.
- surprisal_robustness/   Surprisal robustness to different LLMs.
- participants/           Present number of participants per text.
- sentence_vs_paragraph_ppl/  Diagnose why PPL/surprisal correlations differ between sentence and paragraph resolutions in the main fig (validates stored correlations, checks exp() artefact from calc_correlations.py:408, cross-regime replication).
- regeneration_diff/       Review regenerated Correlations outputs old (git HEAD) vs new (working tree) after a data/pipeline change. Date-agnostic — rerun for any regeneration. See "regeneration_diff entry points" below.

Other:
- simplification types comparison analysis: src/Alignment_Sentences/simplification_types/human_annotation/process_results.ipynb


regeneration_diff entry points
==============================
Compare the Correlations outputs in the working tree against the committed (git HEAD) values.
Run each from the repo root; all outputs go to regeneration_diff/results/ (timestamped, gitignored;
one evidence snapshot is committed). old = git HEAD, new = working tree.

- analyze_regeneration_diff.py   Staleness gate: is each full-sample (fold='all') correlation
                                 CONSISTENT with its own RT_all_metrics input? (recomputes a fresh
                                 Pearson and flags any file value that disagrees). MECE buckets, HTML+txt+csv.
- compare_correlation_ranks.py   Did the predictor RANKINGS reorder? Ranks predictors by |r| per eye
                                 metric × level, old vs new (rank-Spearman, which family tops). HTML+csv.
- verify_bootstrap.py            Are the seeded bootstrap rows reproducible after the deterministic-sort fix?
- review_old_vs_new.py           Full interactive HTML review of every old→new change (see below).

Background: a 2026-06-25 regeneration recomputed only the surprisal predictors after a C0 eye-data
change, leaving ~57% of correlations stale (eye × readability frozen at old values). The committed
evidence snapshot is regeneration_diff/results/regeneration_diff_20260626_*.{txt,html}. Now fixed.

review_old_vs_new.py — the interactive old-vs-new console
---------------------------------------------------------
  python src/Correlations/analysis/regeneration_diff/review_old_vs_new.py            # all changed populations
  python .../review_old_vs_new.py --readers L1 L1_and_L2 --regimes FirstReading      # restrict scope
  python .../review_old_vs_new.py --no-pdfs                                          # skip the PDF compare file

Writes TWO self-contained files to results/ (both open directly in any browser — no server needed):
  review_old_vs_new_<stamp>.html   the main review console (tabs below)
  pdf_compare_<stamp>.html         the figure PDFs rasterised to PNG, old | new side by side

Top filters (population L1/L2/L1_and_L2/… · regime · resolution) drive every tab at once. Tabs:
- Coverage      every headline correlation (agg_folds pearson_corr_all) bucketed MECE — identical /
                negligible / material / significance-change / sign-flip — counts sum to the total and
                recompute live per population. A "Start here" panel ranks the populations and the
                eye-metric × level hotspots with the most changes; one click sets the filters.
- Correlations  filterable table (opens pre-filtered to one eye metric · diff · changed-only); each
                changed row expands to its 10 CV folds old vs new; toggle to show everything.
- Rankings      predictors ranked by |full-sample r| old vs new, computed over the paper's MAIN_TEXT_COLS
                (define_cols.py) by default; the "show" selector adds supplementary cols; rank-Spearman per view.
- Eye measures  filterable per-text table (old value / new / Δ / Δ%) for any eye metric × level, plus the
                old-vs-new distribution scatter per metric — the RT_all_metrics input that drove the changes.
- Distributions three views per metric — scatter (old vs new, y=x), Δ-histogram (new−old), overlaid hist.

The figure-PDF comparison is the SEPARATE pdf_compare_<stamp>.html (old | new, PDFs rasterised to PNG so
they render inline in any viewer incl. the VSCode preview); pick population/regime/figure from the dropdowns.
