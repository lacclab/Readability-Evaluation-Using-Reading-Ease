# Correlations run wrappers

tmux-based orchestrator for the full correlation + plot pipeline. Each task runs in its own tmux window and waits for its dependencies via marker files.

## Quick start

From the repo root:

```bash
bash src/Correlations/run_wrappers/launch_tmux.sh
tmux attach -t readability
```

Switch between windows with `Ctrl-b n` / `Ctrl-b p`. Each window name matches a task ID (`C1_...`, `P1_...`, etc.).

A task is considered **done** when its marker file exists at `src/Correlations/run_wrappers/.done/<task>.done`. To re-run a task, delete its marker and relaunch.

## Task graph

```
C0  prepare_data (OFF by default)       [no deps]  — regenerates underlying data, see below
C1  L1_and_L2 FirstReading              [no deps]
C2  L1_and_L2 Gathering0+Hunting0       [no deps]
C3  L1 FirstReading                     [no deps]
C4  L2 FirstReading                     [no deps]
C5  pair_plots_calc                     [C2, C3, C4]
C6  perm_tests_calc                     [C1]

P1  main + 8 SM plots                   [C1]
P2  SM_L1_L2                            [C5]
P3  SM_hunting                          [C5]
P4  within_metrics_corr                 [C1, C5]
P5  perm_tests_plot                     [C5, C6]
```

## Which plot comes from which run — full map

All figures referenced in `main.tex` + `SI.tex` of the paper, and where they come from in this pipeline.

| Paper location | `\input` / `\includegraphics` path | `fig_list` key | Runs in | Depends on |
|---|---|---|---|---|
| **main** Fig | `Plots/.../FirstReading/Corr/tex_RTxSenPar_pearson_corr_FirstReading_boot` | `main` | P1 | C1 |
| **main** Fig 3 | `Plots/.../FirstReading/Corr/tex_RTxLevel_pearson_corr_by_perplexity_sentence` | `ppl` | P1 | C1 |
| SI simpl_types | `Plots/all/simplification_types_comparison.pdf` | — (separate pipeline) | — | — |
| SI perm test | `Plots/.../FirstReading/Corr/tex_RT_perm_test_grid_RTxSenPar_boot` | *(run_corr_perm_tests)* | P5 | C6 |
| SI readability corr | `Plots/.../FirstReading/Corr/tex_all_readability_measures_correlations` | `within_metrics_corr` | P4 | C5 |
| SI L1 next to L2 | `Plots/all/L1_next_to_L2/FirstReading/Corr/tex_SM_RT_main_...` | `SM_L1_L2` | P2 | C5 |
| SI Gathering/Hunting | `Plots/.../Gathering0_next_to_Hunting0/Corr/tex_SM_RT_main_...` | `SM_hunting` | P3 | C5 |
| SI SM_RT set 1 | `Plots/.../tex_SM_RT_1_RTxSenPar_pearson_corr_boot` | `SM_RT` | P1 | C1 |
| SI SM_RT set 2 | `Plots/.../tex_SM_RT_2_RTxSenPar_pearson_corr_boot` | `SM_RT` | P1 | C1 |
| SI SM_RT set 3 | `Plots/.../tex_SM_RT_3_RTxSenPar_pearson_corr_boot` | `SM_RT` | P1 | C1 |
| SI Spearman | `Plots/.../tex_SM_RT_main_..._pearson_next_to_spearman_corr_boot` | `SM_spearman` | P1 | C1 |
| SI Prompt Variants | `Plots/.../tex_SM_prompt_RTxSenPar_pearson_corr_FirstReading_set_main_boot` | `SM_prompt` | P1 | C1 |
| SI All Levels | `Plots/.../tex_SM_all_levels_pearson_corr_RE_next_to_delta_RE_boot` | `SM_all_levels` | P1 | C1 |
| SI ppl paragraph | `Plots/.../tex_RTxLevel_pearson_corr_by_perplexity_paragraph` | `SM_ppl` | P1 | C1 |
| SI §UID variants | *placeholder in SI line 364* (new) | `SM_uid_cols` | P1 | C1 |

### Plots defined in `run.py` but not currently in the paper
- `SM_text_cols` — kept in P1 for completeness.
- `SM_lextale`, `SM_advcomp`, `SM_repeated` — not included in any task.

## Updating the underlying data (C0 — off by default)

`C0_prepare_data` regenerates everything the C1.. / P1.. tasks read. It is **off by default**
(not in `launch_tmux.sh`) because it's only needed when the source eye-tracking export
changes — e.g. after a new `EYE_DF_L2_VERSION` date in `src/constants.py`. It runs, for all
reader types (`L1`, `L2`, `L1_and_L2`) and reading regimes (`FirstReading`, `Gathering0`,
`Hunting0`, `RepeatedReading`):

1. **Aligned eye df** — `Alignment_Sentences/merge_with_eye_df.py` (word + fixation align, then union L1+L2).
2. **Eye metrics** — `Eye_metrics/calc_eye_metrics.py`.
3. **Reading speed** — `Eye_metrics/calc_reading_speed.py`.
4. **Reading comprehension** — `Reading_Comprehension/calc_reading_comprehension.py` (per reader type).
5. **Item difficulty** — `Reading_Comprehension/calc_item_difficulty.py`.

To regenerate after a data change:

1. Update the relevant `*_VERSION` / path in `src/constants.py`.
2. Run C0 to completion (it's not a dependency of C1.., so don't launch them in parallel):

   ```bash
   python -m src.Correlations.run_wrappers.run_task C0_prepare_data
   ```

   Or run it in its own tmux window (C0 is long; this lets you detach and reattach):

   ```bash
   tmux new-session -d -s readability_c0 -c "$(git rev-parse --show-toplevel)" -n C0_prepare_data
   tmux send-keys -t readability_c0:C0_prepare_data \
     "python -m src.Correlations.run_wrappers.run_task C0_prepare_data" C-m
   tmux attach -t readability_c0
   ```

3. Delete the C1..P5 markers (`rm src/Correlations/run_wrappers/.done/{C,P}*.done`) and relaunch the pipeline.
4. Add a dated note to `src/Alignment_Sentences/readme.md` under **Updates**.

> ⚠️ **After a C0 data change, recompute ALL predictors — not a subset.**
> The C1..C4 calc tasks share one switch: the `CALC_FOR_SPECIFIC_TEXT_COLS` constant at the
> top of [`tasks.py`](tasks.py).
> - `None` → recompute **all** predictors (the default, and what you want after any C0 change).
> - `SURPRISAL_ONLY_TEXT_COLS` → cheap refresh of **only** the surprisal predictors (use only
>   when *just* those metrics changed, e.g. the `add_slor_and_uid` work).
>
> **Do not leave it on the subset after a C0/eye-data change.** A C0 change moves the eye-metric
> side of *every* correlation, but a restricted run recomputes only the listed predictors and
> leaves all others (eye × readability, eye × LLM-prompt) at their **old** values — silently
> stale (the file still has the pre-change correlations even though `RT_all_metrics_df` was
> updated). To check for this, run
> `python src/Correlations/analysis/regeneration_diff/analyze_regeneration_diff.py`
> — it flags any correlation whose value disagrees with a fresh recompute from its own input.
> See the committed evidence snapshot in `src/Correlations/analysis/regeneration_diff/results/`
> (`regeneration_diff_20260626_*.txt`/`.html`), and `src/Correlations/analysis/Readme.md`, for the
> worked example where this exact mistake was caught (2026-06-25).

### C0 is resumable (per-step checkpoints)

C0 is long, so each sub-step writes its own marker under
`.done/C0_prepare_data/<step>.done` and is skipped on a re-run if its marker exists. Steps
that loop over reader types (`L1`, `L2`, `L1_and_L2`) are split per reader type, so a crash
mid-loop only redoes the unfinished reader type. The steps (in order):

```
1_align_{L1,L2}, 1_align_union          aligned eye dfs (word + fixation align, then union)
2_eye_metrics_{L1,L2,L1_and_L2}          eye metrics
3_reading_speed_{L1,L2,L1_and_L2}        reading speed
4_reading_comprehension_{L1,L2,L1_and_L2} reading comprehension
5_item_difficulty                        item difficulty
6_metadata_{L1,L2}, 6_metadata_combined  participant metadata
```

- **Resume after a failure / re-run skipping done steps** — delete only the top-level marker
  (the per-step markers stay, so finished steps are skipped):

  ```bash
  rm -f src/Correlations/run_wrappers/.done/C0_prepare_data.done
  python -m src.Correlations.run_wrappers.run_task C0_prepare_data
  ```

- **Force a full fresh run** — also delete the per-step marker dir:

  ```bash
  rm -rf src/Correlations/run_wrappers/.done/C0_prepare_data.done \
         src/Correlations/run_wrappers/.done/C0_prepare_data/
  ```

- **Redo one step** (e.g. eye metrics for L2) — delete its marker plus the top-level marker:

  ```bash
  rm -f src/Correlations/run_wrappers/.done/C0_prepare_data.done \
        src/Correlations/run_wrappers/.done/C0_prepare_data/2_eye_metrics_L2.done
  python -m src.Correlations.run_wrappers.run_task C0_prepare_data
  ```

## Re-running a single task

1. Delete its marker: `rm src/Correlations/run_wrappers/.done/<task>.done`
2. Either: start a fresh tmux window and run `python -m src.Correlations.run_wrappers.run_task <task>`, or restart the whole session with `launch_tmux.sh`.

Children tasks still pick up because their deps' markers still exist (unless you deleted those too).

## Re-running all plot tasks (P1–P4)

After a plotting-only change (style tweaks, font sizes, etc.), re-run the plot tasks without re-doing any calc:

```bash
rm -f src/Correlations/run_wrappers/.done/{P1_plot_main,P2_plot_sm_l1_l2,P3_plot_sm_hunting,P4_plot_within_metrics}.done && \
for t in P1_plot_main P2_plot_sm_l1_l2 P3_plot_sm_hunting P4_plot_within_metrics; do \
  python -m src.Correlations.run_wrappers.run_task "$t"; \
done
```

Add `P5_plot_perm_tests` to both the `rm` list and the `for` loop if the change also affects perm-test heatmaps.

## Files

- `tasks.py` — task registry: name → (function, dependency list).
- `run_task.py` — worker. Polls for dep markers, runs the task, writes its own marker on success.
- `launch_tmux.sh` — creates the tmux session with one window per task.
- `.done/` *(gitignored)* — marker files, one per completed task.
