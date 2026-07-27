"""Task registry for the correlation pipeline (see README.md for the task graph)."""
from pathlib import Path

from src.Correlations.run import run_corr, run_corr_perm_tests
from src.Correlations.define_cols import SLOR_COLS, UID_COLS


# Cheap subset to recompute when ONLY the surprisal metrics changed (e.g. the add_slor_and_uid work).
SURPRISAL_ONLY_TEXT_COLS = ["PPL Pythia 70M", "mean_entropy_pythia"] + SLOR_COLS + UID_COLS

# SINGLE TOGGLE for which predictors the C1..C4 calc tasks recompute:
#   None                      -> recompute ALL predictors (default; REQUIRED after any C0/eye-data change).
#   SURPRISAL_ONLY_TEXT_COLS  -> cheap refresh of only the surprisal predictors.
# WARNING: a restricted subset leaves the eye-metric side of every OTHER correlation
# (eye x readability, eye x prompt) silently STALE after a C0 change — see run_wrappers/README.md
# and src/Correlations/analysis/regeneration_diff/ for how that was caught.
# CALC_FOR_SPECIFIC_TEXT_COLS = ["max_embedding_depth", "avg_embedding_depth"]
CALC_FOR_SPECIFIC_TEXT_COLS = None

READER_TYPES = ["L1", "L2", "L1_and_L2"]
PREP_READING_REGIMES = ["FirstReading", "Gathering0", "Hunting0", "RepeatedReading"]
# Reading speed additionally reports an "All" regime (mirrors calc_reading_speed.py __main__);
# eye metrics intentionally omits it (mirrors calc_eye_metrics.py __main__).
PREP_READING_SPEED_REGIMES = PREP_READING_REGIMES + ["All"]


# -------- C0 sub-step checkpoints --------
# C0 is long (regenerates all underlying data). To make it resumable, each sub-step
# writes its own marker under .done/C0_prepare_data/<step>.done. A re-run skips any
# step whose marker exists, so only missing/failed steps run again.
#   - Re-run skipping done steps: rm .done/C0_prepare_data.done   (keeps sub-markers)
#   - Force a full fresh run:     rm -rf .done/C0_prepare_data.done .done/C0_prepare_data/
_C0_STEP_MARKER_DIR = Path(__file__).parent / ".done" / "C0_prepare_data"


def _c0_step(name, fn):
    """Run C0 sub-step `name` unless its marker exists; write the marker on success."""
    marker = _C0_STEP_MARKER_DIR / f"{name}.done"
    if marker.exists():
        print(f"[C0:{name}] skip — marker exists ({marker})", flush=True)
        return
    print(f"[C0:{name}] running...", flush=True)
    fn()
    _C0_STEP_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"[C0:{name}] done — marker written ({marker})", flush=True)


# -------- Data prep (C0) — OFF by default --------
# Regenerates the underlying data that every other task reads. Only needed when the
# source eye-tracking export changes (e.g. a new EYE_DF_L2_VERSION date in constants.py).
# Not part of the default launch (see launch_tmux.sh / README): run it manually and let it
# finish before launching the C1.. pipeline. See README "Updating the underlying data".

def _c0(src_path):
    # Lazy imports: these modules pull in heavy deps (thefuzz, tqdm, plotting) that the
    # normal pipeline tasks don't need, so keep them out of module import time.
    from src.constants import (
        EYE_BY_WORD_PATHS, EYE_BY_WORD_ALIGNED_PATHS,
        EYE_BY_FIXATION_PATHS, EYE_BY_FIXATION_ALIGNED_PATHS,
    )
    from src.Alignment_Sentences import merge_with_eye_df
    from src.Eye_metrics import calc_eye_metrics, calc_reading_speed
    from src.Eye_metrics.eye_df_utils import get_eye_dfs_by_reader_type, preprocess_eye_df
    from src.Reading_Comprehension import calc_reading_comprehension
    from src.Reading_Comprehension.calc_item_difficulty import save_item_difficulty
    from src.Participants_Metadata.preprocess_metadata import (
        build_participant_metadata, save_combined_L1_and_L2_metadata,
    )
    from src.Participants_Metadata.plot_hists_metadata import plot_participant_metadata_histograms

    # Every step below is wrapped in _c0_step(): it runs only if its marker is missing, so a
    # re-run resumes where it left off. Sub-steps are split per reader type where a step loops,
    # so a crash mid-loop only redoes the unfinished reader type. See _c0_step for re-run flags.

    # 1) Recreate the aligned eye dfs (EYE_BY_WORD_DF_*_ALIGNED_PATH). Mirrors
    #    merge_with_eye_df.py __main__: align words, align fixations, then union L1+L2.
    def _align(L1_or_L2):
        merge_with_eye_df.run(
            src_path, EYE_BY_WORD_PATHS[L1_or_L2], EYE_BY_WORD_ALIGNED_PATHS[L1_or_L2], L1_or_L2,
        )
        merge_with_eye_df.get_fixation_aligned_df(
            src_path, EYE_BY_FIXATION_PATHS[L1_or_L2], EYE_BY_FIXATION_ALIGNED_PATHS[L1_or_L2], L1_or_L2,
        )
    for L1_or_L2 in ["L1", "L2"]:
        _c0_step(f"1_align_{L1_or_L2}", lambda v=L1_or_L2: _align(v))
    _c0_step("1_align_union", merge_with_eye_df.union_L1_and_L2_aligned_dfs)

    # 2) Eye metrics for all reader types / reading regimes.
    #    calc_eye_metrics.run() reads `src_path` as a module global (set in its __main__).
    calc_eye_metrics.src_path = src_path
    def _eye_metrics(reader_type):
        calc_eye_metrics.run(
            reader_type=reader_type,
            calc_word_metrics=True,
            calc_fixation_metrics=True,
            calc_stats_and_plots=True,
            resolution_list=["sentence", "paragraph"],
            reading_regimes=PREP_READING_REGIMES,
            save_to_processed=True,
            bin_name=None,
        )
    for reader_type in READER_TYPES:
        _c0_step(f"2_eye_metrics_{reader_type}", lambda v=reader_type: _eye_metrics(v))

    # 3) Reading speed for all reader types. Mirrors calc_reading_speed.py __main__
    #    (calc_reading_speed_per_text() reads `src_path` as a module global).
    calc_reading_speed.src_path = src_path
    def _reading_speed(reader_type):
        results_dir = src_path / f"Eye_metrics/data/{reader_type}/reading_speed"
        results_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = src_path / f"Eye_metrics/plots/plot_reading_speed/{reader_type}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        original_eye_word_df, original_fixation_df = get_eye_dfs_by_reader_type(reader_type)
        for reading_regime in PREP_READING_SPEED_REGIMES:
            regime_eye_df = preprocess_eye_df(original_eye_word_df, reading_regime=reading_regime, filter_TF=False)
            regime_fixation_df = preprocess_eye_df(original_fixation_df, reading_regime=reading_regime, filter_TF=False)
            calc_reading_speed.calc_reading_speed_per_subject(
                regime_eye_df=regime_eye_df, reading_regime=reading_regime, results_dir=results_dir,
            )
            calc_reading_speed.calc_reading_speed_per_text(
                regime_eye_df=regime_eye_df, regime_fixation_df=regime_fixation_df,
                reading_regime=reading_regime, results_dir=results_dir, plots_dir=plots_dir,
            )
    for reader_type in READER_TYPES:
        _c0_step(f"3_reading_speed_{reader_type}", lambda v=reader_type: _reading_speed(v))

    # 4) Reading comprehension for all reader types. Mirrors calc_reading_comprehension.py
    #    __main__ (reads `src_path` and `L1_or_L2` as module globals). Run for all three so
    #    each group's comprehension_scores.csv exists for item difficulty below.
    calc_reading_comprehension.src_path = src_path
    def _reading_comprehension(reader_type):
        calc_reading_comprehension.L1_or_L2 = reader_type
        data_path = src_path / f"Reading_Comprehension/data/{reader_type}"
        save_plot_to_path = src_path / f"Reading_Comprehension/plots/{reader_type}"
        data_path.mkdir(parents=True, exist_ok=True)
        save_plot_to_path.mkdir(parents=True, exist_ok=True)
        # comprehension_scores.csv is a cached intermediate that
        # calc_and_plot_reading_comprehension_by_level reuses if present. Delete it so C0
        # recomputes from the current word-aligned df; otherwise a stale cache silently keeps
        # the old subject set after a data-version bump (e.g. L2 stuck at 251 instead of 278).
        (data_path / "comprehension_scores.csv").unlink(missing_ok=True)
        calc_reading_comprehension.calc_and_plot_reading_comprehension_by_level(
            EYE_BY_WORD_ALIGNED_PATHS[reader_type], data_path, save_plot_to_path,
        )
    for reader_type in READER_TYPES:
        _c0_step(f"4_reading_comprehension_{reader_type}", lambda v=reader_type: _reading_comprehension(v))

    # 5) Item difficulty (reads comprehension_scores.csv for L1, L2, L1_and_L2).
    _c0_step("5_item_difficulty", lambda: save_item_difficulty(src_path))

    # 6) Participant metadata. Build the processed table from the unified reading-habits survey
    #    export (both corpora, tagged by `corpus`; L2 restricted to the analyzed cohort inside the
    #    builder), then plot its histograms.
    def _metadata(reader_type):
        build_participant_metadata(reader_type)
        plot_participant_metadata_histograms(reader_type)
    for reader_type in ["L1", "L2"]:
        _c0_step(f"6_metadata_{reader_type}", lambda v=reader_type: _metadata(v))
    _c0_step("6_metadata_combined", save_combined_L1_and_L2_metadata)


# -------- Calc tasks --------

def _c1(src_path):
    run_corr(
        src_path, calc=True, calc_for_L1_or_L2="L1_and_L2",
        calc_for_specific_text_cols=CALC_FOR_SPECIFIC_TEXT_COLS,
        calc_for_reading_regimes=["FirstReading"],
    )


def _c2(src_path):
    run_corr(
        src_path, calc=True, calc_for_L1_or_L2="L1_and_L2",
        calc_for_specific_text_cols=CALC_FOR_SPECIFIC_TEXT_COLS,
        calc_for_reading_regimes=["Gathering0", "Hunting0"],
    )


def _c3(src_path):
    run_corr(
        src_path, calc=True, calc_for_L1_or_L2="L1",
        calc_for_specific_text_cols=CALC_FOR_SPECIFIC_TEXT_COLS,
        calc_for_reading_regimes=["FirstReading"],
    )


def _c4(src_path):
    run_corr(
        src_path, calc=True, calc_for_L1_or_L2="L2",
        calc_for_specific_text_cols=CALC_FOR_SPECIFIC_TEXT_COLS,
        calc_for_reading_regimes=["FirstReading"],
    )


def _c5(src_path):
    run_corr(src_path, calc_for_pair_plots=True)


def _c6(src_path):
    run_corr_perm_tests(
        src_path, calc=True, plot=False,
        calc_for_L1_or_L2="L1_and_L2",
        calc_for_reading_regimes=["FirstReading"],
    )


# -------- Plot tasks --------

def _p1(src_path):
    run_corr(
        src_path, plot=True,
        fig_list=[
            "main", "SM_RT", "SM_spearman", "SM_prompt", "SM_all_levels",
            "ppl", "SM_ppl", "SM_uid_cols", "SM_text_cols",
        ],
    )


def _p2(src_path):
    run_corr(src_path, plot=True, fig_list=["SM_L1_L2"])


def _p3(src_path):
    run_corr(src_path, plot=True, fig_list=["SM_hunting"])


def _p4(src_path):
    run_corr(src_path, plot=True, fig_list=["within_metrics_corr"])


def _p5(src_path):
    run_corr_perm_tests(
        src_path, calc=False, plot=True,
        calc_for_L1_or_L2="L1_and_L2",
        calc_for_reading_regimes=["FirstReading"],
    )


TASKS = {
    # C0 is OFF by default (not in launch_tmux.sh). Run only to regenerate underlying data.
    "C0_prepare_data":         {"fn": _c0, "deps": []},

    "C1_calc_l1_l2_first":     {"fn": _c1, "deps": []},
    "C2_calc_l1_l2_gath_hunt": {"fn": _c2, "deps": []},
    "C3_calc_l1_first":        {"fn": _c3, "deps": []},
    "C4_calc_l2_first":        {"fn": _c4, "deps": []},
    "C5_calc_pair_plots":      {"fn": _c5, "deps": ["C2_calc_l1_l2_gath_hunt", "C3_calc_l1_first", "C4_calc_l2_first"]},
    "C6_calc_perm_tests":      {"fn": _c6, "deps": ["C1_calc_l1_l2_first"]},

    "P1_plot_main":            {"fn": _p1, "deps": ["C1_calc_l1_l2_first"]},
    "P2_plot_sm_l1_l2":        {"fn": _p2, "deps": ["C5_calc_pair_plots"]},
    "P3_plot_sm_hunting":      {"fn": _p3, "deps": ["C5_calc_pair_plots"]},
    # P4 reads RT_all_metrics_df_*.csv (from C1) and corr results (updated by C5 via pair-plots steiger).
    "P4_plot_within_metrics":  {"fn": _p4, "deps": ["C1_calc_l1_l2_first", "C5_calc_pair_plots"]},
    # P5 reads perm_test_*.csv (from C6) and steiger_test_between_readability_formulas_*.csv (from C5).
    "P5_plot_perm_tests":      {"fn": _p5, "deps": ["C5_calc_pair_plots", "C6_calc_perm_tests"]},
}
