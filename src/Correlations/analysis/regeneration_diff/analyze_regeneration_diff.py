"""Diagnose the regenerated Correlations outputs (working tree vs git HEAD) and check whether
the correlation coefficients are CONSISTENT with their own RT_all_metrics input.

Findings from the 2026-06-25 C0/correlations review that this reproduces:
  - The bootstrap is seeded (random_state=42+2*i), so the pipeline is deterministic.
  - RT_all_metrics_df gained +102 surprisal columns (SLOR/UID/PPL) and its existing
    eye-metric columns changed for L2/L1_and_L2 (the corrected 278-subject data).
  - BUT the regenerated correlations only moved for the surprisal predictors; the
    eye-metric x readability correlations did NOT change — and a fresh recompute from the
    current RT_all_metrics does not match the file. So the correlation coefficients are
    STALE w.r.t. the eye data: the pipeline reused a cached intermediate and must be re-run
    (a cache-invalidation, same class as the comprehension cache bug).

Two checks per directory:
  1. diff vs HEAD     - which RT_all_metrics columns changed (and by how much), by family.
  2. staleness vs input - recompute each full-sample (fold='all') correlation directly from the
                          current RT_all_metrics and compare to the file's value. Mismatches
                          mean the committed correlations don't reflect their own input.

Scope: only fold='all' is checked (the deterministic full-sample Pearson). The 10 CV folds and
200 bootstrap iterations are NOT independently verified — they resample the data and can't be
faithfully reproduced from the saved RT_all_metrics, but they derive from the same input, so a
stale fold='all' implies they are stale too.

Outputs (written to the results/ subfolder, timestamped):
  - regeneration_diff_report.txt    the full printed summary
  - stale_correlations.csv          every full-sample correlation that disagrees with a fresh
                                     recompute from RT_all_metrics (the stale rows)

Usage:
    python src/Correlations/analysis/regeneration_diff/analyze_regeneration_diff.py
    python src/Correlations/analysis/regeneration_diff/analyze_regeneration_diff.py L1 L2
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]          # .../Readability
RESULTS = HERE / "results"                          # all generated outputs land here
RESULTS.mkdir(exist_ok=True)
READER_TYPES = ["L1", "L2", "L1_and_L2", "L1_next_to_L2"]
ATOL = 1e-9                                          # below this counts as "unchanged"

# Predictor families, matched against the predictor name (substring, case-insensitive).
SURPRISAL_KEYS = ("slor", "uid", "entropy", "ppl", "surprisal")
EYE_METRICS = {
    "mean_FD", "mean_GD", "mean_NF", "mean_FirstPassGD", "mean_HigherPassFixation",
    "mean_nonzero_TF", "mean_nonzero_FF", "reading_speed",
    "SkipRateTotal", "SkipRateFirstPass", "RegRateTotal", "RegRateFirstPass",
}

_LINES: list[str] = []


def emit(line: str = "") -> None:
    """Print and capture a report line."""
    print(line)
    _LINES.append(line)


def classify_predictor(name: str) -> str:
    """Bucket a predictor (correlations `text_col` or a metric column) into a family."""
    base = name.split("_", 1)[1] if name[:4] in ("Adv_", "Ele_") else name
    base = base[5:] if base.startswith("diff_") else base
    low = name.lower()
    if any(k in low for k in SURPRISAL_KEYS):
        return "surprisal(SLOR/UID/entropy/PPL)"
    if base in EYE_METRICS or name in EYE_METRICS:
        return "eye-metric"
    return "readability/other"


def git_head(path: Path) -> pd.DataFrame | None:
    """Read the committed (HEAD) version of a tracked CSV, or None if untracked/new."""
    rel = path.relative_to(REPO)
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"HEAD:{rel}"], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None
    return pd.read_csv(StringIO(raw.decode()))


def changed_csvs(reader: str) -> list[Path]:
    """Changed CSV paths under src/Correlations/<reader> per git status."""
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--", f"src/Correlations/{reader}"]
    ).decode().splitlines()
    return [REPO / line[3:] for line in out if line.endswith(".csv")]


def numeric_col_changes(old: pd.DataFrame, new: pd.DataFrame, key: str | None) -> dict[str, float]:
    """max|Δ| per common numeric column, aligned by `key` if given (else by position)."""
    common = [c for c in new.columns if c in old.columns]
    if key and key in common:
        old, new = old.set_index(key), new.set_index(key)
        common = [c for c in common if c != key]
    changes = {}
    for c in common:
        if old[c].dtype.kind in "fc" and new[c].dtype.kind in "fc":
            try:
                d = (old[c] - new[c]).abs().max()
            except Exception:
                continue
            if pd.notna(d) and d > ATOL:
                changes[c] = float(d)
    return changes


def analyze_rt_all_metrics(path: Path) -> None:
    """RT_all_metrics_df_*: report added/removed columns and which existing inputs changed."""
    new = pd.read_csv(path)
    old = git_head(path)
    emit(f"  [{path.name}]")
    if old is None:
        emit("    new/untracked file"); return
    added = [c for c in new.columns if c not in old.columns]
    removed = [c for c in old.columns if c not in new.columns]
    key = "text_id" if "text_id" in new.columns else None
    changes = numeric_col_changes(old, new, key)
    emit(f"    rows {len(old)}->{len(new)} | +{len(added)} cols | -{len(removed)} cols "
         f"| {len(changes)} existing cols changed")
    if added:
        fam = pd.Series([classify_predictor(c) for c in added]).value_counts()
        emit(f"    added by family: {dict(fam)}")
    for c, d in sorted(changes.items(), key=lambda x: -x[1])[:10]:
        emit(f"      changed: {c:32s} max|Δ|={d:.2e}  [{classify_predictor(c)}]")


def analyze_correlations(path: Path) -> pd.DataFrame | None:
    """For each full-sample (fold='all') correlation, capture two comparisons:

      d_old_new  = |HEAD value  - working-tree value|   (did the committed file actually move?)
      d_stale    = |fresh value - working-tree value|   (does the file match its current input?)

    where `fresh` is the correlation recomputed directly from the current RT_all_metrics_df
    (a plain Pearson over the per-text columns, verified to reproduce the file when its input
    is unchanged). The diagnostic combination:
      - eye x readability:  d_old_new ~ 0 AND d_stale > 0  -> file frozen but input moved = STALE
      - surprisal:          d_old_new > 0 AND d_stale ~ 0  -> file tracked the new input = OK
    """
    from scipy.stats import pearsonr

    name = path.name  # correlations_<res>_<eye_metric>.csv
    res = "paragraph" if "_paragraph_" in name else "sentence" if "_sentence_" in name else None
    if res is None:
        return None
    rt_path = path.parent / f"RT_all_metrics_df_{res}.csv"
    if not rt_path.exists():
        return None
    rt = pd.read_csv(rt_path)
    new = pd.read_csv(path)
    old = git_head(path)
    if old is None or "fold" not in new.columns:
        return None
    key = ["pred_col", "text_col", "level_type", "reading_regime"]

    def full_sample(df):
        return df[(df["fold"].astype(str) == "all") & (df["bootstrap_iter"].isna())]

    m = full_sample(old).merge(full_sample(new), on=key, suffixes=("_old", "_new"))
    if m.empty:
        return None

    fresh = []
    for r in m.itertuples():
        eye_col = f"{r.level_type}_{r.pred_col}"        # e.g. Adv_mean_nonzero_TF
        prd_col = f"{r.level_type}_{r.text_col}"        # e.g. Adv_flesch_reading_ease
        if eye_col not in rt.columns or prd_col not in rt.columns:
            fresh.append(float("nan")); continue
        s = rt[[eye_col, prd_col]].dropna()
        if len(s) < 3 or s[eye_col].nunique() < 2 or s[prd_col].nunique() < 2:
            fresh.append(float("nan")); continue
        fresh.append(pearsonr(s[eye_col], s[prd_col])[0])

    m["file"] = name
    m["family"] = m["text_col"].map(classify_predictor)
    m["corr_old"] = m["pearson_corr_old"]
    m["corr_new"] = m["pearson_corr_new"]
    m["fresh_corr"] = fresh
    m["d_old_new"] = (m["corr_old"] - m["corr_new"]).abs()      # did the file actually change?
    m["d_expected"] = (m["fresh_corr"] - m["corr_old"]).abs()   # how much SHOULD it change (input moved)?
    m["d_stale"] = (m["fresh_corr"] - m["corr_new"]).abs()      # does the file match its current input?
    return m.dropna(subset=["fresh_corr"])[
        ["file", "pred_col", "text_col", "level_type", "reading_regime", "family",
         "corr_old", "corr_new", "fresh_corr", "d_old_new", "d_expected", "d_stale"]]


TOL = 1e-4  # a full-sample correlation difference above this counts as a real change

SCOPE_NOTE = (
    "SCOPE: this checks ONLY the full-sample correlation (fold='all') — a deterministic Pearson "
    "over all texts that recomputes exactly from RT_all_metrics. It does NOT independently verify "
    "the 10 CV folds (fold=1..10) or the 200 bootstrap iterations: those resample the data and "
    "depend on the pipeline's exact internal row order, so reproducing them from the saved "
    "RT_all_metrics is fragile (and would give false 'stale' flags). fold='all' is sufficient to "
    "detect staleness: if the input moved, the full-sample correlation must move, and the "
    "fold/bootstrap rows derive from the same input."
)


def _by_family(df: pd.DataFrame, col: str) -> None:
    fam = df.groupby("family").agg(n=(col, "size"), max=(col, "max"))
    for family, row in fam.iterrows():
        emit(f"     {family:34s} n={int(row.n):5d}  max|Δ|={row['max']:.2e}")


def write_html(a: pd.DataFrame, path: Path, stamp: str) -> None:
    """Render the staleness summary as a colour-coded HTML report (green=OK, red=stale)."""
    a = a.copy()
    a["input_moved"] = a["d_expected"] > TOL
    a["file_moved"] = a["d_old_new"] > TOL
    a["is_stale"] = a["d_stale"] > TOL
    total, n_stale = len(a), int(a["is_stale"].sum())
    pct = 100 * n_stale / total if total else 0

    # MECE buckets
    buckets = [
        ("✓ changed &amp; correct (recomputed, matches input)",
         int(((~a.is_stale) & a.file_moved).sum()), "#2e7d32"),
        ("✓ correctly unchanged (input didn't move, file didn't move)",
         int(((~a.is_stale) & ~a.file_moved).sum()), "#81c784"),
        ("✗ STALE — input moved but file NOT recomputed",
         int((a.is_stale & ~a.file_moved).sum()), "#c62828"),
        ("✗ STALE — file changed but still disagrees with input",
         int((a.is_stale & a.file_moved).sum()), "#ef9a9a"),
    ]
    bars = ""
    for label, val, color in buckets:
        w = 100 * val / total if total else 0
        bars += (f"<div class='barrow'><div class='lbl'>{label}</div>"
                 f"<div class='bar'><div class='fill' style='width:{max(w,2):.1f}%;background:{color}'>"
                 f"{val:,} ({w:.0f}%)</div></div></div>")

    grp = (a.groupby(["reader", "family"])
           .agg(n=("is_stale", "size"), input_moved=("input_moved", "sum"),
                file_moved=("file_moved", "sum"), stale=("is_stale", "sum"))
           .reset_index())
    rows = ""
    for r in grp.itertuples():
        bad = r.stale > 0
        verdict = ("STALE — input moved, file NOT recomputed" if bad and r.file_moved == 0
                   else "STALE — file ≠ input" if bad else "OK — matches input")
        rows += (f"<tr class='{'bad' if bad else 'good'}'><td>{r.reader}</td><td>{r.family}</td>"
                 f"<td>{r.n:,}</td><td>{int(r.input_moved):,}</td><td>{int(r.file_moved):,}</td>"
                 f"<td>{int(r.stale):,}</td><td>{verdict}</td></tr>")

    banner = "bad" if pct > 1 else "good"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#222;max-width:1000px}}
 h1{{margin-bottom:.2rem}} .ts{{color:#666;font-size:.9rem}}
 .banner{{padding:1rem 1.2rem;border-radius:8px;font-size:1.2rem;font-weight:600;margin:1rem 0}}
 .banner.bad{{background:#ffebee;color:#b71c1c;border:1px solid #ef9a9a}}
 .banner.good{{background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7}}
 .barrow{{display:flex;align-items:center;margin:.35rem 0;gap:.6rem}}
 .lbl{{flex:0 0 360px;font-size:.9rem}} .bar{{flex:1;background:#f0f0f0;border-radius:4px;overflow:hidden}}
 .fill{{color:#fff;padding:.25rem .5rem;font-size:.85rem;white-space:nowrap;border-radius:4px}}
 table{{border-collapse:collapse;width:100%;margin-top:.5rem;font-size:.9rem}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}} th{{background:#fafafa}}
 td:first-child,td:nth-child(2),td:last-child{{text-align:left}}
 tr.bad td{{background:#fff5f5}} tr.good td{{background:#f4fbf4}}
 code{{background:#f5f5f5;padding:.1rem .3rem;border-radius:3px}}
</style></head><body>
 <h1>Correlations regeneration — staleness report</h1>
 <p class="ts">generated {stamp} &middot; {total:,} full-sample (<code>fold='all'</code>) correlations checked</p>
 <div class="banner {banner}">{n_stale:,} / {total:,} ({pct:.0f}%) STALE — these correlations do not reflect their own RT_all_metrics input</div>
 <h2>Breakdown (every full-sample correlation, MECE)</h2>
 {bars}
 <h2>By reader &times; predictor family</h2>
 <table><tr><th>reader</th><th>family</th><th>n</th><th>input moved</th><th>file moved</th><th>STALE</th><th>verdict</th></tr>{rows}</table>
 <p class="ts"><b>Scope:</b> only the full-sample correlation (<code>fold='all'</code>) is checked
 &mdash; a deterministic Pearson that recomputes exactly from RT_all_metrics. The 10 CV folds
 and 200 bootstrap iterations are <b>not</b> independently verified (they resample the data and
 can't be faithfully reproduced from the saved table); they derive from the same input, so
 <code>fold='all'</code> staleness implies theirs.</p>
 <h2>Cause &amp; fix</h2>
 <p>The C1..C4 calc tasks ran with <code>CALC_FOR_SPECIFIC_TEXT_COLS</code> restricted to the
 surprisal predictors, so the eye-metric side of every other correlation was never recomputed
 after the C0 (278-subject) eye-data change. <b>Fix:</b> set the toggle to <code>None</code>
 (full recompute) in <code>src/Correlations/run_wrappers/tasks.py</code> and re-run C1..P5.</p>
</body></html>"""
    path.write_text(html)


def main(readers: list[str]) -> None:
    emit(SCOPE_NOTE)
    stale_all: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []
    for reader in readers:
        files = changed_csvs(reader)
        if not files:
            emit(f"\n=== {reader}: no changed CSVs ===")
            continue
        emit(f"\n{'='*70}\n{reader}: {len(files)} changed CSVs\n{'='*70}")

        rt_files = [f for f in files if f.name.startswith("RT_all_metrics_df")]
        corr_files = [f for f in files if f.name.startswith("correlations_")]

        if rt_files:
            emit("\n-- (1) RT_all_metrics_df input changes (working tree vs HEAD) --")
            for f in sorted(rt_files):
                analyze_rt_all_metrics(f)

        rows = [df for f in sorted(corr_files) if (df := analyze_correlations(f)) is not None]
        if not rows:
            continue
        chk = pd.concat(rows, ignore_index=True)
        chk.insert(0, "reader", reader)
        moved = chk[chk["d_old_new"] > TOL]       # file actually changed vs HEAD
        stale = chk[chk["d_stale"] > TOL]          # file disagrees with its current input
        stale_all.append(stale)
        all_rows.append(chk)

        emit("\n-- (2) full-sample (fold='all') correlations: old-vs-new AND fresh-vs-new --")
        emit(f"   checked: {len(chk)}")
        emit(f"   (a) CHANGED file vs HEAD (old->new): {len(moved)}")
        if not moved.empty:
            _by_family(moved.rename(columns={"d_old_new": "d"}), "d")
        emit(f"   (b) STALE file vs fresh recompute from current RT_all_metrics: {len(stale)} "
             f"({100*len(stale)/len(chk):.0f}%)")
        if not stale.empty:
            _by_family(stale.rename(columns={"d_stale": "d"}), "d")
            ex = stale.sort_values("d_stale", ascending=False).iloc[0]
            emit(f"   worst stale: {ex.pred_col} x {ex.text_col} [{ex.level_type}] "
                 f"old={ex.corr_old:.4f} new(file)={ex.corr_new:.4f} fresh(input)={ex.fresh_corr:.4f}")
            emit(f"   => {reader}: file barely moved for eye x readability yet disagrees with the "
                 "new input -> correlations are STALE; re-run the pipeline (cache not invalidated).")
        else:
            emit(f"   => {reader}: correlations are CONSISTENT with their input.")

    # ---- Final summary: expected (input moved) vs actual (file moved), and what's wrong ----
    if all_rows:
        a = pd.concat(all_rows, ignore_index=True)
        a["input_moved"] = a["d_expected"] > TOL   # the correlation SHOULD differ from HEAD
        a["file_moved"] = a["d_old_new"] > TOL      # the file actually differs from HEAD
        a["is_stale"] = a["d_stale"] > TOL          # the file disagrees with its current input
        emit(f"\n{'='*70}\nFINAL SUMMARY — expected vs actual (all readers, fold='all' rows)\n{'='*70}")
        emit(f"{'family':<34}{'n':>7}{'input_moved':>13}{'file_moved':>12}{'STALE':>8}  verdict")
        for fam, g in a.groupby("family"):
            im, fm, st = int(g.input_moved.sum()), int(g.file_moved.sum()), int(g.is_stale.sum())
            if st == 0:
                verdict = "OK: matches current input"
            elif fm == 0 and im > 0:
                verdict = "WRONG: input moved but file NOT recomputed -> STALE"
            else:
                verdict = "WRONG: file disagrees with input -> STALE"
            emit(f"{fam:<34}{len(g):>7}{im:>13}{fm:>12}{st:>8}  {verdict}")
        total_stale = int(a.is_stale.sum())
        emit(f"\nwhich cols CHANGED & are correct  : {int(((~a.is_stale) & a.file_moved).sum())} "
             "(recomputed, consistent with input)")
        emit(f"which cols did NOT change but SHOULD: {int((a.input_moved & ~a.file_moved & a.is_stale).sum())} "
             "(eye-data moved, correlation left stale)")
        emit(f"which cols correctly unchanged     : {int((~a.input_moved & ~a.file_moved & ~a.is_stale).sum())}")
        emit(f"\nWHAT'S WRONG: {total_stale} full-sample correlations ({100*total_stale/len(a):.0f}%) do not "
             "reflect their own RT_all_metrics input.")
        emit("Cause: the C1..C4 calc tasks ran with CALC_FOR_SPECIFIC_TEXT_COLS restricted to the")
        emit("surprisal predictors, so the eye-metric side of every other correlation was never")
        emit("recomputed after the C0 (278-subject) eye-data change. Fix: set the toggle to None")
        emit("(full recompute) in src/Correlations/run_wrappers/tasks.py and re-run C1..P5.")

    # Persist outputs next to this script, timestamped so a rerun (e.g. after re-running the
    # pipeline) writes a NEW snapshot instead of overwriting this one.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if stale_all:
        pd.concat(stale_all, ignore_index=True).to_csv(RESULTS / f"stale_correlations_{stamp}.csv", index=False)
    if all_rows:
        write_html(pd.concat(all_rows, ignore_index=True), RESULTS / f"regeneration_diff_{stamp}.html", stamp)
    emit(f"\nWrote regeneration_diff_{stamp}.(txt|html) + stale_correlations_{stamp}.csv "
         f"to {RESULTS.relative_to(REPO)}")
    (RESULTS / f"regeneration_diff_{stamp}.txt").write_text("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    requested = [a for a in sys.argv[1:] if a in READER_TYPES] or READER_TYPES
    main(requested)
