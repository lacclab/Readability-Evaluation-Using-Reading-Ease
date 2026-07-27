"""Shared 'review before committing' reminder for the regeneration-diff review tools.

Each review tool builds an old-vs-new HTML report from working-tree-vs-HEAD changes. This writes a
companion, NON-git-ignored TXT reminder next to that report and logs a warning, so a human reviews
the changes before committing them. Used by:
  - src/Eye_metrics/analysis/compare_rt_histogram_stats.py
  - src/Correlations/analysis/regeneration_diff/review_old_vs_new.py
  - src/Simplification_Effects/analysis/regeneration_diff/review_old_vs_new.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger


def write_review_reminder(results_dir, report_path, title: str, summary_lines: list[str],
                          reminder_name: str = "REVIEW_before_commit.txt") -> Path:
    """Write `results_dir/reminder_name` and log a warning. Returns the reminder path.

    results_dir   : dir to write the reminder into (must not be git-ignored, so it shows in status)
    report_path   : the HTML report the user should open
    title         : short headline, e.g. "eye-tracking measures were regenerated"
    summary_lines : pre-formatted lines describing what changed (already bulleted/indented)
    reminder_name : fixed filename → overwrites each run and stays visible in `git status`
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join([
        "=" * 72,
        f"  REVIEW BEFORE COMMITTING — {title}",
        "=" * 72,
        f"generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        "old = git HEAD   ·   new = working tree",
        "",
        *summary_lines,
        "",
        "Open the report and confirm the changes are expected, THEN commit:",
        f"  {report_path}",
        "",
        "(This reminder + the report regenerate on every run and are NOT git-ignored,",
        " so they surface in `git status`.)",
        "",
    ])
    path = results_dir / reminder_name
    path.write_text(body)
    logger.warning(f"REVIEW before committing — {title}:\n" + body)
    return path


# ---------------------------------------------------------------------------- wrapper marker status
# The regeneration pipelines run as a tmux task graph that writes a marker file
# `<wrapper>/.done/<task>.done` per finished task (see each run_wrappers/tasks.py TASKS dict). When a
# task crashes, its marker — and every downstream task's — is simply absent. The review tools surface
# that so a reviewer immediately sees the old-vs-new diff is built on an INCOMPLETE regeneration.

def wrapper_marker_status(done_dir, task_names) -> tuple[list[tuple[str, bool]], list[str]]:
    """(rows, missing) for the given ordered task list. rows = [(task, is_done)]; a task is done iff
    `done_dir/<task>.done` exists. missing = the tasks (in order) with no marker."""
    done_dir = Path(done_dir)
    rows = [(t, (done_dir / f"{t}.done").exists()) for t in task_names]
    missing = [t for t, ok in rows if not ok]
    return rows, missing


def wrapper_markers_banner_html(done_dir, task_names, pipeline_label: str = "") -> str:
    """A self-contained (inline-styled) HTML banner listing which wrapper `.done` markers are missing.

    Green when every task in `task_names` has its marker; amber — with the not-done tasks highlighted
    — when some are missing. Empty string if `task_names` is empty. Inline styles only, so it drops
    into any report regardless of its CSS."""
    if not task_names:
        return ""
    rows, missing = wrapper_marker_status(done_dir, task_names)
    n, m = len(task_names), len(missing)
    lbl = f" · {pipeline_label}" if pipeline_label else ""
    base = ("font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:.6rem .9rem;"
            "margin:.2rem 0 1rem;border-radius:8px;font-size:.86rem;line-height:1.6")
    if not missing:
        return (f"<div style='{base};background:#e8f5e9;border:1px solid #66bb6a;color:#1b5e20'>"
                f"✓ <b>Wrapper markers: all {n} present</b>{lbl} — regeneration ran to completion.</div>")

    def chips(tasks, bg):
        return " ".join(
            f"<span style='display:inline-block;background:{bg};color:#fff;border-radius:10px;"
            f"padding:.1rem .5rem;margin:.12rem .15rem;font-size:.78rem;white-space:nowrap'>{t}</span>"
            for t in tasks) or "<i>(none)</i>"

    done_tasks = [t for t, ok in rows if ok]
    return (f"<div style='{base};background:#fff3e0;border:1px solid #ffb300;color:#5d4037'>"
            f"⚠ <b>Wrapper regeneration INCOMPLETE — {m} of {n} markers NOT done{lbl}.</b> "
            f"The old-vs-new diff below may be partial or inconsistent with the changed inputs: a "
            f"task that crashed or hasn't run leaves its own and every downstream output stale.<br>"
            f"<span style='font-weight:600'>not done:</span> {chips(missing, '#c62828')}<br>"
            f"<span style='opacity:.65'>done: {chips(done_tasks, '#9e9e9e')}</span></div>")
