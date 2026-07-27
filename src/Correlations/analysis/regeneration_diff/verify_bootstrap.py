"""Verify the bootstrap rows are reproducible after the deterministic-sort fix.

The pipeline draws each bootstrap iteration with `all_metrics_df.sample(frac=1, replace=True,
random_state=42+2*i)` after sorting by a UNIQUE row key (fold, text_id [, align_idx]) — see the
fix in calc_correlations._get_corr_df. This script replicates that exact prep + draw for a sample
of (eye metric, predictor, level, iteration) cells and confirms the file value matches the
reproduced value. If the bootstrap were still order-sensitive (the old bug) these would not match.

Usage:
  python src/Correlations/analysis/regeneration_diff/verify_bootstrap.py
  python .../verify_bootstrap.py --reader L2 --regime FirstReading --resolution paragraph
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[4]
ITERS = [1, 50, 100, 150, 200]
LEVELS = ["Adv", "Ele", "diff"]


def reproduce(rt: pd.DataFrame, eye: str, pred: str, level: str, i: int) -> float:
    rt = rt.copy()
    rt["fold"] = rt["article_id"].astype(int)
    sort_keys = ["fold", "text_id"] + (["align_idx"] if "align_idx" in rt.columns else [])
    rt = rt.sort_values(by=sort_keys).reset_index(drop=True)           # same canonical order as pipeline
    res = rt.sample(frac=1, replace=True, random_state=42 + 2 * i)     # same seed
    a, b = f"{level}_{eye}", f"{level}_{pred}"
    if a not in res.columns or b not in res.columns:                  # e.g. QA_RT has no Adv_/Ele_ col
        return float("nan")
    s = res[[a, b]].dropna()
    if len(s) < 3 or s[a].nunique() < 2 or s[b].nunique() < 2:
        return float("nan")
    return pearsonr(s[a], s[b])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="L1_and_L2")
    ap.add_argument("--regime", default="FirstReading")
    ap.add_argument("--resolution", default="paragraph")
    args = ap.parse_args()

    cdir = REPO / "src" / "Correlations" / args.reader / args.regime
    rt = pd.read_csv(cdir / f"RT_all_metrics_df_{args.resolution}.csv")
    files = sorted(cdir.glob(f"correlations_{args.resolution}_*.csv"))

    checked = matched = 0
    worst = 0.0
    for f in files[:6]:   # a representative sample of eye metrics
        eye = f.name.replace(f"correlations_{args.resolution}_", "").replace(".csv", "")
        df = pd.read_csv(f)
        boot = df[df["fold"] == "bootstrap_all"]
        for level in LEVELS:
            preds = boot[boot["level_type"] == level]["text_col"].dropna().unique()
            if len(preds) == 0:
                continue
            pred = preds[len(preds) // 2]   # a mid predictor
            for i in ITERS:
                row = boot[(boot["text_col"] == pred) & (boot["level_type"] == level)
                           & (boot["bootstrap_iter"] == i)]
                if row.empty:
                    continue
                fv = float(row["pearson_corr"].iloc[0])
                rv = reproduce(rt, eye, pred, level, i)
                if pd.isna(rv):
                    continue
                checked += 1
                d = abs(fv - rv)
                worst = max(worst, d)
                if d < 1e-6:
                    matched += 1
                else:
                    print(f"  MISMATCH {eye} x {pred} [{level}] iter {i}: file={fv:.6f} repro={rv:.6f}")
    print(f"\n[{args.reader}/{args.regime}/{args.resolution}] bootstrap cells checked: {checked} | "
          f"matched: {matched} | worst |Δ|: {worst:.2e}")
    print("=> REPRODUCIBLE" if matched == checked and checked > 0 else "=> NOT fully reproducible")


if __name__ == "__main__":
    main()
