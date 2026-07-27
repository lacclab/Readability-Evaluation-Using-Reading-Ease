"""Compare predictor RANKINGS (old HEAD vs new working tree) for the regenerated correlations.

The staleness diagnostic (analyze_regeneration_diff.py) answers "did the numbers change / are
they consistent with their input". This answers the scientific question: "did the *ordering* of
predictors change — e.g. when you sort predictors big->small by |correlation| with an eye metric,
do the surprisal metrics still rank at the top in the new data as in the old?".

For each eye metric (pred_col) and level_type, it ranks all 118 predictors (text_col) by
|full-sample correlation| (fold='all'), in the committed (HEAD) data and the current working tree,
and reports:
  - rank stability: Spearman between the old and new rankings (1.0 = identical order),
  - which family tops each ranking (old vs new), and each family's best rank,
  - a side-by-side top-N ranked table per eye metric, coloured by family.

Outputs to the results/ subfolder (timestamped; gitignored except a deliberately committed snapshot):
  - rank_comparison_<stamp>.html   colour-coded side-by-side rankings
  - rank_comparison_<stamp>.csv    full per-(eye,level,predictor) old/new rank + value table

Usage:
  python src/Correlations/analysis/regeneration_diff/compare_correlation_ranks.py
  python .../compare_correlation_ranks.py --reader L1_and_L2 --regime FirstReading \
         --resolution paragraph --level diff
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
RESULTS = HERE / "results"                          # all generated outputs land here
RESULTS.mkdir(exist_ok=True)

SURPRISAL_KEYS = ("slor", "uid", "entropy", "ppl", "surprisal", "pll")


def family(name: str) -> str:
    low = name.lower()
    if any(k in low for k in SURPRISAL_KEYS) or name.strip().endswith("Mean"):
        return "surprisal"
    if "_prompt" in low or "text_evaluator" in low:
        return "llm-rating"
    return "readability"


FAM_COLOR = {"surprisal": "#1565c0", "llm-rating": "#ef6c00", "readability": "#2e7d32"}


def head_version(path: Path) -> pd.DataFrame | None:
    rel = path.relative_to(REPO)
    try:
        raw = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                                      stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return pd.read_csv(StringIO(raw.decode()))


def full_sample(df: pd.DataFrame, level: str) -> pd.DataFrame:
    d = df[(df["fold"].astype(str) == "all") & (df["bootstrap_iter"].isna())
           & (df["level_type"] == level)].copy()
    return d[["text_col", "pearson_corr"]]


def ranked(df: pd.DataFrame) -> pd.DataFrame:
    """Rank predictors big->small by |pearson_corr| (rank 1 = strongest)."""
    d = df.dropna(subset=["pearson_corr"]).copy()
    d["abs"] = d["pearson_corr"].abs()
    d = d.sort_values("abs", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    d["family"] = d["text_col"].map(family)
    return d


def compare_one(path: Path, level: str):
    new = pd.read_csv(path)
    old = head_version(path)
    if old is None:
        return None
    ro = ranked(full_sample(old, level))
    rn = ranked(full_sample(new, level))
    if ro.empty or rn.empty:
        return None
    m = ro.merge(rn, on="text_col", suffixes=("_old", "_new"))
    m["d_rank"] = m["rank_old"] - m["rank_new"]          # +ve => moved up (stronger) in new
    m["d_corr"] = m["pearson_corr_new"] - m["pearson_corr_old"]
    return ro, rn, m


def stability_row(eye, level, ro, rn, m):
    rho = spearmanr(m["rank_old"], m["rank_new"]).correlation
    top_old, top_new = ro.iloc[0], rn.iloc[0]

    def best_rank(rk, fam):
        s = rk[rk["family"] == fam]
        return int(s["rank"].min()) if len(s) else None
    return {
        "eye_metric": eye, "level": level, "rank_spearman": rho,
        "top1_old": f"{top_old.text_col} ({top_old.family})",
        "top1_new": f"{top_new.text_col} ({top_new.family})",
        "top1_family_changed": top_old.family != top_new.family,
        "surprisal_best_old": best_rank(ro, "surprisal"), "surprisal_best_new": best_rank(rn, "surprisal"),
        "readability_best_old": best_rank(ro, "readability"), "readability_best_new": best_rank(rn, "readability"),
        "llm_best_old": best_rank(ro, "llm-rating"), "llm_best_new": best_rank(rn, "llm-rating"),
    }


def fam_span(name, fam):
    return f"<span style='color:{FAM_COLOR[fam]}'>{name}</span>"


def detail_table(eye, ro, rn, m, topn=15):
    old_rank = {r.text_col: r.rank_old for r in m.itertuples()}
    rows = ""
    for k in range(topn):
        o = ro.iloc[k]; n = rn.iloc[k]
        prev = old_rank.get(n.text_col)
        shift = "" if prev is None else (f" <small>(↑{prev-(k+1)})</small>" if prev > k+1
                                         else f" <small>(↓{(k+1)-prev})</small>" if prev < k+1 else " <small>(=)</small>")
        rows += (f"<tr><td>{k+1}</td>"
                 f"<td>{fam_span(o.text_col, o.family)}</td><td>{o.pearson_corr:+.3f}</td>"
                 f"<td>{fam_span(n.text_col, n.family)}{shift}</td><td>{n.pearson_corr:+.3f}</td></tr>")
    return (f"<h3>{eye} — top {topn} predictors by |r|</h3>"
            f"<table><tr><th>rank</th><th>OLD predictor</th><th>r</th>"
            f"<th>NEW predictor</th><th>r</th></tr>{rows}</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="L1_and_L2")
    ap.add_argument("--regime", default="FirstReading")
    ap.add_argument("--resolution", default="paragraph")
    ap.add_argument("--level", default="diff", help="level_type for the detailed tables")
    args = ap.parse_args()

    cdir = REPO / "src" / "Correlations" / args.reader / args.regime
    files = sorted(cdir.glob(f"correlations_{args.resolution}_*.csv"))
    summary, details, full = [], [], []
    for f in files:
        eye = f.name.replace(f"correlations_{args.resolution}_", "").replace(".csv", "")
        for level in ["Adv", "Ele", "diff"]:
            res = compare_one(f, level)
            if res is None:
                continue
            ro, rn, m = res
            summary.append(stability_row(eye, level, ro, rn, m))
            mm = m.assign(eye_metric=eye, level=level)
            full.append(mm)
            if level == args.level:
                details.append(detail_table(eye, ro, rn, m))

    sdf = pd.DataFrame(summary)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if full:
        pd.concat(full)[["eye_metric", "level", "text_col", "family_new",
                         "rank_old", "rank_new", "d_rank",
                         "pearson_corr_old", "pearson_corr_new", "d_corr"]].to_csv(
            RESULTS / f"rank_comparison_{stamp}.csv", index=False)

    # summary table rows
    srows = ""
    for r in sdf.itertuples():
        cls = "warn" if r.top1_family_changed else ""
        srows += (f"<tr class='{cls}'><td>{r.eye_metric}</td><td>{r.level}</td>"
                  f"<td>{r.rank_spearman:.3f}</td>"
                  f"<td>{r.top1_old}</td><td>{r.top1_new}</td>"
                  f"<td>{r.surprisal_best_old}&rarr;{r.surprisal_best_new}</td>"
                  f"<td>{r.readability_best_old}&rarr;{r.readability_best_new}</td>"
                  f"<td>{r.llm_best_old}&rarr;{r.llm_best_new}</td></tr>")
    legend = " &nbsp; ".join(fam_span(k, k) for k in FAM_COLOR)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#222;max-width:1100px}}
 table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem;font-size:.85rem}}
 th,td{{border:1px solid #ddd;padding:.3rem .5rem;text-align:left}} th{{background:#fafafa}}
 td:nth-child(3){{text-align:right}} tr.warn td{{background:#fff8e1}}
 small{{color:#888}} h3{{margin-top:1.4rem}}
</style></head><body>
 <h1>Correlation predictor rankings — old (HEAD) vs new</h1>
 <p>{args.reader} &middot; {args.regime} &middot; {args.resolution} &middot; generated {stamp}<br>
 Predictors (text_cols) ranked big&rarr;small by <b>|full-sample correlation|</b> (fold='all'),
 <b>against each other within a fixed level_type</b> (Adv / Ele / diff are ranked separately).
 Family: {legend}.</p>
 <h2>Summary — rank stability &amp; which family tops each ranking</h2>
 <p>rank-spearman = Spearman between old and new predictor rankings (1.0 = identical order).
 "best" = the family's strongest rank (1 = top). Rows where the #1 predictor's family changed are highlighted.</p>
 <table><tr><th>eye metric</th><th>level</th><th>rank&nbsp;spearman</th><th>top-1 OLD</th><th>top-1 NEW</th>
 <th>surprisal best old&rarr;new</th><th>readability best</th><th>llm best</th></tr>{srows}</table>
 <h2>Detailed rankings (level = {args.level})</h2>
 {''.join(details)}
</body></html>"""
    out = RESULTS / f"rank_comparison_{stamp}.html"
    out.write_text(html)
    print(f"checked {len(files)} eye metrics x 3 levels; wrote {out.relative_to(REPO)} and the CSV")
    # quick console summary
    print("\nrank stability (Spearman) by level:")
    print(sdf.groupby("level")["rank_spearman"].agg(["min", "median", "max"]).round(3))
    print(f"\n#(eye,level) where the top-1 predictor's family changed: {int(sdf.top1_family_changed.sum())}/{len(sdf)}")


if __name__ == "__main__":
    main()
