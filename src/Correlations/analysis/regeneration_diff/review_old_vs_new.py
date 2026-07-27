"""Interactive old-vs-new review console for the regenerated Correlations outputs.

Where the companions answer narrow questions, this is the *full results review*: a single
self-contained HTML with global filters (population / regime / resolution) and tabs that let you
confirm you have seen **every** old->new change and only the ones that matter:

  - Coverage    every headline correlation (agg_folds `pearson_corr_all`) classified into MECE
                buckets (identical / negligible / material / sign-flip / significance-change);
                counts sum to the total and re-compute live as you change the population filter.
  - Correlations a filterable table (opens pre-filtered to one eye metric x diff x changed-only);
                every changed row expands to its 10 CV folds old vs new; toggle to show all.
  - Rankings    predictors ranked by |full-sample r| old vs new, over the paper's MAIN_TEXT_COLS
                (define_cols.py) by default; a selector adds supplementary cols; rank-Spearman per view.
  - Eye measures a filterable per-text table (old / new / Delta / Delta%) for any eye metric x level,
                plus the old-vs-new distribution scatter -- the RT_all_metrics input behind the changes.
  - Distributions  three views per metric: scatter (old vs new, y=x), Delta-histogram (new-old,
                centred at 0), and overlaid old/new histograms -- for both correlations and eye data.

The figure-PDF comparison is written to a SEPARATE self-contained file, pdf_compare_<stamp>.html
(old | new, PDFs rasterised to PNG so they render inline in any viewer, incl. the VSCode preview).

old = git HEAD, new = working tree (same convention as the other scripts in this folder).

Usage:
  # default: every reader/regime/resolution whose agg_folds changed vs HEAD
  python src/Correlations/analysis/regeneration_diff/review_old_vs_new.py
  # restrict the populations built into the report:
  python .../review_old_vs_new.py --readers L1 L1_and_L2
  python .../review_old_vs_new.py --regimes FirstReading --no-pdfs   # skip the pdf_compare file
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import subprocess
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.review_reminder import (
    write_review_reminder, wrapper_marker_status, wrapper_markers_banner_html,
)

try:
    import fitz  # PyMuPDF — rasterise PDFs to PNG so they render inline anywhere
except Exception:
    fitz = None

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
RESULTS = HERE / "results"                          # all generated outputs (html + pdf png cache) land here
RESULTS.mkdir(exist_ok=True)
CORR = REPO / "src" / "Correlations"

# Wrapper task graph — keep in sync with src/Correlations/run_wrappers/tasks.py TASKS. The review
# banner flags which of these `.done` markers are missing (e.g. a crashed C5 → C5 + every downstream
# P-task absent). C0_prepare_data is intentionally EXCLUDED: it's off-by-default manual data prep, so
# its marker is normally absent on a calc/plot-only rerun and would be a false "incomplete" alarm.
WRAPPER_DONE_DIR = CORR / "run_wrappers" / ".done"
WRAPPER_TASKS = [
    "C1_calc_l1_l2_first", "C2_calc_l1_l2_gath_hunt", "C3_calc_l1_first", "C4_calc_l2_first",
    "C5_calc_pair_plots", "C6_calc_perm_tests",
    "P1_plot_main", "P2_plot_sm_l1_l2", "P3_plot_sm_hunting", "P4_plot_within_metrics",
    "P5_plot_perm_tests",
]


def _load_define_cols():
    spec = importlib.util.spec_from_file_location("_define_cols", CORR / "define_cols.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DC = _load_define_cols()
# The paper's "main" predictors (text_cols), per src/Correlations/define_cols.py. Rankings are
# computed over THESE by default; the Rankings tab lets you add supplementary cols on top.
MAIN_PREDS = list(dict.fromkeys(_DC.MAIN_TEXT_COLS + _DC.MAIN_SURP_COLS))

ATOL = 1e-9        # |Δr| at/below this == identical (float noise floor)
MATERIAL = 0.01    # |Δr| above this == a "material" change worth a look

EYE_METRICS = [
    "mean_nonzero_TF", "mean_nonzero_FF", "mean_FD", "mean_NF", "mean_GD",
    "mean_FirstPassGD", "mean_HigherPassFixation", "reading_speed",
    "SkipRateTotal", "SkipRateFirstPass", "RegRateTotal", "RegRateFirstPass",
]
SURPRISAL_KEYS = ("slor", "uid", "entropy", "ppl", "surprisal", "pll")

# Predictor FAMILY colours (used to colour predictor *names*). Kept distinct from the change-category
# palette below so a name's colour is never confused with its row's change category.
FAM_COLOR = {"surprisal": "#1565c0", "llm-rating": "#ef6c00", "readability": "#2e7d32"}
# Change categories, low->high severity. Each headline cell gets exactly one (MECE).
# Palette deliberately avoids the family blue/orange/green so chip colour != name colour.
CAT_ORDER = ["identical", "negligible", "material", "sig-change", "sign-flip", "added", "removed"]
CAT_COLOR = {
    "identical": "#b0bec5", "negligible": "#78909c", "material": "#5e35b1",
    "sig-change": "#d81b60", "sign-flip": "#c62828", "added": "#00897b", "removed": "#6d4c41",
}
CAT_LABEL = {
    "identical": "identical (|Δr|≤1e-9)",
    "negligible": f"negligible (≤{MATERIAL})",
    "material": f"material (|Δr|>{MATERIAL})",
    "sig-change": "significance changed",
    "sign-flip": "SIGN FLIP",
    "added": "added (new only)",
    "removed": "removed (old only)",
}
LEVELS_CORR = ["Adv", "Ele", "diff", "all"]
LEVELS_EYE = ["Adv", "Ele", "diff"]


def family(name: str) -> str:
    low = name.lower()
    if any(k in low for k in SURPRISAL_KEYS) or name.strip().endswith("Mean"):
        return "surprisal"
    if "_prompt" in low or "text_evaluator" in low:
        return "llm-rating"
    return "readability"


def git_head_df(path: Path) -> pd.DataFrame | None:
    rel = path.relative_to(REPO)
    try:
        raw = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                                      stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return pd.read_csv(StringIO(raw.decode()))


def git_head_bytes(path: Path) -> bytes | None:
    rel = path.relative_to(REPO)
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def categorize(ro, rn, so, sn) -> str:
    """One MECE bucket for a headline correlation cell (old vs new)."""
    if pd.isna(ro) and pd.isna(rn):
        return "identical"
    if pd.isna(ro):
        return "added"
    if pd.isna(rn):
        return "removed"
    d = abs(rn - ro)
    mag = max(abs(ro), abs(rn))
    if (ro > 0) != (rn > 0) and mag > MATERIAL:
        return "sign-flip"
    if str(so) != str(sn) and d > ATOL:
        return "sig-change"
    if d > MATERIAL:
        return "material"
    if d > ATOL:
        return "negligible"
    return "identical"


# ----------------------------------------------------------------------------- data assembly
def discover_combos(readers, regimes, resolutions) -> list[tuple[str, str, str]]:
    """(reader, regime, resolution) triples whose agg_folds changed vs HEAD."""
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--", "src/Correlations"]
    ).decode().splitlines()
    combos = []
    for line in out:
        p = line[3:].strip()
        if "agg_folds_corr_" not in p:
            continue
        parts = Path(p).parts  # src Correlations <reader> <regime> agg_folds_corr_<res>.csv
        if len(parts) < 5:
            continue
        reader, regime, fname = parts[2], parts[3], parts[4]
        res = fname.replace("agg_folds_corr_", "").replace(".csv", "")
        if readers and reader not in readers:
            continue
        if regimes and regime not in regimes:
            continue
        if resolutions and res not in resolutions:
            continue
        combos.append((reader, regime, res))
    return sorted(set(combos))


def build_headline(reader, regime, res) -> tuple[list[dict], dict]:
    """Headline records (one per agg_folds cell) + per-cell fold drilldown for changed cells."""
    path = CORR / reader / regime / f"agg_folds_corr_{res}.csv"
    if not path.exists():
        return [], {}
    new = pd.read_csv(path)
    old = git_head_df(path)
    keys = ["pred_col", "text_col", "level_type"]
    cols = ["pearson_corr_all", "pearson_p_all_symbol", "pearson_corr_boot",
            "CI_yerr_pearson_boot", "spearman_corr_all"]
    keep = keys + [c for c in cols if c in new.columns]
    new = new[keep]
    old = old[keep] if old is not None else new.iloc[0:0]
    m = old.merge(new, on=keys, how="outer", suffixes=("_o", "_n"), indicator=True)

    recs, changed_cells = [], []
    for d in m.to_dict("records"):
        ro, rn = d.get("pearson_corr_all_o"), d.get("pearson_corr_all_n")
        so, sn = d.get("pearson_p_all_symbol_o"), d.get("pearson_p_all_symbol_n")
        if d["_merge"] == "left_only":
            cat = "removed"
        elif d["_merge"] == "right_only":
            cat = "added"
        else:
            cat = categorize(ro, rn, so, sn)
        rec = {
            "rd": reader, "rg": regime, "rs": res,
            "eye": d["pred_col"], "pred": d["text_col"], "lvl": d["level_type"],
            "fam": family(d["text_col"]), "cat": cat,
            "ro": _r(ro), "rn": _r(rn), "dr": _r((rn - ro) if (pd.notna(ro) and pd.notna(rn)) else None),
            "so": _s(so), "sn": _s(sn),
            "bo": _r(d.get("pearson_corr_boot_o")), "bn": _r(d.get("pearson_corr_boot_n")),
            "cio": _r(d.get("CI_yerr_pearson_boot_o")), "cin": _r(d.get("CI_yerr_pearson_boot_n")),
            "spo": _r(d.get("spearman_corr_all_o")), "spn": _r(d.get("spearman_corr_all_n")),
        }
        recs.append(rec)
        if cat in ("material", "sig-change", "sign-flip"):
            changed_cells.append((d["pred_col"], d["text_col"], d["level_type"]))
    folds = build_folds(reader, regime, res, set(changed_cells)) if changed_cells else {}
    return recs, folds


def build_folds(reader, regime, res, changed: set) -> dict:
    """For each changed cell, the 10 CV folds + full-sample, old vs new (from the per-eye files)."""
    out: dict[str, dict] = {}
    eyes = {c[0] for c in changed}
    for eye in eyes:
        path = CORR / reader / regime / f"correlations_{res}_{eye}.csv"
        if not path.exists():
            continue
        new = pd.read_csv(path)
        old = git_head_df(path)
        if old is None or "fold" not in new.columns:
            continue

        def fullish(df):
            return df[df["bootstrap_iter"].isna() & (df["fold"].astype(str) != "bootstrap_all")]

        no, nn = fullish(old), fullish(new)
        for (e, pred, lvl) in [c for c in changed if c[0] == eye]:
            so = no[(no.text_col == pred) & (no.level_type == lvl)][["fold", "pearson_corr"]]
            sn = nn[(nn.text_col == pred) & (nn.level_type == lvl)][["fold", "pearson_corr"]]
            mm = so.merge(sn, on="fold", how="outer", suffixes=("_o", "_n"))
            mm["k"] = mm["fold"].astype(str).map(lambda x: -1 if x == "all" else int(x) if x.isdigit() else 99)
            mm = mm.sort_values("k")
            key = f"{reader}|{regime}|{res}|{eye}|{pred}|{lvl}"
            out[key] = {
                "lab": mm["fold"].astype(str).tolist(),
                "o": [_r(v) for v in mm["pearson_corr_o"]],
                "n": [_r(v) for v in mm["pearson_corr_n"]],
            }
    return out


def _r(v, nd=4):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    return round(float(v), nd)


def _s(v):
    return "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)


# ----------------------------------------------------------------------------- plots
def _b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_correlations(recs: list[dict]) -> str:
    """3-panel: scatter old/new coloured by category, Δ-hist, overlaid old/new hist."""
    ro = np.array([r["ro"] for r in recs if r["ro"] is not None and r["rn"] is not None], float)
    rn = np.array([r["rn"] for r in recs if r["ro"] is not None and r["rn"] is not None], float)
    cats = [r["cat"] for r in recs if r["ro"] is not None and r["rn"] is not None]
    if len(ro) == 0:
        return ""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    cols = [CAT_COLOR.get(c, "#999") for c in cats]
    ax[0].scatter(ro, rn, s=10, c=cols, alpha=.7, edgecolors="none")
    lim = [min(ro.min(), rn.min()) - .05, max(ro.max(), rn.max()) + .05]
    ax[0].plot(lim, lim, "k--", lw=.8); ax[0].axhline(0, color="#bbb", lw=.6); ax[0].axvline(0, color="#bbb", lw=.6)
    ax[0].set(xlabel="r (old)", ylabel="r (new)", title="correlation: old vs new", xlim=lim, ylim=lim)
    dd = rn - ro
    ax[1].hist(dd[np.abs(dd) > ATOL], bins=40, color="#1565c0", alpha=.85)
    ax[1].axvline(0, color="k", lw=.8); ax[1].set(xlabel="Δr (new−old)", ylabel="cells", title="Δ distribution (changed)")
    ax[2].hist(ro, bins=40, color="#888", alpha=.6, label="old")
    ax[2].hist(rn, bins=40, color="#e53935", alpha=.5, label="new")
    ax[2].legend(); ax[2].set(xlabel="r", ylabel="cells", title="overlaid old/new")
    fig.tight_layout()
    return _b64(fig)


def _rt_key_cols(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    """Row-identity key for the RT_all_metrics tables. At sentence level `text_id` repeats (one row
    per sentence), so old↔new MUST align on (text_id, align_idx) — joining on text_id alone is a
    many-to-many cartesian product that fabricates off-diagonal 'changes' for unchanged eye data.
    At paragraph level there is no align_idx and text_id is unique, so the key is just text_id."""
    return [c for c in ("text_id", "align_idx") if c in old.columns and c in new.columns]


def plot_eye_measures(reader, regime, res) -> tuple[str, list[dict]]:
    """Small-multiple scatter (old vs new per text) per eye metric, pooling Adv/Ele/diff; + summary."""
    path = CORR / reader / regime / f"RT_all_metrics_df_{res}.csv"
    if not path.exists():
        return "", []
    new = pd.read_csv(path)
    old = git_head_df(path)
    if old is None or "text_id" not in new.columns:
        return "", []
    key = _rt_key_cols(old, new)
    o = old.set_index(key); n = new.set_index(key)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.ravel()
    lvl_color = {"Adv": "#1565c0", "Ele": "#2e7d32", "diff": "#ef6c00"}
    summary = []
    for i, eye in enumerate(EYE_METRICS):
        ax = axes[i]
        allo, alln = [], []
        for lvl in LEVELS_EYE:
            col = f"{lvl}_{eye}"
            if col not in o.columns or col not in n.columns:
                continue
            j = o[[col]].join(n[[col]], lsuffix="_o", rsuffix="_n", how="inner").dropna()
            if j.empty:
                continue
            ax.scatter(j[f"{col}_o"], j[f"{col}_n"], s=8, alpha=.5,
                       c=lvl_color[lvl], edgecolors="none", label=lvl)
            allo += j[f"{col}_o"].tolist(); alln += j[f"{col}_n"].tolist()
        if allo:
            lo, hi = min(allo + alln), max(allo + alln)
            ax.plot([lo, hi], [lo, hi], "k--", lw=.7)
            d = np.array(alln) - np.array(allo)
            summary.append({"eye": eye, "n": len(allo), "mean_d": _r(float(np.mean(d)), 3),
                            "max_abs_d": _r(float(np.max(np.abs(d))), 3),
                            "corr": _r(float(np.corrcoef(allo, alln)[0, 1]), 4) if len(allo) > 2 else None})
        ax.set_title(eye, fontsize=8.5)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, markerscale=1.4, title="level")
    fig.suptitle(f"Eye measures (RT_all_metrics) per text: old vs new — {reader}/{regime}/{res}\n"
                 f"each point = one text · on the dashed y=x line ⇒ unchanged · off-diagonal ⇒ moved",
                 fontsize=10)
    fig.supxlabel("old value (HEAD)", fontsize=10)
    fig.supylabel("new value (working tree)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .95])
    return _b64(fig), summary


def build_eyedata(reader, regime, res):
    """Per-text old/new values for the filterable Eye-measures table. Compact, aligned by the
    row-identity key (text_id, + align_idx at sentence level — see _rt_key_cols):
    {ids:[...], cols:{f'{lvl}_{eye}': {'o':[...], 'n':[...]}}} (old reindexed onto the new rows).
    ids carry the align_idx ('text_id#align_idx') at sentence level so the rows stay distinct."""
    path = CORR / reader / regime / f"RT_all_metrics_df_{res}.csv"
    if not path.exists():
        return None
    new = pd.read_csv(path)
    old = git_head_df(path)
    if old is None or "text_id" not in new.columns:
        return None
    key = _rt_key_cols(old, new)
    o = old.set_index(key)
    n = new.set_index(key)
    cols = {}
    for eye in EYE_METRICS:
        for lvl in LEVELS_EYE:
            c = f"{lvl}_{eye}"
            if c not in n.columns:
                continue
            ov = o[c].reindex(n.index) if c in o.columns else pd.Series(index=n.index, dtype=float)
            cols[c] = {"o": [_r(v, 3) for v in ov.tolist()], "n": [_r(v, 3) for v in n[c].tolist()]}
    ids = ["#".join(str(p) for p in t) if isinstance(t, tuple) else str(t) for t in n.index.tolist()]
    return {"ids": ids, "cols": cols}


# Rankings are computed CLIENT-SIDE in the browser from the embedded DATA array (each headline cell
# already carries r_old / r_new = pearson_corr_all per eye×level), so they react live to filters and
# to the chosen predictor set (MAIN_PREDS by default, + any supplementary cols you add). No per-eye
# file loading needed here.


# ----------------------------------------------------------------------------- PDF figures (separate file)
def _pdf_png_b64(pdf_bytes: bytes, dpi: int = 96) -> str | None:
    """First page of a PDF -> base64 PNG (renders inline in any viewer, unlike an embedded PDF)."""
    if fitz is None or not pdf_bytes:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
        doc.close()
        return base64.b64encode(png).decode()
    except Exception:
        return None


def pdf_entries(readers, regimes) -> list[dict]:
    """Changed/untracked figure PDFs, each rasterised to a PNG (old HEAD + new working tree)."""
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--", "src/Correlations"]
    ).decode().splitlines()
    entries = []
    for line in out:
        p = line[3:].strip()
        if not p.lower().endswith(".pdf"):
            continue
        parts = Path(p).parts
        if len(parts) < 5:
            continue
        reader, regime = parts[2], parts[3]
        if readers and reader not in readers:
            continue
        if regimes and regime not in regimes:
            continue
        repo_pdf = REPO / p
        if not repo_pdf.exists():
            continue
        new_png = _pdf_png_b64(repo_pdf.read_bytes())
        old_png = _pdf_png_b64(git_head_bytes(repo_pdf) or b"")
        entries.append({"rd": reader, "rg": regime, "name": Path(p).name,
                        "old": old_png or "", "new": new_png or ""})
    return entries


def render_pdf_html(pdfs, stamp) -> str:
    """Standalone old-vs-new PDF figure viewer. PNG previews are embedded base64, so it renders
    inline in ANY viewer (incl. the VSCode preview) — no http server and no save-as prompt."""
    readers = sorted({p["rd"] for p in pdfs})
    regimes = sorted({p["rg"] for p in pdfs})

    def opts(vals):
        return "".join(f"<option value='{v}'>{v}</option>" for v in vals)

    html = """<!doctype html><html><head><meta charset="utf-8"><title>PDF figures — old vs new</title><style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#222}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.7rem 1rem;z-index:5}
 h1{font-size:1.1rem;margin:.1rem 0} .controls{display:flex;gap:.8rem;flex-wrap:wrap;align-items:center;margin-top:.4rem;font-size:.85rem}
 select{font-size:.85rem;padding:.15rem} main{padding:1rem}
 .pair{display:flex;gap:1rem;align-items:flex-start} .pane{flex:1;min-width:0}
 .pane h2{font-size:.9rem;margin:.2rem 0;color:#555} .pane img{width:100%;border:1px solid #ddd;background:#fff;border-radius:6px}
 .count{color:#666;font-size:.82rem} .miss{color:#b71c1c;padding:2rem;text-align:center}
</style></head><body>
 <header><h1>PDF figures — old (HEAD) vs new (working tree) <small style="color:#888">· __STAMP__</small></h1>
 <div class="controls">
   <label>population <select id="rd">__RD__</select></label>
   <label>regime <select id="rg">__RG__</select></label>
   <label>figure <select id="fig" style="min-width:32rem"></select></label>
   <span class="count" id="count"></span>
 </div></header>
 <main><div class="pair">
   <div class="pane"><h2>OLD (HEAD)</h2><div id="old"></div></div>
   <div class="pane"><h2>NEW (working tree)</h2><div id="new"></div></div>
 </div></main>
<script>
const PDFS = __PDFS__; const $=s=>document.querySelector(s);
function list(){ return PDFS.filter(p=>p.rd===$("#rd").value && p.rg===$("#rg").value); }
function img(b64){ return b64? `<img src="data:image/png;base64,${b64}">` : `<div class="miss">(not available)</div>`; }
function show(){ const l=list(), p=l[$("#fig").value]; $("#old").innerHTML=p?img(p.old):""; $("#new").innerHTML=p?img(p.new):""; }
function fillFig(){ const l=list(); $("#fig").innerHTML=l.map((p,i)=>`<option value="${i}">${p.name}</option>`).join("")||"<option>(none)</option>";
  $("#count").textContent=l.length+" changed figures"; show(); }
["rd","rg"].forEach(id=>$("#"+id).onchange=fillFig); $("#fig").onchange=show; fillFig();
</script></body></html>"""
    return (html.replace("__RD__", opts(readers)).replace("__RG__", opts(regimes))
                .replace("__STAMP__", stamp).replace("__PDFS__", json.dumps(pdfs, separators=(",", ":"))))


# ----------------------------------------------------------------------------- HTML
def render_html(data, folds, corr_plots, eye_plots, eye_summ, eyedata, combos, stamp, marker_banner="") -> str:
    readers = sorted({c[0] for c in combos})
    regimes = sorted({c[1] for c in combos})
    resos = sorted({c[2] for c in combos})
    eyes = sorted({d["eye"] for d in data})
    preds = sorted({d["pred"] for d in data})
    main_present = [p for p in MAIN_PREDS if p in set(preds)]      # paper "main" predictors present
    supp_preds = [p for p in preds if p not in set(MAIN_PREDS)]    # everything else (supplementary)
    default_rd = "L1_and_L2" if "L1_and_L2" in readers else readers[0] if readers else ""
    default_rg = "FirstReading" if "FirstReading" in regimes else regimes[0] if regimes else ""
    default_rs = "paragraph" if "paragraph" in resos else resos[0] if resos else ""

    def opts(vals, sel=None, alllabel="(all)"):
        s = f"<option value='__all__'>{alllabel}</option>" if alllabel else ""
        for v in vals:
            s += f"<option value='{v}'{' selected' if v == sel else ''}>{v}</option>"
        return s

    # pre-rendered, combo-tagged blocks (filtered by show/hide)
    def tagged(combo_blocks):
        out = ""
        for (rd, rg, rs), html in combo_blocks.items():
            out += (f"<div class='combo' data-rd='{rd}' data-rg='{rg}' data-rs='{rs}'>"
                    f"<h3 class='combohdr'>{rd} · {rg} · {rs}</h3>{html}</div>")
        return out

    corr_blocks = {c: (f"<img src='data:image/png;base64,{img}'>" if img else "<p>no data</p>")
                   for c, img in corr_plots.items()}
    eye_blocks = {}
    for c in eye_plots:
        img = eye_plots[c]
        summ = eye_summ.get(c, [])
        rows = "".join(
            f"<tr><td>{s['eye']}</td><td class='num'>{s['n']}</td><td class='num'>{s['mean_d']}</td>"
            f"<td class='num'>{s['max_abs_d']}</td><td class='num'>{s['corr']}</td></tr>" for s in summ)
        tbl = (f"<table class='small'><tr><th>eye metric</th><th>n texts</th><th>mean Δ</th>"
               f"<th>max|Δ|</th><th>corr(old,new)</th></tr>{rows}</table>")
        eye_blocks[c] = (f"<img src='data:image/png;base64,{img}'>" if img else "") + tbl

    legend = " ".join(
        f"<span class='chip' style='background:{CAT_COLOR[c]}'>{CAT_LABEL[c]}</span>" for c in CAT_ORDER)
    fam_legend = " · ".join(f"<b style='color:{FAM_COLOR[f]}'>{f}</b>" for f in FAM_COLOR)

    data_json = json.dumps(data, separators=(",", ":"))
    folds_json = json.dumps(folds, separators=(",", ":"))
    eyedata_json = json.dumps({f"{rd}|{rg}|{rs}": eyedata.get((rd, rg, rs))
                               for (rd, rg, rs) in combos}, separators=(",", ":"))

    head = """<!doctype html><html><head><meta charset="utf-8"><title>Correlations old vs new review</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#222;background:#fafafa}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.7rem 1rem;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.05)}
 h1{font-size:1.15rem;margin:.1rem 0}
 .filters{display:flex;gap:.8rem;flex-wrap:wrap;align-items:center;margin-top:.4rem}
 .filters label{font-size:.8rem;color:#555} select{font-size:.85rem;padding:.15rem}
 .tabs{display:flex;gap:.3rem;margin-top:.5rem}
 .tabs button{border:1px solid #ccc;background:#f3f3f3;padding:.35rem .8rem;border-radius:6px 6px 0 0;cursor:pointer;font-size:.85rem}
 .tabs button.active{background:#1565c0;color:#fff;border-color:#1565c0}
 main{padding:1rem}
 .tab{display:none} .tab.active{display:block}
 .chip{color:#fff;padding:.12rem .5rem;border-radius:10px;font-size:.72rem;margin-right:.3rem;white-space:nowrap}
 .bars{margin:.5rem 0 1rem} .barrow{display:flex;align-items:center;gap:.6rem;margin:.25rem 0}
 .barlbl{flex:0 0 260px;font-size:.82rem} .bartrack{flex:1;background:#eee;border-radius:4px;overflow:hidden}
 .barfill{color:#fff;font-size:.78rem;padding:.18rem .5rem;white-space:nowrap;border-radius:4px;min-width:2.5rem}
 table{border-collapse:collapse;width:100%;font-size:.82rem;background:#fff;margin:.3rem 0 1rem}
 th,td{border:1px solid #e2e2e2;padding:.25rem .45rem;text-align:left} th{background:#fafafa;position:sticky;top:0;cursor:pointer}
 td.num{text-align:right;font-variant-numeric:tabular-nums} h3.combohdr{font-size:.95rem;color:#444;border-left:4px solid #1565c0;padding-left:.5rem;margin:1rem 0 .4rem}
 .up{color:#2e7d32} .dn{color:#c62828} .eq{color:#aaa}
 #r_extra{vertical-align:middle} #r_body small,.eye small{color:#aaa}
 img{max-width:100%;border:1px solid #eee;background:#fff;border-radius:6px}
 .fold{background:#f7fbff;font-size:.76rem} .fold td{border-color:#dbeeff}
 .expand{cursor:pointer;color:#1565c0;font-weight:600;user-select:none}
 .controls{margin:.4rem 0;font-size:.85rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
 .count{color:#666;font-size:.8rem}
 .recs{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:.6rem .9rem;margin-bottom:1.2rem}
 .recs h3{margin:.2rem 0 .5rem} .reccol{display:inline-block;vertical-align:top;margin-right:2rem;max-width:48%}
 .recbtns{display:flex;flex-wrap:wrap;gap:.5rem;margin:.4rem 0}
 .recbtn{cursor:pointer;border:1px solid #cdd7e0;background:#f5f9ff;border-radius:8px;padding:.45rem .7rem;font-size:.77rem;text-align:left;line-height:1.3}
 .recbtn:hover{background:#e3f0ff;border-color:#1565c0} .rbnum{font-weight:700;color:#c62828;font-size:.95rem}
 small{color:#888}
</style></head><body>"""

    header = f"""<header>
 <h1>Correlations — old (HEAD) vs new (working tree) review <small>· generated {stamp}</small></h1>
 <div class="filters">
   <label>population <select id="f_rd">{opts(readers, default_rd, None)}</select></label>
   <label>regime <select id="f_rg">{opts(regimes, default_rg, None)}</select></label>
   <label>resolution <select id="f_rs">{opts(resos, default_rs, None)}</select></label>
   <span style="flex:1"></span>
   {legend}
 </div>
 <div class="tabs">
   <button data-tab="coverage" class="active">Coverage</button>
   <button data-tab="corr">Correlations</button>
   <button data-tab="rank">Rankings</button>
   <button data-tab="eye">Eye measures</button>
   <button data-tab="dist">Distributions</button>
 </div>
</header>"""

    coverage = """<section class="tab active" id="tab-coverage">
   <div class="recs" id="recs"></div>
   <h3>Coverage for the selected population</h3>
   <p class="count" id="cov_total"></p>
   <div class="bars" id="cov_bars"></div>
   <h3>Movement by eye metric × level (count of material+sig+sign changes)</h3>
   <div id="cov_matrix"></div>
 </section>"""

    corr_tab = f"""<section class="tab" id="tab-corr">
   <div class="controls">
     <label>eye metric <select id="c_eye">{opts(eyes, alllabel='(all)')}</select></label>
     <label>level <select id="c_lvl">{opts(LEVELS_CORR, alllabel='(all)')}</select></label>
     <label>family <select id="c_fam">{opts(list(FAM_COLOR), alllabel='(all)')}</select></label>
     <label>category <select id="c_cat">
        <option value="__changed__" selected>changed only (material+sig+sign)</option>
        <option value="__all__">(all)</option>
        {''.join(f"<option value='{c}'>{CAT_LABEL[c]}</option>" for c in CAT_ORDER)}
     </select></label>
     <label>search <input id="c_q" placeholder="predictor…" style="font-size:.85rem"></label>
     <span class="count" id="c_count"></span>
   </div>
   <p class="count">row stripe &amp; chip colour = change category (legend above) · predictor-name colour = family: {fam_legend}</p>
   <table id="c_table"><thead><tr>
     <th></th><th data-k="eye">eye</th><th data-k="pred">predictor</th><th data-k="lvl">lvl</th>
     <th data-k="cat">category</th><th data-k="ro" class="num">r old</th><th data-k="rn" class="num">r new</th>
     <th data-k="dr" class="num">Δr</th><th>sig old→new</th><th data-k="bo" class="num">boot old</th>
     <th data-k="bn" class="num">boot new</th><th class="num">CI old→new</th>
   </tr></thead><tbody id="c_body"></tbody></table>
 </section>"""

    rank_tab = f"""<section class="tab" id="tab-rank">
   <div class="controls">
     <label>eye metric <select id="r_eye">{opts(eyes, sel=eyes[0] if eyes else None, alllabel=None)}</select></label>
     <label>level <select id="r_lvl">{opts(LEVELS_CORR, sel='diff', alllabel=None)}</select></label>
     <label>show <select id="r_set">
        <option value="main" selected>main text cols only ({len(main_present)})</option>
        <option value="all">main + all supplementary ({len(preds)})</option>
        <option value="custom">main + selected below</option>
     </select></label>
     <label>add cols <select id="r_extra" multiple size="1" style="min-width:14rem">{opts(supp_preds, alllabel=None)}</select></label>
     <span class="count" id="r_count"></span>
   </div>
   <p class="count">Ranked by |full-sample r| (pearson_corr_all) within the chosen predictor set, for the
   selected population · eye metric · level. Predictor-name colour = family: {fam_legend}.
   ↑/↓ = rank moved up/down old→new.</p>
   <div id="r_body"></div>
 </section>"""
    eye_tab = f"""<section class="tab" id="tab-eye">
   <h3>Per-text values (old vs new) — the RT_all_metrics input that drove the changes</h3>
   <div class="controls">
     <label>eye metric <select id="e_eye">{opts(EYE_METRICS, sel=EYE_METRICS[0], alllabel=None)}</select></label>
     <label>level <select id="e_lvl">{opts(LEVELS_EYE, sel='diff', alllabel=None)}</select></label>
     <label><input type="checkbox" id="e_changed"> changed only (|Δ|&gt;1e-6)</label>
     <label>search text_id <input id="e_q" placeholder="text_id…" style="font-size:.85rem"></label>
     <span class="count" id="e_count"></span>
   </div>
   <table id="e_table"><thead><tr>
     <th data-k="id">text_id</th><th data-k="o" class="num">old value</th>
     <th data-k="n" class="num">new value</th><th data-k="d" class="num">Δ (new−old)</th>
     <th data-k="pct" class="num">Δ %</th></tr></thead><tbody id="e_body"></tbody></table>
   <h3>Distribution per eye metric (old vs new, all levels)</h3>
   {tagged(eye_blocks)}
 </section>"""
    dist_tab = f"<section class='tab' id='tab-dist'><h3>Correlation value distributions</h3>{tagged(corr_blocks)}</section>"

    script = """<script>
const DATA = __DATA__, FOLDS = __FOLDS__, EYEDATA = __EYEDATA__, MAIN_PREDS = __MAINPREDS__;
const CAT_COLOR = __CATCOLOR__, CAT_LABEL = __CATLABEL__, CAT_ORDER = __CATORDER__;
const FAM_COLOR = {surprisal:'#1565c0','llm-rating':'#ef6c00',readability:'#2e7d32'};
const MAIN_SET = new Set(MAIN_PREDS);
const $ = s => document.querySelector(s);
let sortKey = "dr", sortDir = -1, eSortKey = "d", eSortDir = -1;
// pearson on rank-vectors == Spearman (our ranks are tie-free strict orderings).
function spearman(a,b){ const n=a.length; if(n<3) return null;
  const ma=a.reduce((s,v)=>s+v,0)/n, mb=b.reduce((s,v)=>s+v,0)/n; let num=0,da=0,db=0;
  for(let i=0;i<n;i++){ const x=a[i]-ma, y=b[i]-mb; num+=x*y; da+=x*x; db+=y*y; }
  return (da&&db)? num/Math.sqrt(da*db):null; }

function gf(){ return {rd:$("#f_rd").value, rg:$("#f_rg").value, rs:$("#f_rs").value}; }
function popRows(){ const g=gf(); return DATA.filter(d=> d.rd===g.rd && d.rg===g.rg && d.rs===g.rs); }

// ---- tabs
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".tabs button").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); $("#tab-"+b.dataset.tab).classList.add("active");
});

// ---- global filters
["f_rd","f_rg","f_rs"].forEach(id=>$("#"+id).onchange=()=>{ applyComboVis(); renderCoverage(); renderTable(); renderRank(); renderEye(); });
function applyComboVis(){ const g=gf();
  document.querySelectorAll(".combo").forEach(el=>{
    el.style.display = (el.dataset.rd===g.rd && el.dataset.rg===g.rg && el.dataset.rs===g.rs) ? "" : "none";
  });
}
function isChanged(d){ return d.cat==="material"||d.cat==="sig-change"||d.cat==="sign-flip"; }
function activateTab(name){
  document.querySelectorAll(".tabs button").forEach(x=>x.classList.toggle("active",x.dataset.tab===name));
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  $("#tab-"+name).classList.add("active");
}

// ---- "Start here" recommendations (computed across all populations, independent of current filter)
function renderRecs(){
  const byPop={}, byHot={};
  DATA.forEach(d=>{ const k=d.rd+'|'+d.rg+'|'+d.rs; (byPop[k]=byPop[k]||{t:0,c:0}).t++; if(isChanged(d)){byPop[k].c++;
     const hk=[d.rd,d.rg,d.rs,d.eye,d.lvl].join('|'); byHot[hk]=(byHot[hk]||0)+1; } });
  const pops=Object.entries(byPop).map(([k,v])=>({k,...v})).filter(p=>p.c>0).sort((a,b)=>b.c-a.c);
  const hots=Object.entries(byHot).map(([k,c])=>({k,c})).sort((a,b)=>b.c-a.c).slice(0,12);
  let h="<h3>▶ Start here — most changes first (one click sets the filters)</h3>";
  if(!pops.length){ h+="<p class='count'>No material / significance / sign changes in any population — everything is identical or float-noise.</p>"; $("#recs").innerHTML=h; return; }
  h+="<div class='reccol'><b>By population</b><div class='recbtns'>";
  pops.forEach(p=>{ const [rd,rg,rs]=p.k.split('|'); const pct=(100*p.c/p.t).toFixed(0);
    h+=`<button class="recbtn" onclick="setView('${rd}','${rg}','${rs}')"><span class="rbnum">${p.c.toLocaleString()}</span> changed (${pct}%)<br><b>${rd}</b> · ${rg} · ${rs}</button>`; });
  h+="</div></div><div class='reccol'><b>Top hotspots (eye metric × level)</b><div class='recbtns'>";
  hots.forEach(o=>{ const [rd,rg,rs,eye,lvl]=o.k.split('|');
    h+=`<button class="recbtn" onclick="setView('${rd}','${rg}','${rs}','${eye}','${lvl}')"><span class="rbnum">${o.c.toLocaleString()}</span> changed<br><b>${eye}</b> · ${lvl}<br><small>${rd} · ${rg} · ${rs}</small></button>`; });
  h+="</div></div>"; $("#recs").innerHTML=h;
}
function setView(rd,rg,rs,eye,lvl){
  $("#f_rd").value=rd; $("#f_rg").value=rg; $("#f_rs").value=rs;
  applyComboVis(); renderCoverage(); renderRank(); renderEye();
  if(eye){ $("#c_eye").value=eye; $("#c_lvl").value=lvl; $("#c_cat").value="__changed__"; renderTable(); activateTab("corr"); }
  else { renderTable(); activateTab("coverage"); }
  window.scrollTo(0,0);
}

// ---- rankings (computed live from DATA over the chosen predictor set)
["r_eye","r_lvl","r_set"].forEach(id=>{ const el=$("#"+id); if(el) el.onchange=renderRank; });
{ const ex=$("#r_extra"); if(ex) ex.onchange=()=>{ $("#r_set").value="custom"; renderRank(); }; }
function rankSet(){
  const mode=$("#r_set").value;
  if(mode==="all") return null;                       // null = every predictor
  const set=new Set(MAIN_SET);
  if(mode==="custom") Array.from($("#r_extra").selectedOptions).forEach(o=>set.add(o.value));
  return set;
}
function renderRank(){
  const g=gf(), eye=$("#r_eye").value, lvl=$("#r_lvl").value, set=rankSet();
  let rows=DATA.filter(d=> d.rd===g.rd && d.rg===g.rg && d.rs===g.rs && d.eye===eye && d.lvl===lvl);
  if(set) rows=rows.filter(d=> set.has(d.pred));
  const rankBy=(arr,key)=>{ const m={}; arr.filter(d=>d[key]!=null).sort((a,b)=>Math.abs(b[key])-Math.abs(a[key]))
      .forEach((d,i)=> m[d.pred]=i+1); return m; };
  const rkO=rankBy(rows,"ro"), rkN=rankBy(rows,"rn");
  const preds=[...new Set(rows.map(d=>d.pred))];
  const recs=preds.map(p=>{ const d=rows.find(x=>x.pred===p);
    return {pred:p, fam:d.fam, ro:d.ro, rn:d.rn, rkO:rkO[p]??null, rkN:rkN[p]??null}; });
  recs.sort((a,b)=> (a.rkN??1e9)-(b.rkN??1e9));
  const both=recs.filter(r=> r.rkO!=null && r.rkN!=null);
  const rho=spearman(both.map(r=>r.rkO), both.map(r=>r.rkN));
  let body="";
  recs.forEach(r=>{ const dr=(r.rkO!=null&&r.rkN!=null)? r.rkO-r.rkN : null;
    const arr = dr===null?"":dr>0?`<span class="up">↑${dr}</span>`:dr<0?`<span class="dn">↓${-dr}</span>`:`<span class="eq">=</span>`;
    body+=`<tr><td class="num">${r.rkN??"–"}</td><td style="color:${FAM_COLOR[r.fam]}">${r.pred}${MAIN_SET.has(r.pred)?"":" <small>(supp)</small>"}</td>`
        +`<td class="num">${r.rkO??"–"} → ${r.rkN??"–"} ${arr}</td>`
        +`<td class="num">${r.ro==null?"":(r.ro>=0?"+":"")+r.ro.toFixed(3)}</td>`
        +`<td class="num">${r.rn==null?"":(r.rn>=0?"+":"")+r.rn.toFixed(3)}</td></tr>`; });
  $("#r_count").textContent=`${recs.length} predictors · rank-Spearman(old,new)=${rho==null?"n/a":rho.toFixed(3)}`;
  $("#r_body").innerHTML=`<table style="max-width:760px"><thead><tr><th>rank<br>(new)</th><th>predictor</th>`
    +`<th>old → new rank</th><th class="num">r old</th><th class="num">r new</th></tr></thead><tbody>${body}</tbody></table>`;
}

// ---- eye-measures per-text table (from EYEDATA)
["e_eye","e_lvl","e_changed","e_q"].forEach(id=>{ const el=$("#"+id); if(el){ el.oninput=renderEye; el.onchange=renderEye; } });
{ const t=$("#e_table"); if(t) t.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(eSortKey===k) eSortDir*=-1; else {eSortKey=k; eSortDir=-1;} renderEye(); }); }
function renderEye(){
  const g=gf(), ed=EYEDATA[g.rd+"|"+g.rg+"|"+g.rs];
  if(!ed){ $("#e_body").innerHTML=""; $("#e_count").textContent="no eye data for this population"; return; }
  const col=ed.cols[$("#e_lvl").value+"_"+$("#e_eye").value];
  if(!col){ $("#e_body").innerHTML=""; $("#e_count").textContent="no data for this metric/level"; return; }
  const changed=$("#e_changed").checked, q=$("#e_q").value.toLowerCase(); let recs=[];
  for(let i=0;i<ed.ids.length;i++){ const o=col.o[i], n=col.n[i]; if(o==null&&n==null) continue;
    const d=(o!=null&&n!=null)? n-o : null;
    if(changed && (d===null || Math.abs(d)<=1e-6)) continue;
    if(q && !ed.ids[i].toLowerCase().includes(q)) continue;
    recs.push({id:ed.ids[i], o, n, d, pct:(d!=null&&o)? 100*d/Math.abs(o):null}); }
  recs.sort((a,b)=>{ let x=a[eSortKey], y=b[eSortKey]; if(eSortKey==="d"){ x=Math.abs(x||0); y=Math.abs(y||0); }
    if(x==null)x=-1e18; if(y==null)y=-1e18; if(typeof x==="string") return eSortDir*x.localeCompare(y); return eSortDir*(x-y); });
  const cap=800, shown=recs.slice(0,cap);
  $("#e_count").textContent=`${recs.length.toLocaleString()} texts`+(recs.length>cap?` (first ${cap})`:"");
  let html=""; shown.forEach(r=>{ const c=r.d===null?"":Math.abs(r.d)<=1e-6?"#2e7d32":"#c62828";
    html+=`<tr><td>${r.id}</td><td class="num">${r.o==null?"":r.o}</td><td class="num">${r.n==null?"":r.n}</td>`
        +`<td class="num" style="color:${c}"><b>${r.d===null?"":(r.d>=0?"+":"")+r.d.toFixed(3)}</b></td>`
        +`<td class="num">${r.pct===null?"":(r.pct>=0?"+":"")+r.pct.toFixed(1)+"%"}</td></tr>`; });
  $("#e_body").innerHTML=html;
}

// ---- coverage
function renderCoverage(){
  const rows = popRows(), total = rows.length;
  const counts = {}; CAT_ORDER.forEach(c=>counts[c]=0);
  rows.forEach(d=>counts[d.cat]=(counts[d.cat]||0)+1);
  $("#cov_total").textContent = `${total.toLocaleString()} headline correlations (agg_folds pearson_corr_all) for this population — buckets below sum to it.`;
  let bars="";
  CAT_ORDER.forEach(c=>{ const v=counts[c]||0; const w=total? 100*v/total:0;
    bars += `<div class="barrow"><div class="barlbl">${CAT_LABEL[c]}</div>
      <div class="bartrack"><div class="barfill" style="width:${Math.max(w,2).toFixed(1)}%;background:${CAT_COLOR[c]}">${v.toLocaleString()} (${w.toFixed(0)}%)</div></div></div>`; });
  $("#cov_bars").innerHTML = bars;
  // matrix eye x level of changed
  const eyes=[...new Set(rows.map(d=>d.eye))].sort(), lvls=["Adv","Ele","diff","all"];
  const ch = d=>["material","sig-change","sign-flip"].includes(d.cat);
  let h="<table class='small'><tr><th>eye \\\\ level</th>"+lvls.map(l=>`<th>${l}</th>`).join("")+"<th>total</th></tr>";
  eyes.forEach(e=>{ let rt=0; h+=`<tr><td>${e}</td>`;
    lvls.forEach(l=>{ const n=rows.filter(d=>d.eye===e&&d.lvl===l&&ch(d)).length; rt+=n;
      const bg=n? `style="background:rgba(197,40,40,${Math.min(.15+n*0.08,.8)});color:#fff"`:""; h+=`<td class="num" ${bg}>${n||""}</td>`; });
    h+=`<td class="num"><b>${rt||""}</b></td></tr>`; });
  h+="</table>"; $("#cov_matrix").innerHTML=h;
}

// ---- correlations table
["c_eye","c_lvl","c_fam","c_cat","c_q"].forEach(id=>{ const el=$("#"+id); el.oninput=renderTable; el.onchange=renderTable; });
document.querySelectorAll("#c_table th[data-k]").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; if(sortKey===k) sortDir*=-1; else {sortKey=k; sortDir=-1;} renderTable();
});
function fmt(v){ return v===null||v===undefined? "" : (typeof v==="number"? (v>=0?"+":"")+v.toFixed(3): v); }
function renderTable(){
  let rows = popRows();
  const eye=$("#c_eye").value, lvl=$("#c_lvl").value, fam=$("#c_fam").value, cat=$("#c_cat").value, q=$("#c_q").value.toLowerCase();
  rows = rows.filter(d=>{
    if(eye!=="__all__" && d.eye!==eye) return false;
    if(lvl!=="__all__" && d.lvl!==lvl) return false;
    if(fam!=="__all__" && d.fam!==fam) return false;
    if(cat==="__changed__"){ if(!["material","sig-change","sign-flip"].includes(d.cat)) return false; }
    else if(cat!=="__all__" && d.cat!==cat) return false;
    if(q && !d.pred.toLowerCase().includes(q)) return false;
    return true;
  });
  rows.sort((a,b)=>{ let x=a[sortKey],y=b[sortKey];
    if(sortKey==="dr"){ x=Math.abs(x||0); y=Math.abs(y||0); }
    if(x===null||x===undefined)x=-1e9; if(y===null||y===undefined)y=-1e9;
    if(typeof x==="string") return sortDir*x.localeCompare(y);
    return sortDir*(x-y); });
  const cap=600, shown=rows.slice(0,cap);
  $("#c_count").textContent = `${rows.length.toLocaleString()} rows`+(rows.length>cap?` (showing first ${cap})`:"");
  let html="";
  shown.forEach((d,i)=>{
    const key=`${d.rd}|${d.rg}|${d.rs}|${d.eye}|${d.pred}|${d.lvl}`;
    const hasF = FOLDS[key]!==undefined;
    const exp = hasF? `<span class="expand" data-key="${key}" data-row="${i}">▶</span>`:"";
    html += `<tr style="border-left:4px solid ${CAT_COLOR[d.cat]}">
      <td>${exp}</td><td>${d.eye}</td><td style="color:${({surprisal:'#1565c0','llm-rating':'#ef6c00',readability:'#2e7d32'})[d.fam]}">${d.pred}</td>
      <td>${d.lvl}</td><td><span class="chip" style="background:${CAT_COLOR[d.cat]}">${d.cat}</span></td>
      <td class="num">${fmt(d.ro)}</td><td class="num">${fmt(d.rn)}</td>
      <td class="num"><b>${fmt(d.dr)}</b></td><td>${d.so||"–"} → ${d.sn||"–"}</td>
      <td class="num">${fmt(d.bo)}</td><td class="num">${fmt(d.bn)}</td><td class="num">${fmt(d.cio)} → ${fmt(d.cin)}</td></tr>`;
  });
  $("#c_body").innerHTML = html;
  document.querySelectorAll("#c_body .expand").forEach(s=>s.onclick=()=>toggleFold(s));
}
function toggleFold(span){
  const key=span.dataset.key, tr=span.closest("tr");
  if(tr.nextSibling && tr.nextSibling.classList && tr.nextSibling.classList.contains("foldrow")){
    tr.nextSibling.remove(); span.textContent="▶"; return; }
  span.textContent="▼"; const f=FOLDS[key];
  let cells = f.lab.map((l,i)=>{ const o=f.o[i], n=f.n[i], dd=(o!==null&&n!==null)?(n-o):null;
    const c = dd===null?"":Math.abs(dd)>0.01?"#c62828":Math.abs(dd)>1e-9?"#ef6c00":"#2e7d32";
    return `<td>${l}</td><td class="num">${o===null?"":o.toFixed(3)}</td><td class="num">${n===null?"":n.toFixed(3)}</td><td class="num" style="color:${c}">${dd===null?"":(dd>=0?"+":"")+dd.toFixed(3)}</td>`; });
  // chunk into a small table fold/old/new/Δ
  let body=""; for(let i=0;i<f.lab.length;i++){ body+="<tr>"+cells[i]+"</tr>"; }
  const ntr=document.createElement("tr"); ntr.className="foldrow";
  ntr.innerHTML=`<td></td><td colspan="11"><b>per-fold (full-sample 'all' + 10 CV folds)</b>
    <table class="fold"><tr><th>fold</th><th>r old</th><th>r new</th><th>Δ</th></tr>${body}</table></td>`;
  tr.after(ntr);
}

applyComboVis(); renderRecs(); renderCoverage(); renderTable(); renderRank(); renderEye();
</script>"""

    script = (script.replace("__DATA__", data_json).replace("__FOLDS__", folds_json)
              .replace("__EYEDATA__", eyedata_json).replace("__MAINPREDS__", json.dumps(main_present))
              .replace("__CATCOLOR__", json.dumps(CAT_COLOR))
              .replace("__CATLABEL__", json.dumps(CAT_LABEL)).replace("__CATORDER__", json.dumps(CAT_ORDER)))

    return (head + header + "<main>" + marker_banner + coverage + corr_tab + rank_tab + eye_tab
            + dist_tab + "</main>" + script + "</body></html>")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readers", nargs="*", default=None, help="restrict populations (default: all changed)")
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--resolutions", nargs="*", default=None)
    ap.add_argument("--no-pdfs", action="store_true", help="skip building the separate pdf_compare file")
    args = ap.parse_args()

    # Wrapper marker status — surface an incomplete regeneration before anything else.
    _, missing_markers = wrapper_marker_status(WRAPPER_DONE_DIR, WRAPPER_TASKS)
    marker_banner = wrapper_markers_banner_html(WRAPPER_DONE_DIR, WRAPPER_TASKS, "Correlations")
    if missing_markers:
        print(f"⚠ wrapper regeneration INCOMPLETE — {len(missing_markers)} marker(s) NOT done: "
              + ", ".join(missing_markers))
    else:
        print("✓ all wrapper markers present")

    combos = discover_combos(args.readers, args.regimes, args.resolutions)
    if not combos:
        print("No changed agg_folds combos found (nothing to review)."); return
    print(f"Building review for {len(combos)} combo(s): "
          + ", ".join("/".join(c) for c in combos))

    data, folds, corr_plots, eye_plots, eye_summ, eyedata = [], {}, {}, {}, {}, {}
    for (rd, rg, rs) in combos:
        recs, fl = build_headline(rd, rg, rs)
        data += recs; folds.update(fl)
        corr_plots[(rd, rg, rs)] = plot_correlations(recs)
        img, summ = plot_eye_measures(rd, rg, rs)
        eye_plots[(rd, rg, rs)] = img; eye_summ[(rd, rg, rs)] = summ
        eyedata[(rd, rg, rs)] = build_eyedata(rd, rg, rs)
        print(f"  {rd}/{rg}/{rs}: {len(recs)} cells, {sum(1 for r in recs if r['cat'] in ('material','sig-change','sign-flip'))} changed")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = render_html(data, folds, corr_plots, eye_plots, eye_summ, eyedata, combos, stamp,
                       marker_banner=marker_banner)
    out = RESULTS / f"review_old_vs_new_{stamp}.html"
    out.write_text(html)
    nchg = sum(1 for r in data if r["cat"] in ("material", "sig-change", "sign-flip"))
    print(f"\nWrote {out.relative_to(REPO)}")
    print(f"  {len(data):,} headline cells · {nchg:,} changed (material/sig/sign) · {len(folds):,} fold drilldowns")
    print(f"  open: {out}")

    if nchg:
        summary = [f"{nchg:,} of {len(data):,} headline correlation cell(s) changed (material / sig-change / sign-flip).",
                   "scope: " + ", ".join("/".join(c) for c in combos) + "."]
        if missing_markers:
            summary.insert(0, "⚠ wrapper regeneration INCOMPLETE — markers not done: "
                           + ", ".join(missing_markers) + ".")
        write_review_reminder(RESULTS, out, "Correlations outputs were regenerated", summary,
                              "REVIEW_correlations_before_commit.txt")

    # PDF figure comparison goes to its OWN self-contained file (rasterised PNGs, renders anywhere).
    if not args.no_pdfs:
        if fitz is None:
            print("  PDF compare: skipped (PyMuPDF/fitz not installed)")
        else:
            pdfs = pdf_entries(args.readers, args.regimes)
            if pdfs:
                pout = RESULTS / f"pdf_compare_{stamp}.html"
                pout.write_text(render_pdf_html(pdfs, stamp))
                print(f"  PDF compare: {pout.relative_to(REPO)}  ({len(pdfs)} figures, old | new)")
            else:
                print("  PDF compare: no changed figure PDFs for the selected scope")


if __name__ == "__main__":
    main()
