"""Turn raw results into every number, table and figure used in the paper.

Usage: .venv/Scripts/python.exe scripts/analyze.py
Inputs : results/raw/{rq1_scaling,rq3_retrieval,rq3_k_ablation}/*.jsonl
Outputs: results/processed/all_records.csv, results/processed/summary.json,
         paper/tables/*.tex, paper/figures/*.pdf
Works on partial results (prints what is missing), so it can be run mid-experiment.
Nothing in the paper is typed by hand: re-run this script and recompile.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.bfcl_checker import simple_function_checker  # noqa: E402
from toolscale.data import build_catalogs, checker_description, eval_subset, function_pool, load_queries  # noqa: E402,E501
from toolscale.retrieval import ToolRetriever  # noqa: E402
from toolscale.scoring import coerce_args, score  # noqa: E402

PROC, TAB, FIG = ROOT / "results/processed", ROOT / "paper/tables", ROOT / "paper/figures"
for d in (PROC, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

# Fixed model order + colour (validated categorical slots 1-4, light mode) + marker (secondary encoding)
MODELS = {
    "qwen2.5:1.5b-instruct": ("Qwen2.5-1.5B", "#2a78d6", "o"),
    "qwen2.5:3b-instruct": ("Qwen2.5-3B", "#eb6834", "s"),
    "llama3.2:latest": ("Llama-3.2-3B", "#1baf7a", "^"),
    "qwen3.5:2b": ("Qwen3.5-2B", "#eda100", "D"),
}
# Failure categories folded into 5 groups for figures (full breakdown in the table)
GROUPS = {
    "Wrong / hallucinated tool": ["wrong_tool", "hallucinated_tool"],
    "Argument error": ["wrong_value", "wrong_type", "missing_arg", "extra_arg", "missing_optional"],
    "Format error": ["malformed_call", "text_call"],
    "No call": ["no_call"],
    "Other": ["multi_call", "timeout", "error"],
}
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
ARG_CATS = {"correct", "wrong_value", "wrong_type", "missing_arg", "extra_arg", "missing_optional"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------- loading
def lenient_correct(q, rec) -> bool:
    """Re-check a gold-tool call after coercing string-encoded scalars (see scoring.py)."""
    if rec["category"] not in ARG_CATS:
        return False
    if rec["category"] == "correct":
        return True
    call = rec["response"]["tool_calls"][0]
    args = coerce_args(q.func, copy.deepcopy(call["arguments"]))
    try:
        return simple_function_checker(checker_description(q.func), {call["name"]: args}, q.answer)["valid"]
    except Exception:
        return False


_VIS: dict = {}
_RET = None


def visible_tools(r: dict, queries) -> list[str]:
    """Reconstruct the tool names the model saw (catalogs and retrieval are deterministic)."""
    global _RET
    key = (r["N"], r["seed"])
    if key not in _VIS:
        cats, q2c = build_catalogs(queries, r["N"], r["seed"])
        _VIS[key] = ({c.catalog_id: c.names for c in cats}, q2c)
    names, q2c = _VIS[key]
    cat = names[q2c[r["query_id"]]]
    if r.get("k"):
        if _RET is None:
            _RET = ToolRetriever("qwen3-embedding:0.6b", function_pool(queries), {q.id: q.question for q in queries})
        return _RET.top_k(r["query_id"], cat, r["k"])
    return cat


def load() -> pd.DataFrame:
    queries = load_queries()
    qd = {q.id: q for q in queries}
    rows = []
    for exp in ("rq1_scaling", "rq3_retrieval", "rq3_k_ablation"):
        for p in sorted((ROOT / "results/raw" / exp).glob("*.jsonl")):
            with open(p, encoding="utf8") as fh:
                for line in fh:
                    r = json.loads(line)
                    resp = r["response"]
                    # re-derive the category from the saved raw output with the current taxonomy
                    r["correct"], r["category"], _ = score(qd[r["query_id"]], resp, visible_tools(r, queries))
                    rows.append({
                        "exp": exp, "model": r["model"], "N": r["N"], "seed": r["seed"],
                        "k": r["k"] if r["k"] is not None else 0, "query_id": r["query_id"],
                        "correct": r["correct"],
                        "category": r["category"],
                        "lenient": lenient_correct(qd[r["query_id"]], r),
                        "gold_pos": r["gold_pos"], "gold_in_visible": r["gold_in_visible"],
                        "prompt_tokens": resp.get("prompt_eval_count") or np.nan,
                        "gen_tokens": resp.get("eval_count") or np.nan,
                        "latency_s": resp.get("latency_s", np.nan),
                        "overflow": r["ctx_overflow_risk"],
                        "unquoted_name": bool(re.search(r'"name":\s*[A-Za-z_]', resp.get("raw_output") or "")),
                    })
    df = pd.DataFrame(rows)
    # One record per call: keep the FIRST successful record (a crash on 2026-10-18 left two runner
    # processes writing to the same files -> duplicates); fall back to an error record only if no
    # successful one exists (such calls are retried on resume) ...
    df["is_error"] = df.category.eq("error")
    df = (df.reset_index().sort_values(["is_error", "index"])
            .drop_duplicates(["exp", "model", "N", "seed", "k", "query_id"], keep="first")
            .sort_values("index").drop(columns=["index", "is_error"]))
    # ... and only the configured small-scale design: fixed 100-query subsample, seeds 0-1
    keep = eval_subset(load_queries(), 100, 0)
    df = df[df.query_id.isin(keep) & (df.seed <= 1)].reset_index(drop=True)
    df.to_csv(PROC / "all_records.csv", index=False)
    return df


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test on paired booleans (a=condition 1, b=condition 2)."""
    b01 = int(((~a) & b).sum())
    b10 = int((a & ~b).sum())
    n = b01 + b10
    p = stats.binomtest(min(b01, b10), n, 0.5).pvalue if n else 1.0
    return {"only_first": b10, "only_second": b01, "p": float(p)}


def acc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds, plus pooled Wilson CI, for each model × N."""
    out = []
    for (m, n), g in df.groupby(["model", "N"]):
        per_seed = g.groupby("seed").agg(acc=("correct", "mean"), len_=("lenient", "mean"), cnt=("correct", "size"))
        lo, hi = wilson(int(g.correct.sum()), len(g))
        out.append({"model": m, "N": n, "n_seeds": len(per_seed), "n_calls": len(g),
                    "acc": per_seed.acc.mean(), "acc_std": per_seed.acc.std(ddof=1) if len(per_seed) > 1 else 0.0,
                    "acc_lo": lo, "acc_hi": hi,
                    "len": per_seed.len_.mean(), "len_std": per_seed.len_.std(ddof=1) if len(per_seed) > 1 else 0.0,
                    "prompt_tokens": g.prompt_tokens.median(), "latency_s": g.latency_s.median()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures
def end_labels(ax, items, x):
    """Direct labels at line ends, nudged apart so they never overlap."""
    items = sorted(items, key=lambda t: t[1])
    ys, min_gap = [], 0.065
    for _, y, _ in items:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    for (label, _, color), y in zip(items, ys):
        ax.annotate(label, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6.5, color=INK, annotation_clip=False)
        ax.plot([x], [y], marker="s", ms=3, color=color, clip_on=False, transform=ax.transData, alpha=0)


def fig_scaling(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True, gridspec_kw={"wspace": 0.42})
    for ax, col, title in [(axes[0], "acc", "Strict (BFCL) accuracy"),
                           (axes[1], "len", "Lenient accuracy (string→type coercion)")]:
        labels = []
        for m, (name, color, mk) in MODELS.items():
            g = t[t.model == m].sort_values("N")
            if g.empty:
                continue
            ax.errorbar(g.N, g[col], yerr=g[col + "_std"], color=color, marker=mk, ms=4.5,
                        mec="white", mew=0.8, capsize=2, lw=1.6, label=name)
            labels.append((name, g[col].iloc[-1], color))
        end_labels(ax, labels, t.N.max())
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.set_xlim(0.8, 70)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Tools in the catalog (N, log scale)")
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylabel("Accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "fig_scaling.pdf")
    fig.savefig(FIG / "fig_scaling.png", dpi=200)
    plt.close(fig)


def fig_failures(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.2), sharey=True)
    for ax, (m, (name, _, _)) in zip(axes, MODELS.items()):
        g = df[df.model == m]
        if g.empty:
            continue
        sizes = sorted(g.N.unique())
        bottom = np.zeros(len(sizes))
        for (grp, cats), color in zip(GROUPS.items(), GROUP_COLORS):
            vals = np.array([g[g.N == n].category.isin(cats).mean() for n in sizes])
            ax.bar(range(len(sizes)), vals, bottom=bottom, color=color, width=0.72,
                   edgecolor="white", linewidth=1.0, label=grp)
            bottom += vals
        ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("N")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of queries failed")
    axes[0].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "fig_failures.pdf")
    fig.savefig(FIG / "fig_failures.png", dpi=200)
    plt.close(fig)


def fig_position(df: pd.DataFrame) -> dict:
    g = df[(df.exp == "rq1_scaling") & (df.N == 64)].copy()
    res = {}
    if g.empty:
        return res
    g["bin"] = pd.cut(g.gold_pos, bins=[-1, 15, 31, 47, 63], labels=["1–16", "17–32", "33–48", "49–64"])
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    labels = []
    for m, (name, color, mk) in MODELS.items():
        h = g[g.model == m]
        if h.empty:
            continue
        a = h.groupby("bin", observed=False).correct.mean()
        ax.plot(range(4), a.values, color=color, marker=mk, ms=4.5, mec="white", mew=0.8, label=name)
        labels.append((name, a.values[-1], color))
        ct = pd.crosstab(h["bin"], h.correct)
        chi2, p, _, _ = stats.chi2_contingency(ct) if ct.shape == (4, 2) else (np.nan, np.nan, 0, 0)
        res[m] = {"acc_by_bin": a.round(4).tolist(), "chi2": float(chi2), "p": float(p)}
    end_labels(ax, labels, 3)
    ax.set_xticks(range(4), ["1–16", "17–32", "33–48", "49–64"])
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position of the gold tool in the catalog (N = 64)")
    ax.set_ylabel("Accuracy")
    fig.savefig(FIG / "fig_position.pdf")
    fig.savefig(FIG / "fig_position.png", dpi=200)
    plt.close(fig)
    return res


def fig_retrieval(rows: list[dict]) -> None:
    if not rows:
        return
    # lenient accuracy: identical to strict for the Qwen models; removes Llama's string-number artefact
    conds = [("N = 1 (gold only)", "len_n1", "#c3c2b7"), ("N = 64, all tools", "len_full", "#52514e"),
             ("N = 64, top-3 retrieved", "len_ret", "#2a78d6")]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    w = 0.26
    for i, (lab, key, color) in enumerate(conds):
        xs = [j + (i - 1) * w for j in range(len(rows))]
        ax.bar(xs, [r[key] for r in rows], width=w - 0.02, color=color, label=lab, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(rows)), [MODELS[r["model"]][0].replace("-", chr(10), 1) for r in rows], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy (lenient)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    fig.savefig(FIG / "fig_retrieval.pdf")
    fig.savefig(FIG / "fig_retrieval.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- LaTeX tables
def pct(x, s=None):
    if pd.isna(x):
        return "--"
    return f"{100 * x:.1f}" + (f"\\,$\\pm$\\,{100 * s:.1f}" if s is not None and not pd.isna(s) and s > 0 else "")


def write_tables(t: pd.DataFrame, df: pd.DataFrame, ret_rows: list[dict], kab: pd.DataFrame) -> None:
    sizes = sorted(t.N.unique())
    lines = [r"\begin{tabular}{l" + "c" * len(sizes) + "c}", r"\toprule",
             "Model & " + " & ".join(f"$N{{=}}{n}$" for n in sizes) + r" & $\Delta_{1\to64}$ \\", r"\midrule",
             r"\multicolumn{" + str(len(sizes) + 2) + r"}{l}{\textit{Strict (BFCL) accuracy, \% (mean $\pm$ std over 2 catalog seeds)}} \\"]
    for col in ("acc", "len"):
        if col == "len":
            lines += [r"\midrule", r"\multicolumn{" + str(len(sizes) + 2) + r"}{l}{\textit{Lenient accuracy, \% (string-encoded scalars coerced to schema type)}} \\"]
        for m, (name, _, _) in MODELS.items():
            g = t[t.model == m].set_index("N")
            if g.empty:
                continue
            cells = [pct(g.loc[n, col], g.loc[n, col + "_std"]) if n in g.index else "--" for n in sizes]
            d = (g.loc[64, col] - g.loc[1, col]) if (1 in g.index and 64 in g.index) else np.nan
            lines.append(f"{name} & " + " & ".join(cells) + (f" & {100 * d:+.1f}" if not pd.isna(d) else " & --") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_accuracy.tex").write_text("\n".join(lines), encoding="utf8")

    # full failure breakdown at N=1 and N=64 (rq1)
    cats = ["correct", "wrong_tool", "hallucinated_tool", "wrong_value", "wrong_type", "missing_arg",
            "extra_arg", "missing_optional", "malformed_call", "text_call", "no_call", "multi_call",
            "timeout", "error"]
    pretty = {c: c.replace("_", " ") for c in cats}
    r1 = df[df.exp == "rq1_scaling"]
    head = " & ".join(f"\\rotatebox{{70}}{{{MODELS[m][0]}, $N{{=}}{n}$}}" for m in MODELS for n in (1, 64))
    lines = [r"\begin{tabular}{l" + "r" * 8 + "}", r"\toprule", "Category & " + head + r" \\", r"\midrule"]
    for c in cats:
        cells = []
        for m in MODELS:
            for n in (1, 64):
                g = r1[(r1.model == m) & (r1.N == n)]
                cells.append(f"{100 * (g.category == c).mean():.1f}" if len(g) else "--")
        lines.append(pretty[c] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_failures.tex").write_text("\n".join(lines), encoding="utf8")

    if ret_rows:
        lines = [r"\begin{tabular}{lccc|ccc|c}", r"\toprule",
                 r" & \multicolumn{3}{c|}{Strict accuracy} & \multicolumn{3}{c|}{Lenient accuracy} & \\",
                 r"Model & $N{=}1$ & 64 all & 64$\to$top-3 & $N{=}1$ & 64 all & 64$\to$top-3 & McNemar $p$ \\",
                 r"\midrule"]
        for r in ret_rows:
            lines.append(f"{MODELS[r['model']][0]} & {pct(r['acc_n1'])} & {pct(r['acc_full'])} & {pct(r['acc_ret'])} & "
                         f"{pct(r['len_n1'])} & {pct(r['len_full'])} & {pct(r['len_ret'])} & "
                         f"{r['p_fmt']} / {r['pl_fmt']}" + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "tab_retrieval.tex").write_text("\n".join(lines), encoding="utf8")

    if not kab.empty:
        ks = [1, 3, 5, 64]
        lines = [r"\begin{tabular}{l" + "c" * len(ks) + "}", r"\toprule",
                 "Model & " + " & ".join("all 64" if k == 64 else f"top-{k}" for k in ks) + r" \\", r"\midrule"]
        for m, (name, _, _) in MODELS.items():
            g = kab[kab.model == m].set_index("k")
            if g.empty:
                continue
            col = "len" if m == "llama3.2:latest" else "acc"  # Llama: lenient (strict is masked by string numbers)
            lines.append(name + (r"$^\dagger$" if col == "len" else "") + " & "
                         + " & ".join(pct(g.loc[k, col]) if k in g.index else "--" for k in ks) + r" \\")
        lines += [r"\midrule", "Gold tool retrieved & " + " & ".join(
            pct(kab[kab.k == k].recall.mean()) for k in ks) + r" \\", r"\bottomrule", r"\end{tabular}"]
        (TAB / "tab_k_ablation.tex").write_text("\n".join(lines), encoding="utf8")

    # cost table
    lines = [r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\toprule",
             "Model & " + " & ".join(f"$N{{=}}{n}$" for n in sizes) + r" \\", r"\midrule"]
    for m, (name, _, _) in MODELS.items():
        g = t[t.model == m].set_index("N")
        if g.empty:
            continue
        lines.append(name + " & " + " & ".join(
            f"{g.loc[n, 'prompt_tokens']:,.0f} / {g.loc[n, 'latency_s']:.1f}" if n in g.index else "--" for n in sizes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_cost.tex").write_text("\n".join(lines), encoding="utf8")


# ---------------------------------------------------------------- main
def main() -> None:
    df = load()
    print("records per experiment/model:\n", df.groupby(["exp", "model"]).size().to_string())
    print("context-overflow risks:", int(df.overflow.sum()), "| request errors:", int((df.category == "error").sum()))
    summary: dict = {"n_records": len(df)}

    r1 = df[df.exp == "rq1_scaling"]
    t = acc_table(r1)
    summary["accuracy"] = t.round(4).to_dict(orient="records")
    print(t[["model", "N", "n_calls", "acc", "acc_std", "len", "prompt_tokens", "latency_s"]].round(3).to_string())

    # N=1 vs N=64 paired McNemar (each seed's N=64 record paired with the single N=1 record)
    summary["mcnemar_n1_vs_n64"] = {}
    for m in MODELS:
        a = r1[(r1.model == m) & (r1.N == 1)].set_index("query_id").correct
        b = r1[(r1.model == m) & (r1.N == 64)]
        if a.empty or b.empty:
            continue
        pa = b.query_id.map(a)
        ok = pa.notna()
        summary["mcnemar_n1_vs_n64"][m] = mcnemar(pa[ok].astype(bool), b.correct[ok])

    # N=4 vs N=64, paired by (seed, query): same queries, same catalog seed
    summary["mcnemar_n4_vs_n64"] = {}
    for m in MODELS:
        a = r1[(r1.model == m) & (r1.N == 4)].set_index(["seed", "query_id"])
        b = r1[(r1.model == m) & (r1.N == 64)].set_index(["seed", "query_id"])
        common = a.index.intersection(b.index)
        if len(common):
            summary["mcnemar_n4_vs_n64"][m] = {**mcnemar(a.correct.loc[common], b.correct.loc[common]),
                                               "lenient": mcnemar(a.lenient.loc[common], b.lenient.loc[common])}

    # failure group shares
    summary["failure_groups"] = {
        m: {int(n): {grp: round(float(r1[(r1.model == m) & (r1.N == n)].category.isin(c).mean()), 4)
                     for grp, c in GROUPS.items()} for n in sorted(r1[r1.model == m].N.unique())}
        for m in MODELS if (r1.model == m).any()}

    # what format errors are made of (RQ2): share of malformed calls with an unquoted function name
    mf = r1[r1.category == "malformed_call"]
    summary["malformed_unquoted"] = {m: {"n": int((mf.model == m).sum()),
                                         "unquoted": int(mf[mf.model == m].unquoted_name.sum())} for m in MODELS}

    # serialisation (string-number) errors explained by lenient scoring
    summary["lenient_gain"] = {m: round(float(r1[r1.model == m].lenient.mean() - r1[r1.model == m].correct.mean()), 4)
                               for m in MODELS if (r1.model == m).any()}

    summary["position"] = fig_position(df)

    # RQ3: retrieval. Main setting k=3 (configs/rq3_retrieval.yaml), paired per query with the full
    # N=64 catalog of the SAME seed (0); k in {1, 5} is the ablation.
    r3 = df[(df.exp == "rq3_retrieval") & (df.seed == 0)]
    ret_rows = []
    for m in MODELS:
        full = r1[(r1.model == m) & (r1.N == 64) & (r1.seed == 0)].set_index("query_id")
        ret = r3[(r3.model == m) & (r3.k == 3)].set_index("query_id")
        n1 = r1[(r1.model == m) & (r1.N == 1)]
        if ret.empty or full.empty:
            continue
        common = full.index.intersection(ret.index)
        mc = mcnemar(full.correct.loc[common], ret.correct.loc[common])
        mcl = mcnemar(full.lenient.loc[common], ret.lenient.loc[common])
        ret_rows.append({"model": m, "acc_n1": n1.correct.mean(), "acc_full": full.correct.loc[common].mean(),
                         "acc_ret": ret.correct.loc[common].mean(), "len_n1": n1.lenient.mean(),
                         "len_full": full.lenient.loc[common].mean(), "len_ret": ret.lenient.loc[common].mean(),
                         "recall": ret.gold_in_visible.mean(),
                         "acc_given_retrieved": ret[ret.gold_in_visible].correct.mean(),
                         "mcnemar": mc, "mcnemar_lenient": mcl,
                         "p_fmt": ("$<$0.001" if mc["p"] < 1e-3 else f"{mc['p']:.3f}"),
                         "pl_fmt": ("$<$0.001" if mcl["p"] < 1e-3 else f"{mcl['p']:.3f}"),
                         "n_pairs": len(common)})
    summary["retrieval"] = [{k: v for k, v in r.items() if not k.endswith("_fmt")} for r in ret_rows]

    kab = r3.groupby(["model", "k"]).agg(acc=("correct", "mean"), len=("lenient", "mean"),
                                         recall=("gold_in_visible", "mean")).reset_index()
    full0 = (r1[(r1.N == 64) & (r1.seed == 0)].groupby("model")
             .agg(acc=("correct", "mean"), len=("lenient", "mean")).reset_index().assign(k=64, recall=1.0))
    kab = pd.concat([kab, full0], ignore_index=True)
    summary["k_ablation"] = kab.round(4).to_dict(orient="records")

    rr = PROC / "retrieval_recall.csv"
    if rr.exists():
        rec = pd.read_csv(rr)
        summary["retrieval_recall"] = (rec.groupby(["embedder", "N", "k"]).hit.mean().round(4)
                                       .reset_index().to_dict(orient="records"))

    if not t.empty:
        fig_scaling(t)
        fig_failures(r1)
    fig_retrieval(ret_rows)
    write_tables(t, df, ret_rows, kab)
    (PROC / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf8")
    print("wrote", PROC / "summary.json")


if __name__ == "__main__":
    main()
