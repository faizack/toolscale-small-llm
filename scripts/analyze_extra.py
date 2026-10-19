"""Analysis of the native-dataset experiments (RQ3b ToolRet, RQ4 native BFCL, RQ5 multi-turn).

Usage: .venv/Scripts/python.exe scripts/analyze_extra.py   (after scripts/analyze.py)
Outputs: results/processed/summary_extra.json, paper/tables/tab_native.tex, tab_multiturn.tex,
         tab_toolret.tex, paper/figures/fig_native.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyze import FIG, INK, MODELS, PROC, TAB, end_labels, mcnemar, pct, plt, wilson  # noqa: E402,F401

BUCKETS_CALL = ["2", "3", "4", "5", "6+"]
BUCKETS_IRR = ["1", "2-3", "4-5", "6+"]


def bucket(n: int, edges: list[int]) -> str:
    lab = None
    for i, lo in enumerate(edges):
        if n >= lo:
            hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
            lab = f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi else f"{lo}+")
    return lab


def load_jsonl_dir(name: str) -> list[dict]:
    rows = []
    for p in sorted((ROOT / "results/raw" / name).glob("*.jsonl")):
        with open(p, encoding="utf8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def native(summary: dict) -> None:
    rows = load_jsonl_dir("rq4_native")
    if not rows:
        print("rq4_native: no results yet")
        return
    # re-derive call categories from the raw output with the current taxonomy (as in analyze.py)
    from toolscale.data import RAW, Query, safe_name
    from toolscale.scoring import score
    items, gts = {}, {}
    for split in ("live_multiple", "live_irrelevance"):
        for line in open(RAW / f"BFCL_v4_{split}.json", encoding="utf8"):
            it = json.loads(line)
            items[it["id"]] = it
    for line in open(RAW / "possible_answer_BFCL_v4_live_multiple.json", encoding="utf8"):
        a = json.loads(line)
        gts[a["id"]] = a["ground_truth"]
    for r in rows:
        if r["kind"] == "call" and "error" not in r["response"]:
            funcs = items[r["query_id"]]["function"]
            (gname, params), = gts[r["query_id"]][0].items()
            gfunc = next(f for f in funcs if f["name"] == gname)
            q = Query(r["query_id"], "", gfunc, {safe_name(gname): params})
            names = [safe_name(f["name"]) for f in funcs]
            r["correct"], r["category"], _ = score(q, r["response"], names)
            r["lenient"] = score(q, r["response"], names, lenient=True)[0]
        else:
            r["lenient"] = r["correct"]
    df = pd.DataFrame([{k: r.get(k, r.get("correct")) for k in ("key", "model", "split", "kind", "native_n", "correct",
                                                                 "lenient", "category")} for r in rows])
    # keep the first successful record per call (errors only if nothing else exists)
    df["is_error"] = df.category.eq("error")
    df = (df.reset_index().sort_values(["is_error", "index"]).drop_duplicates(["model", "key"], keep="first")
            .sort_values("index").drop(columns=["index", "is_error"]))
    df["bucket"] = [bucket(n, [2, 3, 4, 5, 6]) if k == "call" else bucket(n, [1, 2, 4, 6])
                    for n, k in zip(df.native_n, df.kind)]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for ax, kind, order, title, ylab in [
            (axes[0], "call", BUCKETS_CALL, "live_multiple: correct call", "Accuracy"),
            (axes[1], "irrelevance", BUCKETS_IRR, "live_irrelevance: correctly abstains", "Abstention rate")]:
        labels = []
        for m, (name, color, mk) in MODELS.items():
            g = df[(df.model == m) & (df.kind == kind)]
            if g.empty:
                continue
            a = g.groupby("bucket").correct.mean().reindex(order)
            ax.plot(range(len(order)), a.values, color=color, marker=mk, ms=4.5, mec="white", mew=0.8, label=name)
            labels.append((name, a.values[-1], color))
            lo, hi = wilson(int(g.correct.sum()), len(g))
            out.setdefault(m, {})[kind] = {"overall": float(g.correct.mean()), "lo": lo, "hi": hi, "n": len(g),
                                           "lenient": float(g.lenient.mean()),
                                           "by_bucket": {b: (None if pd.isna(v) else float(v)) for b, v in a.items()},
                                           "categories": g.category.value_counts(normalize=True).round(4).to_dict()}
        end_labels(ax, labels, len(order) - 1)
        ax.set_xticks(range(len(order)), order)
        ax.set_xlim(-0.2, len(order) - 0.1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Native number of tools in the query")
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel(ylab) if kind == "call" else None
    axes[1].set_ylabel("Abstention rate")
    axes[0].legend(frameon=False, loc="lower left", ncol=2)
    fig.savefig(FIG / "fig_native.pdf")
    fig.savefig(FIG / "fig_native.png", dpi=200)
    plt.close(fig)
    summary["native"] = out
    lines = [r"\begin{tabular}{l" + "c" * (len(BUCKETS_CALL) + 2) + "|" + "c" * (len(BUCKETS_IRR) + 1) + "}",
             r"\toprule",
             r" & \multicolumn{" + str(len(BUCKETS_CALL) + 2) + r"}{c|}{\texttt{live\_multiple}: accuracy by native $N$} & "
             r"\multicolumn{" + str(len(BUCKETS_IRR) + 1) + r"}{c}{\texttt{live\_irrelevance}: abstention} \\",
             "Model & " + " & ".join(BUCKETS_CALL) + r" & all & all$^\dagger$ & " + " & ".join(BUCKETS_IRR) + r" & all \\",
             r"\midrule"]
    for m, (name, _, _) in MODELS.items():
        if m not in out:
            continue
        c, i = out[m].get("call", {}), out[m].get("irrelevance", {})
        cells = [pct(c.get("by_bucket", {}).get(b)) for b in BUCKETS_CALL] + [pct(c.get("overall")), pct(c.get("lenient"))]
        cells += [pct(i.get("by_bucket", {}).get(b)) for b in BUCKETS_IRR] + [pct(i.get("overall"))]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_native.tex").write_text("\n".join(lines), encoding="utf8")


def dropped_calls(rows: list[dict]) -> dict:
    """Agent steps with no parsed call whose raw tokens contain an attempted call, i.e. calls the
    server silently dropped; split into unknown tool names vs. invalid JSON (rule of scoring.py)."""
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING as DOCS
    from toolscale.data import RAW, safe_name
    from toolscale.scoring import parse_raw_call
    ents = {}
    for line in open(RAW / "BFCL_v4_multi_turn_base.json", encoding="utf8"):
        e = json.loads(line)
        ents[e["id"]] = e
    out: dict = {}
    for r in rows:
        names = set()
        for cls in ents[r["id"]]["involved_classes"]:
            names |= {safe_name(json.loads(x)["name"])
                      for x in open(RAW / "multi_turn_func_doc" / DOCS[cls], encoding="utf8") if x.strip()}
        c = out.setdefault(r["model"], {"steps": 0, "dropped_unknown_tool": 0, "dropped_invalid_json": 0})
        for t in r["trace"]:
            if "error" in t:
                continue
            c["steps"] += 1
            raw = t.get("raw", "")
            if t["calls"] or not ("<tool_call>" in raw or '"name"' in raw):
                continue
            kind, nm = parse_raw_call(raw)
            if kind == "json" and nm and nm not in names:
                c["dropped_unknown_tool"] += 1
            elif kind == "invalid":
                c["dropped_invalid_json"] += 1
    return out


def multiturn(summary: dict) -> None:
    rows = load_jsonl_dir("rq5_multiturn")
    if not rows:
        print("rq5_multiturn: no results yet")
        return
    df = pd.DataFrame([{k: r[k] for k in ("id", "model", "valid", "error_type", "n_turns", "n_tools", "n_steps",
                                          "n_calls", "n_gt_calls", "n_hallucinated", "n_exec_errors",
                                          "forced_quit", "request_errors", "seconds")} for r in rows])
    df = (df.assign(_err=df.request_errors.gt(0)).reset_index().sort_values(["_err", "index"])
            .drop_duplicates(["model", "id"], keep="first"))
    kept = [rows[i] for i in df["index"]]
    df = df.drop(columns=["index", "_err"])
    drops = dropped_calls(kept)
    out = {}
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & Success (95\% CI) & Calls / GT calls & Exec.\ errors / ep. & Dropped calls & Forced quit & Top failure \\",
             r"\midrule"]
    for m, (name, _, _) in MODELS.items():
        g = df[df.model == m]
        if g.empty:
            continue
        lo, hi = wilson(int(g.valid.sum()), len(g))
        top = g[~g.valid].error_type.str.replace("multi_turn:", "", regex=False).value_counts()
        out[m] = {"success": float(g.valid.mean()), "lo": lo, "hi": hi, "n": len(g),
                  "calls_ratio": float(g.n_calls.sum() / max(g.n_gt_calls.sum(), 1)),
                  "exec_err_per_ep": float(g.n_exec_errors.mean()), "halluc_per_ep": float(g.n_hallucinated.mean()),
                  "forced_quit": float(g.forced_quit.mean()), "error_types": top.to_dict(),
                  "median_seconds": float(g.seconds.median()),
                  "server_errors": int(g.request_errors.gt(0).sum()), **drops.get(m, {})}
        d = drops.get(m, {"steps": 1, "dropped_unknown_tool": 0, "dropped_invalid_json": 0})
        out[m]["dropped_share"] = (d["dropped_unknown_tool"] + d["dropped_invalid_json"]) / max(d["steps"], 1)
        lines.append(f"{name} & {pct(g.valid.mean())} ({pct(lo)}--{pct(hi)}) & "
                     f"{out[m]['calls_ratio']:.2f} & {out[m]['exec_err_per_ep']:.1f} & "
                     f"{pct(out[m]['dropped_share'])}" + r"\% of steps & "
                     f"{pct(out[m]['forced_quit'])} & \\small{{{(top.index[0] if len(top) else '--').replace('_', ' ')}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_multiturn.tex").write_text("\n".join(lines), encoding="utf8")
    summary["multiturn"] = out


def toolret(summary: dict) -> None:
    p = PROC / "toolret_retrieval.csv"
    if not p.exists():
        print("toolret: no results yet")
        return
    s = pd.read_csv(p)
    summary["toolret"] = s.round(4).to_dict(orient="records")
    order = ["web", "code", "customized", "all"]
    lines = [r"\begin{tabular}{lcccc|cc}", r"\toprule",
             r"Retriever & \multicolumn{4}{c|}{nDCG@10} & \multicolumn{2}{c}{Recall (all)} \\",
             r" & web & code & custom. & all & @10 & @50 \\", r"\midrule"]
    for v in s.variant.unique():
        g = s[s.variant == v].set_index("category")
        cells = [f"{100 * g.loc[c, 'ndcg@10']:.1f}" if c in g.index else "--" for c in order]
        cells += [f"{100 * g.loc['all', 'recall@10']:.1f}", f"{100 * g.loc['all', 'recall@50']:.1f}"]
        lines.append(v + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_toolret.tex").write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    summary: dict = {}
    native(summary)
    multiturn(summary)
    toolret(summary)
    (PROC / "summary_extra.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf8")
    print(json.dumps(summary, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
