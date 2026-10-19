"""Write paper/tables/macros.tex: one LaTeX macro per number quoted in the paper text.

Example: \\AccQwenThreeNSixtyFour -> 58.0  (strict accuracy, %, mean over seeds)
Run after scripts/analyze.py and scripts/analyze_extra.py. Prose must use these macros, never
typed-in numbers. Missing results produce "??" so a gap is visible in the PDF.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/processed/summary.json").read_text(encoding="utf8"))
X_PATH = ROOT / "results/processed/summary_extra.json"
X = json.loads(X_PATH.read_text(encoding="utf8")) if X_PATH.exists() else {}

MODEL = {"qwen2.5:1.5b-instruct": "QwenOneFive", "qwen2.5:3b-instruct": "QwenThree",
         "llama3.2:latest": "Llama", "qwen3.5:2b": "QwenThreeFive"}
NUM = {1: "One", 3: "Three", 4: "Four", 5: "Five", 10: "Ten", 16: "Sixteen", 50: "Fifty", 64: "SixtyFour"}
GRP = {"Wrong / hallucinated tool": "Tool", "Argument error": "Arg", "Format error": "Format",
       "No call": "NoCall", "Other": "Other"}

out = []


def m(name: str, value) -> None:
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def p(x) -> str:
    return "??" if x is None else f"{100 * x:.1f}"


def pv(x: float) -> str:
    return "$<$0.001" if x < 1e-3 else f"{x:.3f}"


m("NumRecords", f"{S['n_records']:,}")
acc = {}
for r in S["accuracy"]:
    k, n = MODEL[r["model"]], NUM[r["N"]]
    acc[(r["model"], r["N"])] = r
    m(f"Acc{k}N{n}", p(r["acc"]))
    m(f"AccStd{k}N{n}", p(r["acc_std"]))
    m(f"Len{k}N{n}", p(r["len"]))
    m(f"Tok{k}N{n}", f"{r['prompt_tokens']:,.0f}")
    m(f"Lat{k}N{n}", f"{r['latency_s']:.1f}")
for mod, k in MODEL.items():
    if (mod, 1) in acc and (mod, 64) in acc:
        a1, a64 = acc[(mod, 1)], acc[(mod, 64)]
        m(f"Drop{k}", p(a1["acc"] - a64["acc"]))
        m(f"LenDrop{k}", p(a1["len"] - a64["len"]))
        best = max(acc[(mod, n)]["acc"] for n in (1, 4, 16, 64) if (mod, n) in acc)
        m(f"DropFromBest{k}", p(best - a64["acc"]))
for mod, d in S.get("mcnemar_n1_vs_n64", {}).items():
    m(f"McNemarP{MODEL[mod]}", pv(d["p"]))
for mod, d in S.get("malformed_unquoted", {}).items():
    m(f"MalformedN{MODEL[mod]}", d["n"])
    m(f"MalformedUnquoted{MODEL[mod]}", d["unquoted"])
for mod, d in S.get("lenient_gain", {}).items():
    m(f"LenGain{MODEL[mod]}", p(d))
for mod, byn in S.get("failure_groups", {}).items():
    for n, groups in byn.items():
        for g, v in groups.items():
            m(f"Fail{GRP[g]}{MODEL[mod]}N{NUM[int(n)]}", p(v))
for mod, d in S.get("position", {}).items():
    m(f"PosP{MODEL[mod]}", pv(d["p"]))
for r in S.get("retrieval", []):
    k = MODEL[r["model"]]
    for key, name in [("acc_ret", "Ret"), ("acc_full", "RetFull"), ("acc_n1", "RetOne"),
                      ("len_ret", "RetLen"), ("len_full", "RetFullLen"), ("len_n1", "RetOneLen"),
                      ("recall", "RetRecall"), ("acc_given_retrieved", "RetGivenHit")]:
        m(f"{name}{k}", p(r[key]))
    m(f"RetGain{k}", p(r["acc_ret"] - r["acc_full"]))
    m(f"RetLenGain{k}", p(r["len_ret"] - r["len_full"]))
    m(f"RetP{k}", pv(r["mcnemar"]["p"]))
    m(f"RetLenP{k}", pv(r["mcnemar_lenient"]["p"]))
for r in S.get("k_ablation", []):
    k = MODEL[r["model"]]
    m(f"KAcc{k}K{NUM[int(r['k'])]}", p(r["acc"]))
    m(f"KLen{k}K{NUM[int(r['k'])]}", p(r["len"]))
for r in S.get("retrieval_recall", []):
    e = "Qwen" if r["embedder"].startswith("qwen") else "Bge"
    m(f"Recall{e}N{NUM[int(r['N'])]}K{NUM[int(r['k'])]}", p(r["hit"]))

# ---- extra experiments (RQ3b ToolRet, RQ4 native, RQ5 multi-turn)
for mod, d in X.get("native", {}).items():
    k = MODEL[mod]
    if "call" in d:
        m(f"NatAcc{k}", p(d["call"]["overall"]))
        bb = d["call"]["by_bucket"]
        m(f"NatAccLow{k}", p(bb.get("2")))
        m(f"NatAccHigh{k}", p(bb.get("6+")))
        cats = d["call"]["categories"]
        m(f"NatWrongTool{k}", p(cats.get("wrong_tool", 0.0)))
        m(f"NatLen{k}", p(d["call"].get("lenient")))
    if "irrelevance" in d:
        m(f"NatAbst{k}", p(d["irrelevance"]["overall"]))
        m(f"NatAbstInv{k}", p(1 - d["irrelevance"]["overall"]))
        bb = d["irrelevance"]["by_bucket"]
        m(f"NatAbstLow{k}", p(bb.get("1")))
        m(f"NatAbstHigh{k}", p(bb.get("6+")))
for mod, d in X.get("multiturn", {}).items():
    k = MODEL[mod]
    m(f"MtSucc{k}", p(d["success"]))
    m(f"MtLo{k}", p(d["lo"]))
    m(f"MtHi{k}", p(d["hi"]))
    m(f"MtN{k}", d["n"])
    m(f"MtCalls{k}", f"{d['calls_ratio']:.2f}")
    m(f"MtExecErr{k}", f"{d['exec_err_per_ep']:.1f}")
    m(f"MtDropped{k}", p(d.get("dropped_share")))
    m(f"MtServerErr{k}", d.get("server_errors", 0))
for r in X.get("toolret", []):
    if r["category"] != "all":
        continue
    v = {"bge-m3": "Bge", "qwen3-emb (generic instr.)": "QwenGen", "qwen3-emb (task instr.)": "QwenTask"}[r["variant"]]
    m(f"TrNdcg{v}", p(r["ndcg@10"]))
    for kk in (1, 5, 10, 50):
        m(f"TrRecall{v}K{NUM[kk]}", p(r[f"recall@{kk}"]))

for mod, d in S.get("mcnemar_n4_vs_n64", {}).items():  # N=4 vs N=64, paired by (seed, query)
    m(f"McNemarFourP{MODEL[mod]}", pv(d["p"]))
    m(f"McNemarFourLenP{MODEL[mod]}", pv(d["lenient"]["p"]))

(ROOT / "paper/tables/macros.tex").write_text("\n".join(out) + "\n", encoding="utf8")
print(f"wrote {len(out)} macros")
