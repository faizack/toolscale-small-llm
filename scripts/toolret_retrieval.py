"""RQ3b: embedding retrieval on ToolRet's native 44k-tool corpus (no LLM calls).

ToolRet (mangopy/ToolRet-Queries @ b8c76ad, mangopy/ToolRet-Tools @ e06c38c) aggregates 35
tool-use datasets into one retrieval benchmark. We sample 200 queries per category
(web / code / customized), retrieve from the FULL corpus, and report Recall@k and nDCG@10.

Variants: bge-m3 (query only) | qwen3-embedding (generic instruction) |
          qwen3-embedding (ToolRet's per-query task instruction)
Output: results/processed/toolret_retrieval.csv (+ per-query hits in toolret_hits.csv)
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.retrieval import _cached_embed  # noqa: E402

DATA = ROOT / "data/raw/toolret"
PER_CATEGORY, SEED, MAX_DOC_CHARS, KS = 200, 0, 2000, [1, 5, 10, 50]
GENERIC = "Instruct: Given a user request, retrieve the API function that can fulfill it\nQuery: {q}"
TASK = "Instruct: {i}\nQuery: {q}"


def load():
    tools = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "tools/*.parquet")))])
    tools = tools.drop_duplicates("id").reset_index(drop=True)
    qs = pd.concat([pd.read_parquet(f).assign(subset=os.path.basename(f)[:-8])
                    for f in sorted(glob.glob(str(DATA / "queries/*.parquet")))]).reset_index(drop=True)
    rng = random.Random(f"toolret-{SEED}")
    picked = []
    for cat in sorted(qs.category.unique()):
        idx = sorted(qs.index[qs.category == cat])
        picked += rng.sample(idx, min(PER_CATEGORY, len(idx)))
    qs = qs.loc[sorted(picked)].reset_index(drop=True)
    qs["gold"] = qs.labels.map(lambda s: [x["id"] for x in json.loads(s) if x.get("relevance", 1) > 0])
    return tools, qs


def ndcg_at(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, t in enumerate(ranked[:k]) if t in gold)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    tools, qs = load()
    print(f"corpus={len(tools)} tools, queries={len(qs)} ({qs.category.value_counts().to_dict()})")
    ids = tools.id.tolist()
    docs = [d[:MAX_DOC_CHARS] for d in tools.documentation]
    variants = [("bge-m3", "bge-m3:latest", "{q}"),
                ("qwen3-emb (generic instr.)", "qwen3-embedding:0.6b", GENERIC),
                ("qwen3-emb (task instr.)", "qwen3-embedding:0.6b", TASK)]
    rows, hits = [], []
    for label, model, tmpl in variants:
        T = _cached_embed(model, docs, "toolret_tools")
        Q = _cached_embed(model, [tmpl.format(q=q, i=i) for q, i in zip(qs["query"], qs.instruction)],
                          "toolret_queries")
        sims = Q @ T.T
        top = np.argsort(-sims, axis=1)[:, :max(KS)]
        for qi, r in qs.iterrows():
            ranked = [ids[j] for j in top[qi]]
            gold = set(r.gold)
            rec = {"variant": label, "query_id": r.id, "category": r.category, "subset": r.subset,
                   "n_gold": len(gold), "ndcg@10": ndcg_at(ranked, gold)}
            for k in KS:
                rec[f"recall@{k}"] = len(gold & set(ranked[:k])) / len(gold) if gold else 0.0
            hits.append(rec)
    df = pd.DataFrame(hits)
    df.to_csv(ROOT / "results/processed/toolret_hits.csv", index=False)
    cols = ["ndcg@10"] + [f"recall@{k}" for k in KS]
    summ = df.groupby(["variant", "category"])[cols].mean().round(4)
    summ = pd.concat([summ, df.groupby("variant")[cols].mean().round(4).assign(category="all")
                      .set_index("category", append=True)])
    summ.to_csv(ROOT / "results/processed/toolret_retrieval.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
