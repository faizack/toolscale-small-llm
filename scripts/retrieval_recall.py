"""RQ3a: how often does embedding retrieval put the gold tool in the top-k?

Pure retrieval experiment (no LLM calls). Uses the same N=64 catalogs as RQ1.
Output: results/processed/retrieval_recall.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolscale.data import build_catalogs, function_pool, load_queries  # noqa: E402
from toolscale.retrieval import ToolRetriever  # noqa: E402

EMBEDDERS = ["qwen3-embedding:0.6b", "bge-m3:latest"]
SIZES, SEEDS, KS = [16, 64], [0, 1, 2], [1, 3, 5, 10]


def main() -> None:
    qs = load_queries()
    pool = function_pool(qs)
    rows = []
    for emb in EMBEDDERS:
        r = ToolRetriever(emb, pool, {q.id: q.question for q in qs})
        for n in SIZES:
            for seed in SEEDS:
                cats, _ = build_catalogs(qs, n, seed)
                for c in cats:
                    for qid in c.query_ids:
                        gold = next(q.gold_name for q in qs if q.id == qid)
                        ranked = r.top_k(qid, c.names, max(KS))
                        for k in KS:
                            rows.append({"embedder": emb, "N": n, "seed": seed, "k": k,
                                         "query_id": qid, "hit": gold in ranked[:k]})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "processed" / "retrieval_recall.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = (df.groupby(["embedder", "N", "k", "seed"]).hit.mean()
                 .groupby(["embedder", "N", "k"]).agg(["mean", "std"]).round(4))
    print(summary.to_string())


if __name__ == "__main__":
    main()
