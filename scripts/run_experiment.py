"""Run an experiment config; resumable; one JSON line per model call.

Usage:
    .venv/Scripts/python.exe scripts/run_experiment.py configs/rq1_scaling.yaml
    .venv/Scripts/python.exe scripts/run_experiment.py configs/rq1_scaling.yaml --limit 20   # smoke test

Output: results/raw/<config name>/<model>.jsonl  (+ logs/<config name>_<timestamp>.log)
Re-running the same command skips calls that are already recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolscale.data import build_catalogs, eval_subset, function_pool, load_queries, to_tool  # noqa: E402
from toolscale import runlock  # noqa: E402
from toolscale.llm import chat_with_tools  # noqa: E402
from toolscale.retrieval import ToolRetriever  # noqa: E402
from toolscale.scoring import score  # noqa: E402


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "src", "scripts", "configs"],
                                        cwd=ROOT, text=True).strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="max queries per condition (smoke test)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model tags")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf8"))
    name = cfg["name"] + ("_smoke" if args.limit else "")
    out_dir = ROOT / "results" / "raw" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    runlock.acquire(out_dir)
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / f"{name}_{stamp}.log", encoding="utf8"),
                                  logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    commit = git_commit()
    logging.info("config=%s hash=%s commit=%s", args.config, cfg_hash, commit)

    queries = load_queries(cfg.get("split", "simple_python"))
    keep = eval_subset(queries, cfg.get("subsample"), cfg.get("subsample_seed", 0))
    logging.info("evaluating %d of %d queries (subsample=%s)", len(keep), len(queries), cfg.get("subsample"))
    qd = {q.id: q for q in queries}
    pool = function_pool(queries)
    retriever = None
    if cfg.get("retrieval"):
        retriever = ToolRetriever(cfg["retrieval"]["embed_model"], pool, {q.id: q.question for q in queries})

    for m in cfg["models"]:
        if args.models and m["tag"] not in args.models:
            continue
        path = out_dir / f"{m['tag'].replace(':', '_')}.jsonl"
        done = set()
        if path.exists():
            with open(path, encoding="utf8") as fh:
                # failed requests (category "error") are retried; analysis keeps the last record per key
                done = {r["key"] for r in map(json.loads, filter(str.strip, fh)) if r["category"] != "error"}
        logging.info("model=%s already done=%d", m["tag"], len(done))
        with open(path, "a", encoding="utf8") as fh:
            for n in cfg["sizes"]:
                seeds = [0] if n == 1 else cfg["seeds"]
                for seed in seeds:
                    cats, _ = build_catalogs(queries, n, seed)
                    ks = cfg["retrieval"]["k"] if retriever else [None]
                    for k in ks:
                        num_ctx = cfg["num_ctx"][k if k else n] if k else cfg["num_ctx"][n]
                        t_cond, n_new, n_ok = time.time(), 0, 0
                        count = 0
                        for c in cats:
                            for qid in c.query_ids:
                                if qid not in keep:
                                    continue
                                count += 1
                                if args.limit and count > args.limit:
                                    break
                                key = f"N{n}|s{seed}|k{k}|{qid}"
                                if key in done:
                                    continue
                                q = qd[qid]
                                visible = retriever.top_k(qid, c.names, k) if k else c.names
                                # each query sees its *own* doc for its gold tool (27 names have variant docs)
                                tools = [to_tool(q.func if nm == q.gold_name else pool[nm]) for nm in visible]
                                resp = chat_with_tools(m["tag"], q.question, tools, num_ctx=num_ctx,
                                                       seed=seed, think=m.get("think"))
                                ok, cat, detail = score(q, resp, visible)
                                pe = resp.get("prompt_eval_count") or 0
                                rec = {
                                    "key": key, "model": m["tag"], "N": n, "seed": seed, "k": k,
                                    "query_id": qid, "catalog_id": c.catalog_id,
                                    "gold": q.gold_name, "gold_pos": c.names.index(q.gold_name),
                                    "gold_in_visible": q.gold_name in visible,
                                    "visible_size": len(visible), "num_ctx": num_ctx,
                                    "ctx_overflow_risk": pe >= num_ctx - 512,
                                    "correct": ok, "category": cat, "detail": detail,
                                    "response": resp, "config_hash": cfg_hash, "commit": commit,
                                    "ts": datetime.now().isoformat(timespec="seconds"),
                                }
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                fh.flush()
                                n_new += 1
                                n_ok += ok
                                if rec["ctx_overflow_risk"]:
                                    logging.warning("context near limit: %s pe=%d num_ctx=%d", key, pe, num_ctx)
                                if cat == "error":
                                    logging.warning("request error: %s %s", key, detail[:200])
                            if args.limit and count > args.limit:
                                break
                        if n_new:
                            logging.info("model=%s N=%d seed=%d k=%s new=%d acc_new=%.3f time=%.0fs",
                                         m["tag"], n, seed, k, n_new, n_ok / n_new, time.time() - t_cond)
    logging.info("finished %s", name)


if __name__ == "__main__":
    main()
