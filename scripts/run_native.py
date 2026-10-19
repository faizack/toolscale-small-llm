"""RQ4 (native catalogs): BFCL `live_multiple` and `live_irrelevance`, no catalog construction.

Each query is evaluated with exactly the tool list that BFCL ships with it (real user-contributed
queries, 1-37 tools). We sample queries stratified by the native number of tools.

  live_multiple    correct = BFCL checker on the single expected call (wrong tool -> fail)
  live_irrelevance correct = the model makes NO tool call (BFCL FC-mode rule)

Usage: .venv/Scripts/python.exe scripts/run_native.py configs/rq4_native.yaml
Output: results/raw/rq4_native/<model>.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolscale.data import RAW, Query, safe_name, to_tool  # noqa: E402
from toolscale import runlock  # noqa: E402
from toolscale.llm import chat_with_tools  # noqa: E402
from toolscale.scoring import looks_like_call, score  # noqa: E402

from run_experiment import git_commit  # noqa: E402


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bucket(n: int, edges: list[int]) -> str:
    """Label of the bucket that n falls in; edges are inclusive lower bounds."""
    lab = None
    for i, lo in enumerate(edges):
        if n >= lo:
            hi = edges[i + 1] - 1 if i + 1 < len(edges) else None
            lab = f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi else f"{lo}+")
    return lab


def stratified(items: list[dict], key, per_bucket: int, seed: int) -> list[dict]:
    rng = random.Random(f"native-{seed}")
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(key(it), []).append(it)
    out = []
    for b in sorted(by):
        pool = sorted(by[b], key=lambda x: x["id"])
        out += rng.sample(pool, min(per_bucket, len(pool)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf8"))
    out_dir = ROOT / "results/raw" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    runlock.acquire(out_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / f"{cfg['name']}_{stamp}.log", encoding="utf8"),
                                  logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    commit = git_commit()

    tasks = []
    for split, spec in cfg["splits"].items():
        items = read_jsonl(RAW / f"BFCL_v4_{split}.json")
        answers = {}
        if spec["kind"] == "call":
            answers = {a["id"]: a["ground_truth"] for a in read_jsonl(RAW / f"possible_answer_BFCL_v4_{split}.json")}
        items = [it for it in items if len(it["question"]) == 1 and len(it["function"]) >= 1]
        sample = stratified(items, lambda it: bucket(len(it["function"]), spec["buckets"]),
                            spec["per_bucket"], cfg["seed"])
        logging.info("%s: %d eligible, %d sampled", split, len(items), len(sample))
        for it in sample:
            tasks.append((split, spec["kind"], it, answers.get(it["id"])))

    for m in cfg["models"]:
        path = out_dir / f"{m['tag'].replace(':', '_')}.jsonl"
        done = set()
        if path.exists():
            done = {r["key"] for r in map(json.loads, filter(str.strip, open(path, encoding="utf8")))
                    if r["category"] != "error"}
        t0, n_new, n_ok = time.time(), 0, 0
        with open(path, "a", encoding="utf8") as fh:
            for split, kind, it, gt in tasks:
                key = f"{split}|{it['id']}"
                if key in done:
                    continue
                funcs = it["function"]
                names = [safe_name(f["name"]) for f in funcs]
                tools = [to_tool(f) for f in funcs]
                messages = it["question"][0]
                resp = chat_with_tools(m["tag"], messages, tools, num_ctx=cfg["num_ctx"],
                                       seed=cfg["seed"], think=m.get("think"))
                if kind == "irrelevance":
                    if "error" in resp:
                        ok, detail = False, resp["error"]
                        cat = "timeout" if "after watchdog restart" in detail else "error"
                    elif resp["tool_calls"]:
                        ok, cat, detail = False, "over_call", resp["tool_calls"][0]["name"]
                    else:
                        txt = resp.get("raw_output", "") + resp.get("content", "")
                        ok, cat, detail = True, ("abstain_malformed" if looks_like_call(txt, names) else "abstain"), ""
                    gold = None
                else:
                    (gname, params), = gt[0].items()
                    gfunc = next(f for f in funcs if f["name"] == gname)
                    q = Query(it["id"], "", gfunc, {safe_name(gname): params})
                    ok, cat, detail = score(q, resp, names)
                    gold = safe_name(gname)
                rec = {"key": key, "model": m["tag"], "split": split, "kind": kind, "query_id": it["id"],
                       "native_n": len(funcs), "gold": gold,
                       "gold_pos": names.index(gold) if gold in names else None,
                       "correct": ok, "category": cat, "detail": detail,
                       "ctx_overflow_risk": (resp.get("prompt_eval_count") or 0) >= cfg["num_ctx"] - 512,
                       "response": resp, "commit": commit, "ts": datetime.now().isoformat(timespec="seconds")}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_new += 1
                n_ok += ok
                if cat == "error":
                    logging.warning("request error: %s %s", key, detail[:200])
        if n_new:
            logging.info("model=%s native new=%d acc_new=%.3f time=%.0fs", m["tag"], n_new, n_ok / n_new, time.time() - t0)
    logging.info("finished %s", cfg["name"])


if __name__ == "__main__":
    main()
