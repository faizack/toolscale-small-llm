"""RQ5 (agentic): BFCL v4 `multi_turn_base` with BFCL's executable tool environments.

Mirrors bfcl_eval BaseHandler.inference_multi_turn_FC (bfcl-eval==2025.12.17):
  * tools = function docs of all involved API classes (9-42 native tools);
  * for each user turn, loop: model call -> if it returns tool calls, execute them on the
    stateful BFCL Python environment and append the results as `tool` messages; a reply
    without tool calls ends the turn; at most MAXIMUM_STEP_LIMIT (20) steps per turn;
  * scoring with BFCL's official `multi_turn_checker` (state + execution-result checks).

Usage: .venv/Scripts/python.exe scripts/run_multiturn.py configs/rq5_multiturn.yaml
Output: results/raw/rq5_multiturn/<model>.jsonl (one line per episode, full trajectory)
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
from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolscale.data import RAW, to_tool  # noqa: E402
from toolscale import runlock  # noqa: E402
from toolscale.llm import chat_with_tools  # noqa: E402

from run_experiment import git_commit  # noqa: E402


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_call_string(name: str, args: dict) -> str:
    """bfcl_eval.model_handler.utils.convert_to_function_call for one call."""
    return f"{name}({','.join(f'{k}={v!r}' for k, v in args.items())})"


def run_episode(model: dict, entry: dict, cfg: dict, run_tag: str) -> dict:
    funcs = []
    for cls in entry["involved_classes"]:
        funcs += read_jsonl(RAW / "multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls])
    tools = [to_tool(f) for f in funcs]
    names = {t["function"]["name"] for t in tools}
    messages: list[dict] = []
    decoded_turns: list[list[list[str]]] = []
    trace, n_calls, n_hallucinated, n_exec_errors, n_steps, forced, errors = [], 0, 0, 0, 0, False, 0
    t0 = time.time()
    for turn_idx, turn_msgs in enumerate(entry["question"]):
        messages += turn_msgs
        steps: list[list[str]] = []
        for step in range(MAXIMUM_STEP_LIMIT + 1):
            resp = chat_with_tools(model["tag"], messages, tools, num_ctx=cfg["num_ctx"],
                                   seed=cfg["seed"], think=model.get("think"))
            n_steps += 1
            if "error" in resp:
                errors += 1
                trace.append({"turn": turn_idx, "step": step, "error": resp["error"]})
                break
            messages.append(resp["message"])
            calls = resp["tool_calls"]
            trace.append({"turn": turn_idx, "step": step, "content": resp["content"][:500],
                          "raw": resp["raw_output"][:800], "calls": calls,
                          "prompt_tokens": resp["prompt_eval_count"], "latency_s": round(resp["latency_s"], 2)})
            if not calls:  # text reply (or undecodable output) ends the turn, as in BFCL
                break
            decoded = [to_call_string(c["name"], c["arguments"]) for c in calls]
            n_calls += len(decoded)
            n_hallucinated += sum(c["name"] not in names for c in calls)
            results, _ = execute_multi_turn_func_call(decoded, entry["initial_config"], entry["involved_classes"],
                                                      run_tag, entry["id"], long_context=False, is_evaL_run=False)
            n_exec_errors += sum(str(r).startswith("Error") or '"error"' in str(r).lower() for r in results)
            steps.append(decoded)
            for c, r in zip(calls, results):
                messages.append({"role": "tool", "content": str(r), "tool_name": c["name"]})
            trace[-1]["results"] = [str(r)[:300] for r in results]
            if step == MAXIMUM_STEP_LIMIT:
                forced = True
        decoded_turns.append(steps)
        if forced:
            break
    gt = entry["_ground_truth"]
    while len(decoded_turns) < len(gt):  # force-quit episodes are scored with empty remaining turns
        decoded_turns.append([])
    try:
        check = multi_turn_checker(decoded_turns, gt, entry, "multi_turn_base", run_tag)
    except Exception as e:  # noqa: BLE001 - record checker crashes instead of losing the episode
        check = {"valid": False, "error_type": f"checker_exception:{type(e).__name__}", "error_message": str(e)[:300]}
    return {"valid": bool(check.get("valid")), "error_type": check.get("error_type", ""),
            "error_message": str(check.get("error_message", ""))[:400],
            "n_turns": len(entry["question"]), "n_tools": len(tools), "n_steps": n_steps, "n_calls": n_calls,
            "n_gt_calls": sum(len(t) for t in gt), "n_hallucinated": n_hallucinated,
            "n_exec_errors": n_exec_errors, "forced_quit": forced, "request_errors": errors,
            "decoded": decoded_turns, "trace": trace, "seconds": round(time.time() - t0, 1)}


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

    entries = read_jsonl(RAW / "BFCL_v4_multi_turn_base.json")
    gts = {a["id"]: a["ground_truth"] for a in read_jsonl(RAW / "possible_answer_BFCL_v4_multi_turn_base.json")}
    rng = random.Random(f"multiturn-{cfg['seed']}")
    sample = sorted(rng.sample(sorted(entries, key=lambda e: e["id"]), cfg["n_episodes"]), key=lambda e: e["id"])
    logging.info("multi_turn_base: %d entries, %d sampled", len(entries), len(sample))

    for m in cfg["models"]:
        path = out_dir / f"{m['tag'].replace(':', '_')}.jsonl"
        done = set()
        if path.exists():
            done = {r["id"] for r in map(json.loads, filter(str.strip, open(path, encoding="utf8")))
                    if r["request_errors"] == 0 or "after watchdog restart" in json.dumps(r["trace"])}
        run_tag = m["tag"].replace(":", "_").replace(".", "_") + "_toolscale"
        t0, n_new, n_ok = time.time(), 0, 0
        with open(path, "a", encoding="utf8") as fh:
            for e in sample:
                if e["id"] in done:
                    continue
                e = dict(e, _ground_truth=gts[e["id"]])
                res = run_episode(m, e, cfg, run_tag)
                rec = {"id": e["id"], "model": m["tag"], **res, "commit": commit,
                       "ts": datetime.now().isoformat(timespec="seconds")}
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                n_new += 1
                n_ok += res["valid"]
                logging.info("model=%s %s valid=%s steps=%d calls=%d/%d gt err=%s (%.0fs)", m["tag"], e["id"],
                             res["valid"], res["n_steps"], res["n_calls"], res["n_gt_calls"], res["error_type"],
                             res["seconds"])
        if n_new:
            logging.info("model=%s multiturn new=%d acc_new=%.3f time=%.0fs", m["tag"], n_new, n_ok / n_new, time.time() - t0)
    logging.info("finished %s", cfg["name"])


if __name__ == "__main__":
    main()
