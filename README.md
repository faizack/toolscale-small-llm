# How Many Tools Can a Small Language Model Handle?

Scaling and failure analysis of tool selection in small (1.5B–3B) LLM agents, run entirely on a
**4 GB laptop GPU** with Ollama. Everything in the paper (every number, table and figure) is
regenerated from raw model outputs by the scripts below.

- Plan and learning roadmap: [`ROADMAP.md`](ROADMAP.md)
- Day-by-day research log (decisions, bugs, surprises): [`notes/lab_notebook.md`](notes/lab_notebook.md)
- Literature: [`notes/protocol.md`](notes/protocol.md), [`notes/paper_tracker.csv`](notes/paper_tracker.csv),
  [`notes/novelty_check.md`](notes/novelty_check.md)
- Paper source: [`paper/main.tex`](paper/main.tex)

## Results at a glance (small "learning" scale, 2026-10-19)

| Finding | Evidence |
| --- | --- |
| Accuracy is flat up to ~4 tools, then drops (−21.5 / −25.0 pts at 64 tools for Qwen2.5-1.5B / 3B) | `paper/figures/fig_scaling.pdf`, Table 1 |
| The extra loss is mostly *no call* and *invalid call syntax*, not wrong-tool picks | `paper/figures/fig_failures.pdf`, Table 2 |
| Ollama silently drops malformed / unknown-tool calls → log raw tokens (`logprobs`) | lab notebook Entries 5, 19 |
| Llama-3.2-3B writes numbers as strings → strict scoring hides a 17.5-pt decline | Table 1 (lenient rows) |
| Top-3 embedding retrieval recovers +10…+16 pts; on ToolRet's 44k tools retrieval itself is the bottleneck | Tables 3, 4, 7 |
| Native data: wrong-tool errors rare; Llama calls a tool in 92 % of irrelevant queries | Table 5 |
| Multi-turn agents: ≤ 23 % episode success | Table 6 |

The compiled paper is `paper/build/main.pdf` (build with `cd paper && latexmk`, or open the folder in VS Code with LaTeX Workshop and save any `.tex` file). **Before submitting anywhere, read `LEARNING_GUIDE.md`
("Before you submit") and the open items at the end of the lab notebook.**

## Research questions

| RQ | Question | Where |
| --- | --- | --- |
| RQ1 | How does tool-calling accuracy change as the catalog grows from 1 to 64 tools? | `configs/rq1_scaling.yaml` |
| RQ2 | Which failure types appear, and how does the mix change with N? | same runs, `src/toolscale/scoring.py` |
| RQ3 | Does embedding-based top-k tool retrieval recover the lost accuracy? | `configs/rq3_retrieval.yaml`, `scripts/retrieval_recall.py`, `scripts/toolret_retrieval.py` |
| RQ4 | Do the trends hold on *native* tool lists (real user queries, no construction)? | `configs/rq4_native.yaml` (BFCL live_multiple / live_irrelevance) |
| RQ5 | How do the same models do as multi-turn agents on executable environments? | `configs/rq5_multiturn.yaml` (BFCL multi_turn_base, official checker) |

## Reproduce everything

Hardware used: Windows 11, RTX 3050 Laptop (4 GB), 16 GB RAM. Total compute ≈ 4 h (small "learning" scale).

```bash
# 0) prerequisites: Python 3.12, uv, Ollama >= 0.32, MiKTeX/TeX Live with latexmk
# 1) models (exact digests are in logs/env_*.json)
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:3b-instruct
ollama pull llama3.2            # Llama-3.2-3B-Instruct, Q4_K_M
ollama pull qwen3.5:2b
ollama pull qwen3-embedding:0.6b
ollama pull bge-m3

# 2) Python environment (exact versions)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --no-deps -r requirements.lock.txt   # Linux/macOS: .venv/bin/python
# --no-deps matters: bfcl-eval would otherwise pull PyTorch (~5 GB); we only need its pure-Python parts

# 3) data is already in data/raw (BFCL v4, gorilla@f7cf735); verify integrity
sha256sum -c data/raw/SHA256SUMS.txt

# 4) run all experiments (resumable: re-run after any interruption)
bash scripts/run_all.sh      # RQ1-RQ3 on BFCL simple_python (~2 h)
bash scripts/run_extra.sh    # RQ4 native BFCL, RQ5 multi-turn agent, RQ3b ToolRet (~2 h)

# 5) analysis -> results/processed/, paper/tables/, paper/figures/
.venv/Scripts/python.exe scripts/analyze.py
.venv/Scripts/python.exe scripts/analyze_extra.py
.venv/Scripts/python.exe scripts/make_macros.py

# 6) build the paper
cd paper && latexmk      # PDF -> paper/build/main.pdf (VS Code: just save a .tex file)
```

Quick smoke test (≈ 5 min): `.venv/Scripts/python.exe scripts/run_experiment.py configs/rq1_scaling.yaml --limit 3`

## Repository layout

```text
configs/            one YAML per experiment (the only place where settings live)
data/raw/           BFCL v4 files (Apache-2.0), ToolRet parquet files, checksums
src/toolscale/      data.py (catalogs), llm.py (Ollama), scoring.py (taxonomy),
                    bfcl_checker.py (vendored BFCL checker), retrieval.py
scripts/            run_experiment.py, run_all.sh, retrieval_recall.py,
                    analyze.py, make_macros.py, env_snapshot.py
results/raw/        one JSON line per model call (inputs, raw tokens, timings, score)
results/processed/  all_records.csv, summary.json, retrieval_recall.csv
logs/               environment snapshots and run logs
notes/              lab notebook, literature protocol, paper notes, novelty check
paper/              LaTeX (TMLR style), sections/, tables/ (generated), figures/ (generated)
```

## What each raw record contains

`key, model, N, seed, k, query_id, catalog_id, gold, gold_pos, gold_in_visible, visible_size,
num_ctx, ctx_overflow_risk, correct, category, detail, response{content, raw_output, tool_calls,
prompt_eval_count, eval_count, latency_s, ...}, config_hash, commit, ts`

## Known caveats (details in the lab notebook)

- Temperature-0 decoding is **not bitwise reproducible** with KV-cache reuse; expect small differences.
- Ollama silently drops malformed tool calls; we recover the raw tokens via `logprobs`.
- `qwen3.5:2b` is 8-bit while the other models are 4-bit.

## License and credits

BFCL data and the vendored checker: Apache-2.0, © the Gorilla/BFCL authors. Paper style: TMLR.
