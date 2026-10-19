# Lab Notebook — Tool Selection in Small LLM Agents

> Append-only research log. One entry per working session: what was done, why,
> what was observed, and what was decided. Never edit old entries; add corrections
> as new entries. (Written by the AI research assistant together with Faijan.)

---

## 2026-10-18 · Entry 1 — Setup and environment

**Goal of the session:** set up a reproducible project, choose data, and pilot the setup.

- Machine: Windows 11, NVIDIA RTX 3050 Laptop (4 GB VRAM), 16 GB RAM, Python 3.12.2, Ollama 0.32.6.
  Full snapshot: `logs/env_20261018-221525.json` (model digests, package versions).
- Python environment: `uv venv .venv`, exact versions in `requirements.lock.txt`.
- Models already installed in Ollama (all report the `tools` capability via `/api/show`):

  | Ollama tag | Params | Quant | Native ctx |
  | --- | --- | --- | --- |
  | qwen2.5:1.5b-instruct | 1.5B | Q4_K_M | 32k |
  | qwen2.5:3b-instruct | 3.1B | Q4_K_M | 32k |
  | llama3.2:latest (3B Instruct) | 3.2B | Q4_K_M | 128k |
  | qwen3.5:2b | 2.3B | **Q8_0** | 256k |

  ⚠️ Confound to report: qwen3.5:2b is 8-bit, the others 4-bit.
- Gotcha: `ollama show` (CLI) hung for >2 min because the CLI tried to start the desktop app.
  Workaround: query the HTTP API (`curl http://127.0.0.1:11434/api/show -d '{"model":...}'`).

## 2026-10-18 · Entry 2 — Data choice

- Benchmark: **BFCL v4 `simple_python`** (400 queries, each with exactly one gold function and a
  list of accepted argument values). Pinned to `ShishirPatil/gorilla@f7cf735` (2025-12-17), the
  commit matching PyPI `bfcl-eval==2025.12.17`. SHA256 of the files: `data/raw/SHA256SUMS.txt`.
  License Apache-2.0 (`data/raw/LICENSE_gorilla.txt`).
- Why this data: automatic, well-known scoring; one gold tool per query makes "which tool?" unambiguous;
  the 400 gold functions form a natural pool of realistic distractors.
- Facts found while loading: 400 queries → **370 unique function names**; **27 names appear with
  different docs** in different queries. Decision: each query always sees *its own* doc for its gold
  tool; distractors use the first occurrence.
- Also downloaded `live_simple` (258 user-contributed queries) for a possible robustness check later.

## 2026-10-18 · Entry 3 — Faithfulness to BFCL scoring

- Vendored BFCL's Python AST checker (`src/toolscale/bfcl_checker.py`), keeping only the
  single-call path and removing Java/JS branches; logic otherwise unchanged (header lists edits).
- Replicated BFCL's FC-mode preprocessing (`src/toolscale/data.py:to_tool`): append
  " Note that the provided function is in Python 3 syntax.", map types with GORILLA_TO_OPENAPI
  (dict→object, float→number + "This is a float type value.", tuple→array, any→string),
  and replace "." with "_" in function names (as BFCL does for OpenAI-style FC models).

## 2026-10-18 · Entry 4 — Experiment design (decided before running)

- **Catalog construction:** for N>1, shuffle queries with a seed, split into blocks of N, the
  catalog of a block = the gold functions of its queries (padded with random pool functions if
  duplicates/short last block), order shuffled. Every query is evaluated exactly once per (N, seed).
  Distractors are therefore *real* tools of other tasks.
- **Why blocks?** All queries of a block share an identical tool prefix → Ollama reuses the KV cache.
  Pilot: first call in a block at N=64 ≈ 10–18 s, following calls ≈ 1–4 s. This makes the study
  feasible on a 4 GB laptop GPU.
- **Conditions:** N ∈ {1, 4, 16, 64}; seeds {0,1,2} (N=1 has no randomness → 1 run).
  num_ctx = 4096 / 4096 / 8192 / 16384 (pilot: N=64 prompts are 7.9k–10.1k tokens). Each record
  stores `ctx_overflow_risk` to prove no truncation happened.
- **Decoding:** temperature 0, seed = catalog seed, num_predict 512, single turn, no system prompt
  (as BFCL FC mode). qwen3.5 thinking disabled (`think: false`) for comparability and cost.
- **RQ3 (retrieval):** embed tools + queries, keep top-k tools of the same N=64 catalog, call the LLM
  with only those. Paired with RQ1 at N=64 (same catalog, same query) → McNemar tests.

## 2026-10-18 · Entry 5 — Pilot findings (important!)

1. **Llama-3.2-3B emits numbers/booleans as strings** (`{"base": "10"}`), which BFCL's strict checker
   marks as type errors. In the smoke test, 18 of 30 Llama calls failed only for this reason.
   Decision: report **strict** (BFCL-identical) accuracy as the primary metric and **lenient**
   accuracy (string→schema-type coercion, `scoring.coerce_args`) as a secondary metric, computed
   offline from the same raw responses.
2. **Silent drops by the serving stack.** Some qwen2.5-3B responses had `eval_count≈35` tokens but
   *empty* content and *no* tool call. Using Ollama's `logprobs` to recover the generated tokens:
   `<tool_call>\n{"name": air_quality, "arguments": {...}}\n</tool_call>` — the **function name is
   unquoted → invalid JSON → Ollama silently discards the call**. In 2 of 4 cases the call was
   otherwise *correct*. Without the raw tokens this would be mislabeled "no call".
   Fix: `llm.py` now calls `/api/chat` with `logprobs: true` and stores `raw_output`;
   taxonomy category renamed `unparsed_call` → `malformed_call`.
3. **Non-determinism at temperature 0.** The same request gave different outputs depending on
   whether the KV cache was reused (e.g. simple_python_159: empty vs. a `get_case_info` call).
   Caching/batching changes floating-point results. → Report variance across seeds; don't claim
   bitwise reproducibility.
4. qwen3.5:2b sometimes answers math questions directly in text instead of calling the tool
   ("To find the area of a triangle ... = 25"), even at N=1 → `no_call`.

## 2026-10-18 · Entry 6 — RQ3a retrieval recall (no LLM)

`scripts/retrieval_recall.py` → `results/processed/retrieval_recall.csv`. Mean over 3 seeds:

| Embedder | N | R@1 | R@3 | R@5 | R@10 |
| --- | --- | --- | --- | --- | --- |
| qwen3-embedding:0.6b | 64 | 0.960 | 0.999 | 1.000 | 1.000 |
| bge-m3 | 64 | 0.907 | 0.988 | 0.997 | 0.998 |

Decision: use **qwen3-embedding:0.6b, k=3** for the main retrieval run (`configs/rq3_retrieval.yaml`),
k ∈ {1, 5} as a seed-0 ablation. Caveat: BFCL queries are lexically close to their function
descriptions, so retrieval here is likely easier than in the wild.

## 2026-10-18 · Entry 7 — Full run launched

- Code commit `3955baf`. Command: `bash scripts/run_all.sh` (resumable).
- Expected: RQ1 = 4 models × 10 conditions × 400 = 16,000 calls; RQ3 = 4,800; ablation = 3,200.
- Console log: `logs/run_all_console.log`; per-run logs `logs/<config>_<timestamp>.log`.

## 2026-10-18: Literature review and novelty check (AI-assisted)

**What was done**
- Wrote the search protocol (`notes/protocol.md`): review questions, sources, exact search strings, inclusion/exclusion criteria, and a log of every search actually run.
- Ran 8 arXiv API searches (about 1,156 total hits; 281 screened by title and abstract), 15 web searches for discovery and snowballing, and known-item lookups.
- Semantic Scholar keyword search returned HTTP 429 on every call, and DBLP was blocked by bot protection. Used the S2 `/paper/batch` endpoint (which worked) and the full ACL Anthology BibTeX dump for venue verification instead.
- Verified 57 references through the arXiv API, ACL Anthology, PMLR, or neurips.cc/iclr.cc pages. Wrote them to `paper/references.bib`, each with a `% verified:` URL. Test-compiled with BibTeX + natbib: 0 warnings.
- Created `notes/paper_tracker.csv` (57 rows), 12 paper notes in `notes/papers/`, `notes/novelty_check.md`, and `notes/related_work_draft.md` (about 550 words, 56 keys, all present in the bib).

**Key findings**
- The general claim that "more tools means lower accuracy" is already established. LongFuncEval (BFCL plus random distractors and a gold-position sweep, models of 8B and up, 49 to 741 tools) is the closest prior work to RQ1. RAG-MCP covers RQ1 and RQ3 with a large model. Rabinovich & Anaby-Tavor covers semantically similar distractors in large models.
- Concurrent 2026 work on small models is arriving: AgentWeave (Hammer2.1-1.5B on BFCL with padded distractors, where semantic top-8 solved 0/48 tasks) and the NeurIPS'26 SLM-Agents workshop papers (failure taxonomies for Qwen3-1.7B/4B and other GGUF SLMs under quantization).
- Still open: a controlled study of sub-4B quantized instruct models in the small-N regime (1 to 64 tools) that crosses random vs similar distractors with gold position and retrieval top-k, and that reports how the failure-type mix shifts. Also open: a decomposition of retrieval mitigation into recall@k and accuracy given the gold tool was retrieved.
- Recommendations:
  - Split RQ1 by distractor type (random vs similar).
  - Make position an explicit, balanced factor.
  - Phrase RQ2 as how the failure mix shifts across conditions.
  - Decompose RQ3 into recall@k and conditional accuracy.
  - Deprioritize the quantization RQ, since that area is now crowded.
- Qwen3.5 has no general technical report (only Qwen3.5-Omni). Cite the model card.

**Next steps**
- Mentor review of `novelty_check.md`.
- Re-run the Semantic Scholar and DBLP searches with an API key or a browser.
- Do pass-2 reading of LongFuncEval, AgentWeave, and roshaan2026 / singh2026.

---

## 2026-10-18/19 · Entry 9 — Incident: disk full, run stopped

- At ~22:34 the C: drive hit 100 % (242 MB free) while I test-installed `bfcl-eval` with all its
  dependencies (it pulls `sentence-transformers` → PyTorch). Ollama crashed mid-request:
  one record `N4|s2|simple_python_219` has category `error` (`WinError 10054`). The Claude session
  also ended, so the background run stopped after 1,558 calls (qwen2.5-1.5B only).
- Checked integrity: all 1,558 JSON lines parse; nothing lost.
- **Lessons:** check `df -h` before installing packages; install research packages with
  `--no-deps` when you only need a few pure-Python modules; make the runner *retry* failed calls
  (fixed: resume now skips only non-error records; analysis keeps the last record per call).

## 2026-10-19 · Entry 10 — Scope decision: small "learning" version

Faijan asked for a small experiment + paper for learning. New design (decided before seeing more data):

| Experiment | Data | Scale |
| --- | --- | --- |
| RQ1/RQ2 scaling | BFCL simple_python, **fixed random 100 of 400 queries** (`eval_subset`, seed 0), catalogs still built from all 400 tools | N ∈ {1,4,16,64}, seeds {0,1} |
| RQ3 retrieval | same 100 queries, N=64 catalogs of seed 0 | k ∈ {1,3,5} (k-ablation merged) |
| RQ3a recall | BFCL catalogs, no LLM | unchanged |

Records already computed for the other 300 queries / seed 2 are kept on disk but filtered out in
`analyze.py`, so the reported design is exactly the one above.

## 2026-10-19 · Entry 11 — Using existing (native) datasets

Faijan asked to prefer existing datasets over constructed ones. Surveyed options (HF + literature):

| Dataset | Native tools/query | Gold args | Agentic | Decision |
| --- | --- | --- | --- | --- |
| BFCL `live_multiple` (1,053 real queries) | 2–37 | yes | no | **use** (RQ4) |
| BFCL `live_irrelevance` (884) | 0–37 | n/a (should not call) | no | **use** (RQ4) |
| BFCL `multi_turn_base` (200, executable APIs) | 9–42 | state/result checks | **yes** | **use** (RQ5) |
| ToolRet (`mangopy/ToolRet-*`, 44,453 tools, 7,961 queries) | full corpus | no (relevance labels only) | no | **use** for retrieval (RQ3b) |
| Seal-Tools (`casey-martin/Seal-Tools`, MIT) | always 5 | yes | multi-call | not used: N never varies |

Note for the paper: our RQ1 catalogs are *not* a new dataset — they are subsets of BFCL's own tools,
the same protocol as LongFuncEval / AgentWeave. The native experiments check that conclusions hold
without any construction.

Implementation notes:
- `bfcl-eval==2025.12.17` installed with `--no-deps`: the multi-turn environments and the official
  `multi_turn_checker` are pure Python and import fine without PyTorch.
- `scripts/run_multiturn.py` mirrors `BaseHandler.inference_multi_turn_FC` (20-step limit per turn,
  text reply ends a turn, tool results appended as `tool` messages, calls executed as
  `func(arg=repr(value))` strings). **Sanity check:** feeding the ground truth through our decoding
  path → checker says valid for 5/5 entries; empty answers → invalid.
- Smoke episode (`multi_turn_base_64`, qwen2.5-1.5B): the model called `lockDoors(unlock=False)`,
  got "missing 1 required positional argument: 'door'", then gave up in text. In the next turn it
  read tire pressures of 32–35 PSI and replied "all tires are within the safe range of 40 PSI or
  higher" — a hallucinated reading of a tool result. Good qualitative example.
- ToolRet: 600 queries (200 per category, seed 0) against the **full** 44,453-tool corpus; documents
  truncated to 2,000 chars (99th percentile is far above the median of 439 chars).
- Data provenance: all new files are in `data/raw/` with SHA256 in `data/raw/SHA256SUMS.txt`;
  ToolRet revisions Queries@b8c76ad3, Tools@e06c38c7.
- Commits: `e1dcd24` (native experiments). Queue: `scripts/queue_extra.sh` starts
  `scripts/run_extra.sh` as soon as the core run prints "CORE EXPERIMENTS FINISHED".

## 2026-10-19 · Entry 12 — First RQ1 numbers (qwen2.5-1.5B, 100 queries)

| N | 1 | 4 (s0, s1) | 16 (s0, s1) | 64 (s0) |
| --- | --- | --- | --- | --- |
| strict acc. | ≈0.72 | ≈0.72–0.75 | 0.57, 0.51 | 0.44 |

(Read from the run log; the official numbers come from `scripts/analyze.py`.) Accuracy is flat up
to 4 tools and then drops sharply — the interesting region is 4→64.

## 2026-10-18 · Entry 13 — Incident: low system RAM, background jobs stopped

- 23:18: Claude Code stopped the Ollama server and both experiment jobs because system RAM was
  critically low (free ≈ 3.3 GB of 15.2 GB; Ollama's `llama-server` held 4.2 GB). Cause: Llama-3.2-3B
  with a 16k context does not fit in 4 GB VRAM, so part of its KV cache lives in system RAM.
- Saved results were intact (every JSON line parses). One request failed (`N16|s0|simple_python_137`,
  connection reset) and will be retried automatically on resume.
- Fix (no numerical effect): restarted Ollama with `OLLAMA_NUM_PARALLEL=1` and
  `OLLAMA_MAX_LOADED_MODELS=1`, so only one model and one context slot are held at a time.
  Deliberately did **not** enable KV-cache quantization (`OLLAMA_KV_CACHE_TYPE`), because that would
  change model outputs halfway through the study.
- Relaunched `run_all.sh && run_extra.sh` (resumable).

## 2026-10-18 · Entry 14 — Incident 2: orphaned runners, duplicates, corrupt lines

- The low-memory stop at 23:55 killed the *shell* wrappers but **not** the Python runners. Process
  list showed two `run_experiment.py configs/rq1_scaling.yaml` processes (started 22:46 and 23:20)
  appending to the same files. This doubled GPU/RAM load (likely the real cause of incident 2) and,
  once Ollama was gone, they wrote "connection refused" error records. I stopped both (PIDs
  21276/21280, 17384/19440).
- Damage on `llama3.2_latest.jsonl`: 2 corrupt (interleaved) lines → removed with
  `scripts/repair_jsonl.py` (backup `*.bak-<timestamp>`); 177 calls completed twice; 11 calls with
  only error records.
- Fixes: (1) `analyze.py` keeps the **first successful** record per call (error records only if no
  success exists); (2) new `src/toolscale/runlock.py` — a runner refuses to start if another live
  process holds `<out_dir>/.lock`; (3) recommend running long jobs in a separate terminal, not as
  Claude Code background jobs.
- Lesson: when a job is killed, *check the process list* before relaunching.

## 2026-10-19 · Entry 15 — Relaunch with memory constraints

- Free RAM before relaunch: 2.0 GB (Chrome 3.6 GB, VS Code 2.7 GB, Slack 1.0 GB). The Ollama desktop
  app had restarted its own server without our settings; stopped it and started
  `ollama serve` with `OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1`.
- **Design change (compute constraint, decided before seeing RQ5 results):** Llama-3.2-3B is excluded
  from the multi-turn agent experiment (RQ5), because its 16k-context KV cache spills into system
  RAM and was the main trigger of both shutdowns. It remains in RQ1–RQ4. To add it back, uncomment
  one line in `configs/rq5_multiturn.yaml`.

## 2026-10-19 · Entry 16 — Stalled requests; shorter timeout + retry pass

- During Qwen3.5-2B (N=4) some requests stalled for 8–10 min (latency 477 s; two 600 s timeouts)
  while normal calls took ~2 s. GPU utilisation was low and Ollama's log shows no slow request,
  so the stall happens outside generation (likely memory paging: 1.6 GB RAM free).
- Also observed: Qwen3.5 is a hybrid/recurrent architecture, so llama.cpp often cannot reuse a
  cached prefix ("forcing full prompt re-processing ... hybrid/recurrent memory", 87 times).
  This makes it slower, but does not change outputs.
- Change (no effect on results): client timeout 600 s → 180 s (slowest normal call ≈ 30 s), and
  `scripts/run_everything.sh` now ends with a retry pass that re-runs only failed calls.
- Note to self: when stopping processes on Windows, match the exact Python script path; a broad
  pattern also matched this session's own helper shells.
- Ollama's console log is UTF-16 (PowerShell `Tee-Object`); read it with `iconv -f UTF-16LE -t UTF-8`.

## 2026-10-19 · Entry 17 — Qwen3.5 requests that hang (serving-stack bug), counted as failures

- The "stalls" of Entry 16 are **not** random paging: the same Qwen3.5-2B requests hang every time
  (simple_python_389, _41, _348 at N=4, seed 0). A streamed reproduction received *no first chunk*
  within 170 s, i.e. Ollama 0.32.6 hangs before emitting any token. Other models never hang on the
  same prompts. Probably related to the hybrid/recurrent architecture (Entry 16).
- Decision (applies to all models): a request without a response within the 180 s client timeout
  is a final failure with category **`timeout`** (failure group "Other"); it is not retried. A
  deployed agent would hang too. The paper reports the timeout rate per model.
- Implementation: runners treat ReadTimeout records as done; `scoring.py` and `analyze.py` map them
  to `timeout`; connection resets / server crashes stay `error` and are still retried.

## 2026-10-19 · Entry 18 — CORRECTION of Entry 17: the server was wedged, not the queries

- Entry 17 was wrong. After the "hangs", even a trivial request (`"Say hi"`, 5 tokens) did not
  answer within 60 s, while `/api/ps` and `/api/version` still responded. So Ollama 0.32.6 had
  **wedged** at ~13:08 (during Qwen3.5-2B), and *every* later request hung — the same query ids
  just happened to be next in line after each restart of the runner. Killing the model's
  `llama-server` process was not enough; the main `ollama.exe` server had to be restarted.
- The PowerShell window launcher (`start_ollama_lowmem.ps1` via `Start-Process`) silently failed
  to start the server this time; replaced by `scripts/start_ollama_lowmem.cmd` (`cmd /c start`).
- New policy (replaces Entry 17): `llm.py` has a **watchdog**. On a client timeout (180 s) it
  (1) kills `llama-server` and probes with a 5-token request; (2) if the probe fails, restarts the
  whole Ollama server; then retries the request once. Only a timeout *after* a watchdog restart is
  a final `timeout` failure. All records produced while the server was wedged are category
  `error` and are re-run.
- Lesson: before blaming the input, test the system with a trivial input. One minimal probe would
  have shown at once that the server, not the query, was the problem.

## 2026-10-19 · Entry 19 — Taxonomy refinement: what "malformed" really contained

- Observation: *hallucinated tool* was 0.0 % for every model, although the pilot had shown a call to
  a non-existent `recipe_search`. Parsing the saved raw tokens of all 66 "malformed_call" records:
  48 invalid JSON (mostly Qwen2.5-3B, unquoted function names), 6 **valid calls to unknown tool
  names that Ollama silently dropped**, 8 valid JSON calls **written as prose/markdown** instead of a
  tool call (Qwen2.5-1.5B), 4 other.
- Change: `scoring.py` now inspects the raw tokens when no call was parsed → `hallucinated_tool`
  (dropped by server) / `text_call` / `malformed_call`. `analyze.py` and `analyze_extra.py`
  **re-derive every category from the saved raw output**, so old and new records use one taxonomy.
  Figures group `malformed_call` + `text_call` as "Format error".
- Lesson: a category that is always zero deserves suspicion — check that the measurement can
  actually observe it.

## 2026-10-19 · Entry 20 — Core results in; manual checks behind three claims in the text

- Core runs (RQ1–RQ3) finished 15:19. Analysis: `scripts/analyze.py` (commit 0055fce + later).
- Fixed an analysis bug before writing: the retrieval summary had averaged k=1,3,5 together; now the
  main row is k=3 (as planned in `configs/rq3_retrieval.yaml`), paired per query with the seed-0 full
  catalog, and k∈{1,5} form the ablation table.
- Manual checks (so the prose is backed by data):
  1. Random sample of 14 `wrong_tool` errors (pandas `sample(14, random_state=0)`): 13 are plausible
     near-duplicates (e.g. solve_quadratic_equation→solve_quadratic, get_prime_factors→
     number_analysis_prime_factors, calculate_entropy_change→entropy_change_calculate); 1 unrelated
     (game_valuation→book_hotel). → BFCL's overlapping tools make some large-N "errors" ambiguous.
  2. Qwen2.5-3B `malformed_call`: 44 of 48 have an unquoted function name (now computed in analyze.py).
  3. Qwen3.5-2B `no_call` at N=1 (16 queries): 12 step-by-step math answers, 4 factual answers from
     memory (e.g. "Radium was discovered by Marie Curie").
- Position effect: Llama p=0.043, Qwen3.5 p=0.032, others >0.29 → none survive Bonferroni (4 tests);
  reported as "no consistent position effect".

## 2026-10-19 · Entry 21 — RQ4 done; RQ5 crashed on a missing dependency

- RQ4 (native BFCL) finished 15:46 for all four models.
- RQ5 crashed at once: `ModuleNotFoundError: No module named 'mpmath'` — BFCL's executable `MathAPI`
  needs it, and `bfcl-eval` was installed with `--no-deps`. The retry pass hit the same error; the
  chain ended with exit=1. Fix: `uv pip install mpmath` (lock file updated). Checked every BFCL
  environment module: all classes used by `multi_turn_base` import; only memory/web-search ones
  (not used here) still miss `overrides`/`html2text`.
- Lesson: after a `--no-deps` install, import every module you will use *before* the long run
  (the smoke test only exercised VehicleControlAPI).
- Relaunched `scripts/run_extra.sh` (RQ4 is skipped as done; RQ5 + ToolRet run).

## 2026-10-19 · Entry 22 — All experiments finished; paper compiled

- RQ5 finished 16:17; ToolRet (RQ3b) finished ~17:20 (embedding 44,453 tools took ≈30 min per
  embedder). Retried the two Qwen3.5 episodes with HTTP 500: multi_turn_base_48 now completes
  (fails for model reasons); multi_turn_base_22 hits the 500 again after 24 calls (likely context
  exhaustion from over-acting; not verified because the cmd-window Ollama does not log to file).
- Final analysis: `logs/analyze_final.log`, `logs/analyze_extra_final.log`; 372 macros.
- Headline numbers (see paper/main.pdf):
  * RQ1: flat to N=4; at N=64 Qwen2.5-3B −25.0, Qwen2.5-1.5B −21.5 pts (McNemar p<0.001);
    Llama-3.2-3B −17.5 pts only under lenient scoring; Qwen3.5-2B −14 pts from its N=4 peak.
  * RQ2: wrong/hallucinated tool ≤10 % even at N=64 (mostly near-duplicates); growth in "no call"
    (1.5B: 4→19.5 %) and format errors (3B: 0→9 %, 13.5 % at N=16; 44/48 unquoted names).
  * RQ3: top-3 retrieval +10…+16 pts (all p<0.05); Qwen2.5-3B top-1 81 % ≈ gold-only 83 %.
    ToolRet full corpus: nDCG@10 27.1 (bge-m3) / 40.0 / 48.7 (qwen3-emb generic / task instr.).
  * RQ4: no trend with native N; wrong tool 4–6 %; Llama calls a tool in 92 % of irrelevant cases.
  * RQ5: success 6.7 / 23.3 / 20.0 % (Qwen2.5-1.5B / 3B / Qwen3.5-2B); 1.5B under-acts,
    Qwen3.5 over-acts (1.42 calls per GT call); ~6 % of Qwen2.5 agent steps are dropped calls.
- Paper: TMLR style, 17 pages, clean build (no errors, no overfull boxes, no undefined refs).
- Gotcha: VS Code's LaTeX Workshop rebuilds on save and races with `latexmk` (transient
  "main.log not found"/spurious errors). Use `latexmk -C && latexmk -pdf` for a clean final build.

### Open items before any submission (for Faijan)
1. Read every cited paper you rely on; double-check `mastoi2026beyond` author list and the 2026
   workshop papers (verified only via conference pages).
2. Manually label ~100 failures yourself and report agreement with the automatic taxonomy.
3. Disclose AI assistance per the venue's policy; rewrite the prose in your own words.
4. Optional scale-up: `subsample: 400`, `seeds: [0,1,2]` in the configs (≈ 10–15 h).
