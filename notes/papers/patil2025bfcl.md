# The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models

- **Key:** `patil2025bfcl` · ICML 2025, PMLR 267:48371–48392 · Patil, Mao, Yan, Ji, Suresh, Stoica, Gonzalez (UC Berkeley)
- **Read:** pass 1 (PMLR page and abstract); the leaderboard docs need a separate read. 2026-10-18
- **Relevance:** 3/3 (our dataset and metric)

## Problem
There was no standard benchmark for function calling, for two reasons: it is hard to decide when a call is valid, and hard to collect diverse real-world functions.

## Method
- **AST evaluation:** parse the generated call and match the function name, required parameters and allowed values against the possible answers. This scales to thousands of functions without executing them.
- **Data:** expert-curated plus user-contributed functions. Categories cover simple, multiple, parallel and parallel-multiple calls, relevance/irrelevance, several languages, and live data.
- **Versions:** v3 added multi-turn; v4 added agentic evaluation (web search, memory, format sensitivity).

## Experiments
A large panel of proprietary and open models on the leaderboard, including small ones.

## Key results
- Frontier models do well on single-turn calls.
- Memory, dynamic decision-making and long-horizon reasoning remain open problems.

## Limitations
- `simple` provides a single candidate function, so it does not test selection at all. That is why we build catalogs ourselves.
- `multiple` has only 2–4 candidates.

## Relation to our paper
- We use BFCL v4 `simple_python` (400 queries with one gold function each) and its AST checker.
- Our pooled catalogs (N ∈ {1, 4, 16, 64}) turn `simple` into a controlled selection task. LongFuncEval (`kate2025longfunceval`) and BoR (`repantis2026howmany`) build similar BFCL-derived pools, which supports this choice as standard practice.
- Report BFCL version and commit, and follow their AST checker exactly for comparability.
