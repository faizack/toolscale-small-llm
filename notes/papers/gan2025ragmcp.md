# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation

- **Key:** `gan2025ragmcp` · arXiv:2505.03275 (May 2025) · Tiantian Gan, Qiyao Sun · preprint
- **Read:** pass 2 (arXiv HTML), 2026-10-18
- **Relevance:** 3/3 (RQ1 + RQ3 + position, MCP framing)

## Problem
Putting every MCP server or tool description in the prompt causes "prompt bloat", and selection accuracy falls as the number of tools grows.

## Method
1. Index the MCP server descriptions in an external store.
2. Retrieve the most relevant MCP(s) for each query with a "lightweight LLM-based retriever (e.g., Qwen)". The exact embedder and k are not specified.
3. Inject only the selected schemas into the prompt.

**MCP stress test:** each trial has one ground-truth MCP and N−1 distractors drawn from mcp.so (more than 4,400 servers). N is swept, and the gold item is placed at different positions.

## Experiments
- **Benchmark:** MCPBench web-search subset, 20 trials per baseline.
- **Base LLM:** qwen-max-0125.
- **Baselines:** Blank Conditioning (all tools) and Actual Match (keyword).

## Key results
- **Accuracy:** RAG-MCP reaches 43.13%, against 18.20% for Actual Match and 13.62% for all tools.
- **Prompt tokens:** about 1,084 vs 2,134, roughly a 49% reduction (the abstract says "over 50%").
- **Stress test:** success is above 90% when the gold item is in positions below 30, failures cluster at positions 31–70, and failures dominate beyond about 100. The trend is described as non-monotonic.

## Limitations
- Very small evaluation (20 tasks per condition).
- Unnamed stress-test model, unspecified retriever and k, and few reported numbers.
- No small or local models and no error taxonomy.
- Evaluators are described inconsistently (DeepSeek-v3 vs a Llama judge).

## Relation to our paper
This is the best-known "retrieval fixes tool bloat" citation, and it has the same RQ1+RQ3 structure as our paper. Our contribution is to redo this carefully:

- local sub-4B models instead of qwen-max
- 400 BFCL queries with deterministic AST scoring instead of 20 web-search tasks
- named embedders (bge-m3 and qwen3-embedding)
- an explicit top-k sweep
- retrieval recall@k reported separately from end-to-end accuracy
- repeated runs with confidence intervals
