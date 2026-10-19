# Beyond Fluent Generation: A CPU Reliability Benchmark for MCP-Style Tool Calling in Sub-2B Small Language Models for Edge Deployment

- **Key:** `mastoi2026beyond` · arXiv:2609.07370 (Sep 2026) · author line on arXiv: "Abrar Shahriar Qurat-Ul-Ain Mastoi" (possibly two people; check before citing) · preprint, code at github.com/Abrar051/sml_mcp_benchmark
- **Read:** pass 1 (abstract), 2026-10-18
- **Relevance:** 3/3 (sub-2B models, local hardware, tool-name / argument / value scoring)

## Problem
For agents on single-board computers, MCP-style tool calls need valid JSON, the correct tool, complete arguments and correct values, not just fluent text.

## Method
- **Models:** Phi-1.5, Pythia-1.4B, TinyLlama-1.1B-Chat, Qwen2.5-0.5B and Qwen2.5-1.5B, run on CPU.
- **Data:** 100 prompts over **5 tools** (weather, web search, calculator, email, task creation).
- **Decoding:** greedy and nucleus sampling.
- **Scoring:** a recovery parser (strips Markdown fences, extracts the brace-delimited part) scores parseability, tool-name correctness, argument completeness and value agreement.

## Key results
- **Qwen2.5-1.5B:** 75% (greedy) and 79% (sampling).
- **Qwen2.5-0.5B:** 72% (greedy), but only 32% with sampling.
- **Others:** Phi-1.5 scores 0%; Pythia and TinyLlama reach at most 7%.
- **Strict audit:** only 5 of 1,000 raw responses were directly parseable JSON, so results depend almost entirely on output recovery.
- **Resources:** Qwen2.5-1.5B needs about 8 GB of RAM and about 31 s per call on CPU.

## Limitations
- Only 5 tools and 100 synthetic prompts, with no catalog scaling and no retrieval.
- Older base models.
- The paper itself says it does not cover the named boards or a full MCP implementation.

## Relation to our paper
- It shares our model (Qwen2.5-1.5B) and our scoring dimensions, so it confirms there is interest in local SLM tool reliability.
- It also shows that **parser leniency changes conclusions**. We must fix and report our parsing policy: native Ollama tool calls vs a recovery parser, and count "malformed" separately.
- Our study goes beyond it with catalog scaling, a public benchmark, a GPU with quantization, and retrieval.
