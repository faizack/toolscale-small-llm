# Compression Breaks Discipline: Quantization Increases Missed Tool Calls in Small Language Models, but Not Mid-Size Ones

- **Key:** `roshaan2026compression` · NeurIPS 2026 Workshop "SLM-Agents: 1st Workshop on SLMs for Agentic Systems" (poster) · Muhammad Roshaan, Muhammad Aaliyan, Saifullah Bin Zubair
- **Source:** https://neurips.cc/virtual/2026/171153. No arXiv version was found as of 2026-10-18.
- **Read:** pass 1 (workshop abstract), 2026-10-18
- **Relevance:** 3/3 (RQ2 taxonomy on small models; optional RQ4 quantization)

## Problem
Aggregate accuracy may hide *which* failure types grow when small tool-calling models are quantized to 4 bits.

## Method
- A **seven-category error taxonomy**, applied to 600 controlled generations.
- **Models:** Qwen3-1.7B and Qwen3-4B, each at FP16 and NF4.
- **Data:** BFCL single-turn tasks.
- **Analysis:** a paired interaction test for the effect of model size.

## Key results
- **Qwen3-1.7B:** NF4 raises **missed required tool calls** from 6.0% to 17.3%, and the rise appears in every task category.
- **Qwen3-4B:** no change in misses (7.3% in both settings), with slightly higher accuracy.
- The authors' conclusion: quantization damages tool-calling "discipline" rather than knowledge, so 4-bit safety must be validated for each checkpoint.
- The taxonomy classifier, protocol, labels and code are released.

## Limitations
- One model family and one quantization scheme (bitsandbytes NF4, not GGUF).
- Single-turn BFCL with the original small tool lists, so no catalog-size variable and no retrieval.
- Workshop paper.

## Relation to our paper
- This is a **direct overlap for RQ2's "taxonomy on sub-4B models"** and for any quantization RQ. It shows a taxonomy over BFCL failures of small models is already being done.
- Our differentiation is that we study how the taxonomy mix **shifts with catalog size and with retrieval**, not with precision.
- Use their released classifier or categories for comparability, and cite them for "no call" as a category that matters for small models.
- If we keep the optional Q4 vs Q8 RQ, we must cite this paper together with `singh2026samequant` and `ponrajan2026permission`, and present our result as replication on GGUF and Ollama.
