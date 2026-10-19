# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models (ToolRet)

- **Key:** `shi2025toolret` · Findings of ACL 2025 (2025.findings-acl.1258) · Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, et al.
- **Read:** pass 2 (arXiv HTML, first ~2/3), 2026-10-18
- **Relevance:** 3/3 (RQ3 retriever choice and the downstream cost of retrieval)

## Problem
Tool-use agents rely on retrievers to choose tools from large corpora, but nobody has benchmarked how well general information-retrieval models do at *tool* retrieval.

## Method
- **ToolRet benchmark:** 7,615 retrieval tasks over a corpus of 43,215 tools (web APIs, code functions, custom apps), with and without task instructions.
- **ToolRet-train:** more than 200K training instances.

## Experiments
- **Retrievers evaluated:** BM25; dense single-task retrievers (GTR, Contriever, ColBERTv2, COLT); multi-task embedders (MiniLM, GTE, BGE, E5, GTE-Qwen2-1.5B, E5-Mistral-7B, NV-Embed-v1); cross-encoder and LLM rerankers.
- **BGE-M3 and Qwen3-Embedding were *not* evaluated.**
- **Downstream:** pass rate on ToolBench using retrieved toolsets.

## Key results
- The best embedder averages only about 34 nDCG@10 without instructions and about 46–48 with instructions (GTE-Qwen2-1.5B: 45.96).
- Instructions help, while rerankers often hurt.
- **Retrieved toolsets lower end-to-end pass rate.** For example, GPT-3.5 on ToolBench-G1 drops 11.4 points with BGE-large retrieval.
- Training on ToolRet-train improves retrieval and downstream pass rates.

## Limitations
- Downstream models are large (GPT-3.5, ToolLlama), with no small models.
- Retrieval is evaluated on a huge corpus, unlike the small catalogs we use (≤64 tools).

## Relation to our paper
- This justifies RQ3's focus on **retrieval misses**: even good embedders miss tools, and a miss caps downstream accuracy.
- Our setting (pooled BFCL functions, N ≤ 64) is far easier for retrieval, so we should report recall@k to show whether retrieval or the generator is the bottleneck.
- Our embedders (bge-m3 and qwen3-embedding-0.6b) are newer than or absent from ToolRet, which is a small extra contribution.
- Consider adding an instruction prefix for the query, since instructions helped in ToolRet.
