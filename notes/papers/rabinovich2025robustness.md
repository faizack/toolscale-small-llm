# On the Robustness of Agentic Function Calling

- **Key:** `rabinovich2025robustness` · TrustNLP Workshop @ NAACL 2025 (ACL Anthology 2025.trustnlp-main.20) · Ella Rabinovich, Ateret Anaby-Tavor (IBM)
- **Read:** pass 2 (arXiv HTML), 2026-10-18
- **Relevance:** 3/3 (RQ1 with similar distractors; RQ2 error types)

## Problem
Two robustness questions for function calling: (a) whether models handle naturalistic rephrasings of the query, and (b) whether tool selection stays stable when the toolkit grows with **semantically related** tools.

## Method
- **Data:** 200 examples from a single-turn BFCL subset.
- **Distractors:** Llama-3.1-70B generates related but distinct request variants, and CodeLlama-13B writes a matching tool definition for each. Tools that are functionally equivalent to the original are removed by embedding the tool "signature" and dropping candidates with cosine similarity above 0.8.
- **Toolkit size:** grows from 2.7 to 5.6 tools on average, i.e. about 3 distractors per case.

## Experiments
- **Models:** Llama-3.1-70B, Llama-3.3-70B, DeepSeek-V2.5, Qwen2.5-72B, Granite-3.1-8B, Claude-3.5-Haiku/Sonnet, GPT-4o-mini, o1-mini.
- **Metric:** BFCL AST.

## Key results
- Rephrasing the query hurts most, with drops of 8–19%.
- Toolkit expansion hurts less but consistently, with drops of 1–8%. The largest drop (8%) is for the smallest model, Granite-8B.
- **Errors caused by expansion:** wrong function and wrong parameter assignment dominate. Wrong number of calls and wrong syntax are rarer. No single pattern prevails across models.

## Limitations
- Large or proprietary models only.
- Small catalogs (about 6 tools), with no scaling curve.
- Only 200 examples, a single run, and no retrieval.

## Relation to our paper
This is the template for our **semantically similar distractor** condition. It also shows that even about 3 similar distractors hurt, with the biggest effect on the smallest model. That supports our hypothesis that sub-4B models degrade much faster.

Their 4-category error breakdown is a subset of our taxonomy (we add hallucinated tool, no call, missing or extra arguments, and malformed output). Cite it for the error categories, and contrast our scale: 1–64 tools, sub-4B models, 400 queries, repeated runs.
