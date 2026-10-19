# Small Models, Big Tasks: An Exploratory Empirical Study on Small Language Models for Function Calling

- **Key:** `kavathekar2025small` · EASE 2025 (AI Models and Data Evaluation track) · Kavathekar, Donakanti, Kumaraguru, Vaidhyanathan (IIIT Hyderabad)
- **Read:** pass 2 (arXiv HTML), 2026-10-18
- **Relevance:** 3/3 (small models, error categories, edge quantization)

## Problem
Can small language models generate correct function calls across domains? How do zero-shot, few-shot and fine-tuned settings compare, and how robust are SLMs to prompt injection and to running on an edge device?

## Method
- **Models:** Deepseek-coder-1.3B-instruct, Phi-3-mini-4k (3.8B), Phi-2 (2.8B), Starcoder2-3B and Stable-code-3B.
- **Data:** the Salesforce xLAM function-calling dataset (60K examples derived from ToolBench): 5K for test, 55K for fine-tuning.
- **Metrics:**
  - JSON parsability
  - task accuracy (F1)
  - correct ratio
  - function selection performance (FSP)
  - argument completeness (ACS)
  - argument value correctness (AVC)

## Experiments
- Zero-shot, few-shot and LoRA fine-tuning, each with and without prompt injection.
- An edge run on a Snapdragon 8 Gen 2 (QIDK) with **GGUF Q4_K_M** models on 100 samples.

## Key results
- **Zero-shot:** essentially total failure. Only Deepseek-coder-1.3B produced any parsable JSON (7.3%). All other models scored 0.
- Few-shot helps, and fine-tuning helps most.
- Manual inspection of 200 failures found three patterns: format non-adherence, uncontrolled generation (a correct answer followed by an invented new query), and malformed JSON.
- Prompt injection causes only a slight decline.

## Limitations
- These are mostly code or base-style models, not modern instruct models with native tool templates.
- The number of tools per query is not controlled.
- No catalog-size or retrieval analysis.

## Relation to our paper
- Their metrics (FSP, ACS, AVC, parsability) map almost one-to-one onto our taxonomy (wrong tool, missing or extra args, wrong values, malformed), so cite them for the metric design.
- We differ in three ways: modern instruct models with native tool calling via Ollama, controlled catalog size, and retrieval.
- Their near-zero zero-shot results warn us to report the format-failure rate separately so that format failures do not dominate RQ1.
