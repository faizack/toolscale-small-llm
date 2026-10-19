# LongFuncEval: Measuring the effectiveness of long context models for function calling

- **Key:** `kate2025longfunceval` · arXiv:2505.10570 (April 2025) · Kate, Pedapati, Basu, Rizk, et al. (IBM) · preprint
- **Read:** pass 2 (abstract + HTML experimental section), 2026-10-18
- **Relevance:** 3/3. **The closest prior work to our RQ1.**

## Problem
How do long-context LLMs handle function calling when the context is long? The paper looks at three sources of length: (1) a large tool catalog, (2) long tool responses, and (3) long multi-turn conversations.

## Method
For challenge (1), the authors take BFCL `simple`, `live_simple` and `multiple` (858 samples). They build a pool of unique BFCL functions, deduplicated by name, and pad each query's tool list with **randomly sampled irrelevant functions** until a token budget is reached. The gold function goes in at a relative position α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.

## Experiments
- **Context budgets:** 8K, 16K, 32K, 64K and 120K tokens, which works out to an average of about 49, 102, 207, 417 and 741 tools.
- **Models:** Mistral-Large-2411, ToolACE-8B, BitAgent-8B, Granite-3.1-8B, Llama-3.1-8B/70B, DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, and GPT-4o (a subset of settings only). **The smallest model is 8B. Nothing under 4B and no quantized models.**
- **Metric:** BFCL AST accuracy, using the BFCL prompts and checker.

## Key results
- Accuracy drops by 7% to 85% as the number of tools grows. The size of the drop varies strongly by model family.
- Position effects get stronger with context length. Several models do better when the gold tool comes late (recency). Granite shows a lost-in-the-middle U-shape. GPT-4o barely degrades.
- **Error types:** invalid format (AST errors), wrong function, and right function with wrong parameters. At very long contexts, models hallucinate function names (for example `prime_factors` instead of `get_prime_factors`).

## Limitations
- Only large models (≥8B) at full precision, and no retrieval mitigation.
- Distractors are random only, with no control over semantic similarity.
- The smallest catalog is about 49 tools, so it does not cover the 1–16 range that matters for small models.
- The error analysis is qualitative, not a labelled taxonomy.
- The fetched summary noted that some tables disagree with each other.

## Relation to our paper
This paper **pre-empts the basic RQ1 design**: BFCL simple, random distractors from other BFCL functions, and gold-position sweeps. We must cite it prominently and say how we differ:

- sub-4B quantized models on consumer hardware
- the small-N regime (1–64), where small models break
- random **vs** semantically similar distractors
- a quantified failure taxonomy (RQ2)
- embedding retrieval as mitigation (RQ3)

Consider reusing their α-position protocol so the results are directly comparable.
