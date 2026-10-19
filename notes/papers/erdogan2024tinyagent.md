# TinyAgent: Function Calling at the Edge

- **Key:** `erdogan2024tinyagent` · EMNLP 2024 System Demonstrations (2024.emnlp-demo.9) · Erdogan, Lee, Jha, Kim, et al. (UC Berkeley)
- **Read:** pass 2 (arXiv HTML), 2026-10-18
- **Relevance:** 3/3 (small quantized models + tool retrieval, on a laptop)

## Problem
Can small, locally run models do reliable function calling for a personal-assistant agent, without the cloud?

## Method
- Fine-tune TinyLlama-1.1B and Wizard-2-7B with LoRA on about 80K synthetic function-calling plans.
- Add **Tool RAG**: a fine-tuned DeBERTa-v3-small multi-label classifier picks the tools to include (probability above 0.5).
- Quantize to 4-bit with llama.cpp.
- Run on a MacBook Pro M3.

## Experiments
- 16 macOS tools.
- **Metric:** success rate on the authors' own task set, plus tokens and latency.
- **Retrieval variants compared:** all tools, basic embedding RAG, and the classifier Tool RAG.

## Key results
| Setting | Prompt tokens | 1.1B success | 7B success |
|---|---|---|---|
| All tools | ~2,762 | 78.89% | 83.09% |
| Basic **embedding** RAG | ~1,674 | 74.88% (worse) | 78.50% (worse) |
| Classifier Tool RAG (~3.97 tools, recall 0.998) | ~1,397 | 80.06% | 84.95% |

- 4-bit quantization kept or slightly improved success. The 1.1B model went from 2.2 GB to 0.68 GB.

## Limitations
- Only 16 tools in one domain, and the models are fine-tuned specifically for it.
- The text and table disagree on the embedding-RAG k.

## Relation to our paper
- Important counter-evidence for RQ3: **plain embedding retrieval reduced accuracy** for small models, because retrieval misses cost more than the context it saves. Our paper should measure retrieval recall@k and attribute failures to "gold not retrieved" vs "model error given gold".
- Unlike TinyAgent, we use **off-the-shelf** instruct models (no task fine-tuning), larger and controlled catalogs (up to 64), and a public benchmark.
