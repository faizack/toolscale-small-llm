# Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models (PA-Tool)

- **Key:** `lee2026patool` · ACL 2026 Main (2026.acl-long.948) · Jonggeun Lee, Woojung Song, Jongwook Han, Haesung Pyun, et al.
- **Read:** pass 1 (abstract), 2026-10-18
- **Relevance:** 3/3 (RQ2 "hallucinated tool" category in SLMs)

## Problem
Small language models struggle to choose tools and fill parameters. A common failure is **schema misalignment**: the model outputs plausible tool names that do not exist in the provided schema, because of naming conventions it learned in pretraining.

## Method
PA-Tool is a training-free method that renames tool components to match what the model already expects:

1. Generate several candidate names for each tool component.
2. Score each candidate by *peakedness*, a contamination-detection signal that indicates familiarity from pretraining.
3. Keep the most "familiar" candidate.

## Experiments
- MetaTool and RoTBench with small models.
- The exact model list was not read in this pass. Check it in pass 2.

## Key results
- Accuracy improves by up to 17%.
- **Schema-misalignment (hallucinated-name) errors fall by 80%.**

## Limitations
- Not yet read in detail. The catalog-size dependence of hallucinated names is not reported in the abstract.

## Relation to our paper
- This is strong prior evidence that **hallucinated tool names are a major SLM failure mode**, which justifies a separate "hallucinated tool" category in RQ2.
- Hypothesis to test: hallucinated-name errors grow with N, because similar names compete.
- Possible follow-up for paper #2: tool-name or description rewriting as a mitigation alongside retrieval.
