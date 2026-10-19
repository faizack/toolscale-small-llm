# AgentWeave: Routing Before Reasoning for Efficient Function Calling in Tool-Rich Language Models

- **Key:** `singla2026agentweave` · arXiv:2608.23078 (Aug 2026) · Singla, Singla, Gupta, Gupta · preprint, open-source artifacts
- **Read:** pass 1 (abstract), 2026-10-18
- **Relevance:** 3/3 (RQ3 with a 1.5B model on BFCL)

## Problem
As the candidate action space grows, a function-calling model has to read more schemas and tell apart more similar or irrelevant alternatives. Can the candidate set be reduced *before* inference, without changing the model?

## Method
AgentWeave is a deterministic pre-inference routing layer. It builds a bounded, model-visible action space from eligibility, requirement, capability and routing signals.

## Experiments
- **Protocol:** a frozen BFCL-derived "routing-pressure" protocol that pads the candidate tools.
- **Data:** 48 fresh BFCL v4 multiple-function tasks.
- **Model:** **Hammer2.1-1.5B**, run locally.
- **Baselines:** all tools, deterministic random top-8, and **semantic top-8**.

## Key results
- AgentWeave solves 6/48 tasks (12.5%). **All three baselines, including semantic top-8 retrieval, solve 0/48.**
- The paired difference is +12.5 pp (bootstrap 95% CI +4.2 to +22.9; McNemar p=0.031).
- It shows 70% fewer tools, uses 62% fewer input tokens, and has 51% lower latency.

## Limitations
- The authors call the result deliberately narrow: one 1.5B model, 48 tasks, very low absolute success, and not an official BFCL score.

## Relation to our paper
This is the closest work that combines **small model + BFCL + padded distractors + semantic top-k**. Its 0/48 for semantic top-8 with a 1.5B model is a warning sign: our absolute accuracies may be low, and retrieval alone may not rescue small models.

Our study is broader: four models, 400 queries, an N sweep, a k sweep, two embedders, and a failure taxonomy. It can show *where* retrieval helps. Cite it as concurrent, narrow evidence.
