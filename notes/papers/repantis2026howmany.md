# How Many Tools Should an LLM Agent See? A Chance-Corrected Answer

- **Key:** `repantis2026howmany` · arXiv:2605.24660 (May 2026) · Repantis, Gawde, Singh, Blackwell · preprint
- **Read:** pass 1 (abstract + secondary summary), 2026-10-18
- **Relevance:** 3/3 (RQ3: choosing top-k)

## Problem
A retriever must decide how many candidate tools to show the agent. Too many and the model gets confused; too few and the gold tool may be missing. Most systems use a fixed k, and there is no principled way to judge whether that k is right.

## Method
- Applies **Bits-over-Random (BoR)**, a chance-corrected metric that asks whether success at depth k beats random selection at the same depth.
- Turns BoR into an RL reward for choosing the shortlist depth per query. The authors describe the RL agent as a deliberately simple probe, not a proposed system.

## Experiments
- Three tool-selection benchmarks with registries of 20 to 3,251 tools, including BFCL (370 tools) and ToolBench (3,251 tools).
- Multiple scorers (retrievers).
- Downstream check with Claude Sonnet 4.6.

## Key results
- **BFCL:** the adaptive policy nearly matches the coverage of a fixed k=50 (90.3% vs 90.8%) while showing only about 7 tools on average.
- **ToolBench:** a fixed k=5 has higher aggregate coverage (64.7% vs 61.9%) but misses all hard queries, where the gold tool ranks 6th–20th. Adaptive depth recovers 16.7% of those.
- **Downstream:** shorter adaptive lists improve Sonnet's selection accuracy (93.1% vs 87.1% with a fixed k=5; 76.8% vs 60.9% on medium-difficulty queries).

## Limitations
- One frontier model downstream, and no small models.
- The RL agent is a toy probe.
- Single preprint with no independent replication.

## Relation to our paper
- This gives a principled way to report our top-k sweep: report recall@k next to a chance baseline, and think about adaptive k.
- It also shows the BFCL function pool (370 tools) is a standard registry, so our pooled catalogs from `simple_python` are a defensible choice.
- **Differentiator:** they assume a strong downstream model. We test whether small models gain more from a short list, or are more hurt when retrieval misses the gold tool.
