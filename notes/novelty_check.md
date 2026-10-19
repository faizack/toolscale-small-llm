# Novelty Check: How Many Tools Can a Small Language Model Handle?

*Date: 2026-10-18. Based on 57 verified references; the search protocol is in `notes/protocol.md`. All claims about other papers come from their abstracts or full text as noted in `notes/papers/`.*

## TL;DR verdict

**The general idea is already well covered. The specific slice we planned is still open, but it is narrow, so we have to position it carefully.**

These points are already established, and we cannot claim them as new:

- Tool-selection accuracy falls as the catalog grows: `kate2025longfunceval`, `gan2025ragmcp`, `rabinovich2025robustness`, `babu2026toolchoiceconfusion`, `liu2026toolscope`, `antonioni2025jsplit`.
- Gold-tool position matters: `kate2025longfunceval`, `gan2025ragmcp`, `blankenstein2026biasbusters`, `huang2024stability`.
- Retrieving a top-k shortlist reduces tokens and often helps: `gan2025ragmcp`, `repantis2026howmany`, `mudunuri2026semantic`, `patel2026dtdr`, `fei2025mcpzero`.

What remains open is a **controlled, reproducible, full-factorial study of sub-4B quantized instruct models on consumer hardware**. It would cross catalog size (small-N regime) × distractor similarity × retrieval (top-k, two embedders) × gold position, and report a **labelled failure taxonomy that shows how the error mix shifts along those axes**. No paper we found does more than two of these together for sub-4B models.

The paper is publishable as a careful empirical, workshop- or journal-level contribution, not as "the first to show that more tools hurt". Concurrent 2026 work (AgentWeave, the NeurIPS'26 SLM-Agents workshop papers) shows this niche is filling up quickly. **Speed matters**: aim for arXiv within about 3 months.

---

## 1. Closest prior works, ranked by overlap

| # | Paper | Overlap with us | What they did NOT do (our gap) |
|---|---|---|---|
| 1 | `kate2025longfunceval` (LongFuncEval, 2025) | **RQ1 + position.** BFCL simple/multiple, random distractors from the BFCL function pool, gold position α ∈ {0.1…0.9}, 7–85% drop. Error types: format, wrong function, wrong params, hallucinated names. | Models ≥8B only, full precision. Catalogs ≥ ~49 tools (8K–120K tokens). Random distractors only. No retrieval. No quantified taxonomy. |
| 2 | `gan2025ragmcp` (RAG-MCP, 2025) | **RQ1 + RQ3 + position.** MCP stress test of accuracy vs #servers and gold position; retrieval triples accuracy (13.6% → 43.1%). | One large proprietary model (qwen-max), 20 tasks, unspecified retriever and k, no small models, no error taxonomy. |
| 3 | `singla2026agentweave` (AgentWeave, Aug 2026) | **RQ3 with a 1.5B model.** BFCL v4 multiple tasks padded with distractors; all-tools vs random top-8 vs **semantic top-8** on Hammer2.1-1.5B, run locally. | One model, 48 tasks, no N sweep, no k sweep, no taxonomy. Semantic top-8 scored 0/48, which is a warning for us. |
| 4 | `rabinovich2025robustness` (TrustNLP 2025) | **RQ1 (semantic distractors) + RQ2.** BFCL subset with about 3 semantically related distractors; breakdown of wrong function / wrong params / wrong count / syntax. | Large models only (8B–72B and APIs), tiny catalogs (~6 tools), no scaling curve, no retrieval. |
| 5 | `roshaan2026compression` + `singh2026samequant` (NeurIPS'26 SLM-Agents ws) | **RQ2 on sub-4B models.** Error taxonomies (7 categories) for Qwen3-1.7B/4B and six SLMs at GGUF F16→Q4_K_M on BFCL; quantization raises "missed call" errors at 1.7B. | The variable is precision, not catalog size. No retrieval. No distractors. |

Also close, in a supporting role:

- `erdogan2024tinyagent`: 1.1B/7B 4-bit models with Tool RAG on a laptop. Plain embedding RAG *hurt* compared with all tools, but there were only 16 tools and the models were fine-tuned.
- `kavathekar2025small` and `mastoi2026beyond`: SLM function-calling evaluations with tool / argument / value metrics and Q4_K_M or CPU, but no catalog scaling.
- `repantis2026howmany`: the principled choice of k (chance-corrected); BFCL 370-tool registry.
- `lee2026patool`: SLMs hallucinate tool names (schema misalignment); this supports our "hallucinated tool" category.
- `shi2025toolret`: embedders are weak tool retrievers, and retrieval misses lower end-to-end success.
- `choi2026toolmath`: distractors graded by similarity to the gold tools, plus trace-level failure analysis (math domain, long-horizon).
- `blankenstein2026biasbusters` and `li2026metadata`: positional bias in tool selection; position-balanced designs.

## 2. What is already known (do not claim as new)

1. **More tools → lower accuracy** for large models, with strong differences between model families. Long catalogs bring format errors and hallucinated function names (`kate2025longfunceval`). Semantically similar tools hurt even in small numbers (`rabinovich2025robustness`, `liu2026toolscope`, `choi2026toolmath`).
2. **Position of the correct tool matters.** Recency or lost-in-the-middle patterns depend on the model (`kate2025longfunceval`, `liu2024lost`). Earlier-listed tools are favoured among equivalent tools (`blankenstein2026biasbusters`). Shuffling the order changes success (`huang2024stability`).
3. **Retrieval / shortlisting reduces tokens and usually improves selection for large models** (`gan2025ragmcp`, `mudunuri2026semantic`, `antonioni2025jsplit`, `patel2026dtdr`). Fixed k is suboptimal (`repantis2026howmany`). Retrievers are imperfect on tools (`shi2025toolret`).
4. **Small models are weak at zero-shot tool calling**, mainly through format and discipline failures (`kavathekar2025small`, `mastoi2026beyond`, `haque2025tinyllm`). Fine-tuned SLMs can be strong (`erdogan2024tinyagent`, `lin2024hammer`, `zhang2025xlam`).
5. **4-bit quantization changes the failure mix of SLM tool calling** even when aggregate accuracy is flat (`roshaan2026compression`, `singh2026samequant`, `ponrajan2026permission`, `jang2026flatscore`). Conclusions depend on the scoring policy and the prompt (`hu2026quantrecovery`).

## 3. What is NOT known (our real gap)

- **The accuracy-vs-N curve in the small-N regime (1→64) for sub-4B instruct models.** LongFuncEval starts at about 49 tools with ≥8B models, and Rabinovich stops at about 6 tools. Where is the "knee" for a 1.5B–3B model? Is it already at N=4–16?
- **Random vs semantically similar distractors at matched N for small models.** Nobody separates "more tools" from "more confusable tools" for SLMs.
- **How the failure mix shifts with N and with retrieval.** For example: does "hallucinated tool" grow with N? Does retrieval turn "wrong tool" errors into "no call" errors or "gold not retrieved" errors? Existing taxonomies vary precision (`roshaan2026compression`) or are qualitative (`kate2025longfunceval`).
- **Whether embedding top-k actually helps sub-4B off-the-shelf models.** The evidence conflicts: TinyAgent found embedding RAG hurt, AgentWeave found semantic top-8 got 0/48, and RAG-MCP found it helps a large model. A careful decomposition is missing: retrieval recall@k × accuracy given the gold tool is retrieved.
- **Position effects in short catalogs for small quantized models.** Existing position studies use long contexts or large models.
- **A reproducible consumer-hardware protocol** (Ollama, GGUF, 4 GB GPU) with repeated runs and confidence intervals. Most prior SLM papers report a single run.

## 4. Recommended positioning

> *"A controlled, reproducible study on consumer hardware (4 GB GPU, Ollama/GGUF) of how sub-4B quantized instruct models degrade as the tool catalog grows from 1 to 64, separating catalog size from distractor similarity and gold-tool position. We contribute a labelled failure taxonomy that shows how the error mix changes with catalog size and with embedding retrieval, and we decompose retrieval-based mitigation into retrieval recall and conditional selection accuracy."*

Framing tips:

- **Do not** say "we are the first to show that accuracy drops with more tools". Instead say: "prior work documented this for ≥8B models at hundreds of tools \citep{kate2025longfunceval} or with a handful of related tools \citep{rabinovich2025robustness}. We characterize the small-N regime for sub-4B models, where the drop begins much earlier."
- Make **RQ2 (the shift in the failure mix)** and **the RQ3 decomposition** the headline contributions. They are the least covered.
- Present the work as practical guidance: "for a 3B model on a laptop, keep at most X tools in context, or use top-k with k=Y."
- Cite the concurrent 2026 SLM papers (AgentWeave, the SLM-Agents workshop papers) openly as complementary work.

## 5. Suggested RQ adjustments

1. **RQ1 → split into RQ1a/RQ1b.**
   - RQ1a: accuracy vs N ∈ {1, 4, 16, 64} with **random** distractors.
   - RQ1b: the same N with **similar** distractors (top-similarity by embedding, as in `rabinovich2025robustness` and `choi2026toolmath`).
   - The comparison between them is new. Consider adding N=8 and N=32 to find the knee; small N is cheap.
2. **Position as its own factor (RQ1c).** Place the gold tool at first / middle / last (or α ∈ {0, 0.5, 1}) with a balanced design, and analyse it with a mixed or conditional logistic model (as in `li2026metadata`). Reuse LongFuncEval's α convention for comparability.
3. **RQ2: an explicit "shift" question.** "How does the distribution of failure types change with N, distractor type, and retrieval?"
   - Keep the planned categories. Add "gold not retrieved" for RQ3 conditions, and separate "no call" from "malformed".
   - Report inter-annotator agreement, or use a deterministic classifier with a manually checked subset.
   - Align category names with `roshaan2026compression` / `singh2026samequant` where possible.
4. **RQ3: decompose.** Report three quantities:
   - (i) retrieval recall@k for bge-m3 and qwen3-embedding-0.6b
   - (ii) model accuracy *given* the gold tool is in the top-k
   - (iii) end-to-end accuracy against the all-tools baseline at the same N

   Sweep k ∈ {1, 3, 5, 10} and add the chance-corrected view (`repantis2026howmany`). Report latency and tokens too.
5. **Optional RQ4 (Q4 vs Q8): deprioritize, or present it as replication.** It is now crowded (three NeurIPS'26 workshop papers plus `jang2026flatscore` and `dong2025acbench`). Keep it only as a robustness check on one model.
6. **Rigor that sets us apart from the closest works:**
   - 400 queries × ≥3 seeds or permutations
   - bootstrap confidence intervals and McNemar tests (AgentWeave and singh2026 set this bar)
   - a fixed and reported parsing policy (lenient vs strict; see `mastoi2026beyond`, `hu2026quantrecovery`)
   - released code and raw outputs

## 6. Risks

- **Low absolute accuracy at N=64 for 1.5B models** (AgentWeave got 0/48 with semantic top-8). Report the format-failure rate and accuracy conditional on a valid format, so that the whole curve is not just format errors.
- **BFCL `simple_python` pooled distractors may be too easy to retrieve** (topically distinct functions). The "similar distractor" condition is essential, otherwise RQ3 is trivial.
- **Qwen3.5-2B has no technical report** (only Qwen3.5-Omni exists on arXiv). Cite the official model card or blog, and the Qwen3 report for architecture lineage. Llama-3.2-3B likewise: cite `grattafiori2024llama3` plus the model card.
- **Some closest works are 2026 preprints or workshop papers.** Check them again for updates before submission.
