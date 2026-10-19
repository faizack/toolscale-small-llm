# Literature Search Protocol

Paper (working title): *How Many Tools Can a Small Language Model Handle? Scaling and Failure Analysis of Tool Selection in Small LLM Agents*

- Protocol written and searches run: **2026-10-18**
- Reviewer: one person, with an AI research assistant. All hits were screened by title and abstract. The papers kept for close reading also had their full text (HTML) checked.
- Outputs: `paper/references.bib` (57 verified entries), `notes/paper_tracker.csv`, `notes/papers/*.md`, `notes/novelty_check.md`, `notes/related_work_draft.md`

---

## 1. Review questions

These questions guide the review. They are separate from the paper's research questions.

| ID | Review question | Linked paper RQ |
|---|---|---|
| LQ1 | What is known about how function-calling accuracy changes as the number of tools in context grows (tool-catalog scaling, distractors, context length)? | RQ1 |
| LQ2 | Which error and failure taxonomies exist for function calling, and how do error types shift with catalog size, model size, or compression? | RQ2 |
| LQ3 | How well does embedding-based tool retrieval (top-k) mitigate large catalogs, and what does a retrieval miss cost downstream? | RQ3 |
| LQ4 | What is known about small (<4B) and quantized models as tool users, especially local or on-device ones? | all |
| LQ5 | Does the position of the gold tool in the prompt matter (positional or recency bias)? | RQ1 (position) |

## 2. Sources

| Source | Access method | Status on 2026-10-18 |
|---|---|---|
| arXiv | `export.arxiv.org/api/query` (search + `id_list` lookup) | Worked. Main search source and metadata ground truth |
| Semantic Scholar | Graph API `/paper/search` (keyword) and `/paper/batch` (by arXiv ID) | `/paper/search` returned HTTP 429 on every call, even with 4–30 s sleeps. One `/paper/batch` POST worked and was used to check venues and citation counts |
| DBLP | `dblp.org/search/publ/api` | **Blocked** by the Anubis bot-protection page, from both curl and WebFetch. DBLP BibTeX could not be used |
| ACL Anthology | Full `anthology.bib.gz` dump (12.7 MB), title matching | Worked. Gave the BibTeX for every ACL/EMNLP/NAACL/TACL/Findings/workshop paper |
| OpenReview / conference sites | neurips.cc, iclr.cc, proceedings.mlr.press via WebFetch/WebSearch | Worked. Used to verify venues (BFCL at ICML 2025 PMLR, Gorilla at NeurIPS 2024, ToolACE at ICLR 2025, three NeurIPS 2026 workshop papers) |
| General web search | WebSearch tool | Worked. Used for discovery (snowballing) only. Every hit was then verified on arXiv, the ACL Anthology or a conference page |

## 3. Search strings

**arXiv API (field-restricted to abstract or title):**

| ID | Query string |
|---|---|
| Q1 | `abs:"tool retrieval" AND (abs:LLM OR abs:"language model")` |
| Q2 | `(abs:"number of tools" OR abs:"large toolset" OR abs:"many tools" OR abs:"tool catalog") AND (abs:"function calling" OR abs:"tool selection" OR abs:"tool use")` |
| Q3 | `(abs:"small language model" OR abs:"small language models" OR abs:SLM OR abs:"on-device") AND (abs:"function calling" OR abs:"tool calling")` |
| Q4 | `(abs:"Model Context Protocol" OR abs:MCP) AND abs:benchmark AND abs:tool` |
| Q5 | `abs:quantization AND (abs:"tool calling" OR abs:"function calling" OR abs:agentic)` |
| Q6 | `(abs:"position bias" OR abs:positional OR abs:"tool order") AND (abs:"tool selection" OR abs:"function calling")` |
| Q7 | `ti:survey AND (ti:"tool learning" OR ti:"tool use" OR ti:"function calling")` |
| Q8 | `(abs:"error analysis" OR abs:"failure taxonomy" OR abs:"failure modes") AND (abs:"function calling" OR abs:"tool calling")` |

All queries were sorted by relevance, and the top 40 results of each were screened.

**Semantic Scholar keyword queries** (attempted, `year=2023-2026`, `limit=20`): "tool retrieval large language models", "tool selection large number of tools LLM", "function calling small language models", "MCP tool selection benchmark", "Model Context Protocol benchmark agents", "tool learning large language models survey", "scaling number of tools function calling accuracy", "quantization tool use LLM agents", "position bias tool selection LLM", "on-device function calling small model", "tool calling error analysis taxonomy LLM", "function calling benchmark leaderboard".

**Web searches (discovery and snowballing):** listed in the log below.

**Backward snowballing:** from the related-work sections of LongFuncEval, *How Many Tools Should an LLM Agent See?*, and the NeurIPS 2026 SLM-Agents workshop listings.

## 4. Date range

- Main window: **2023-01-01 to 2026-10-18**. The novelty check gives priority to 2024–2026 work.
- Exceptions for foundational work: ReAct (arXiv 2022, ICLR 2023) and Lost in the Middle (arXiv 2023, TACL 2024).

## 5. Inclusion and exclusion criteria

**Include** a paper if it meets at least one of these:

- **I1:** It proposes or evaluates LLM tool use or function calling (a method, a benchmark, or an empirical study).
- **I2:** It studies tool retrieval or selection over large catalogs, or the effects of tool count or distractors.
- **I3:** It evaluates small (roughly ≤8B, preferably ≤4B) or quantized or compressed models on tool use.
- **I4:** It is an MCP tool-use benchmark that reports tool-selection or failure analysis.
- **I5:** It is a technical report for a model or embedding model we use, or a foundational paper on long-context position bias.
- **I6:** It is a survey of tool learning or of small-model agents from 2024–2026.

**Exclude** a paper if any of these apply:

- **E1:** Its tools are physical or robotic, or "tool" is used in a non-LLM sense.
- **E2:** It is about MCP security or attacks only, with no tool-selection measurement.
- **E3:** It is a domain application (finance, bio, remote sensing) with no general finding about tool selection.
- **E4:** It is RL or training-method work with no evaluation insight relevant to our RQs. Exception: well-known small-model tool-use models we may compare against.
- **E5:** It cannot be verified (title, authors and year must all match a primary source).
- **E6:** It is a blog post, a vendor page, or a non-archival preprint with no identifiable author.

## 6. Verification procedure

1. Get the title, author list and first-submission date for every arXiv ID from `export.arxiv.org/api/query?id_list=...` (63 IDs checked; all resolved).
2. Take the published venue from, in order of preference: the ACL Anthology BibTeX dump, then PMLR or conference pages, then the arXiv `comment` / `journal_ref`, then the Semantic Scholar batch API `venue` field. If none confirmed a venue, the entry stays `@misc` (arXiv).
3. Accept workshop papers without an arXiv version only when the official neurips.cc page gives the title, authors and workshop (3 papers).
4. Check `paper/references.bib` compiles with BibTeX + natbib (`plainnat`): 57 entries, 0 BibTeX warnings.

## 7. Search log

| # | Date | Source | Query (abbreviated; full strings above) | #hits | #kept |
|---|---|---|---|---|---|
| 1 | 2026-10-18 | Semantic Scholar /search | 12 keyword queries (Section 3) | 0 (HTTP 429 on all; script stopped after ~10 min) | 0 |
| 2 | 2026-10-18 | Semantic Scholar /search | 3-query retry with 30 s sleeps | 0 (HTTP 429) | 0 |
| 3 | 2026-10-18 | Semantic Scholar /paper/batch | 63 arXiv IDs (venue and citation lookup) | 63 resolved | used for verification |
| 4 | 2026-10-18 | DBLP publ API | 29 title queries | 0 (bot-protection block) | 0 |
| 5 | 2026-10-18 | arXiv API | Q1 tool retrieval | 65 | 4 (COLT, ToolGen, ToolRet, Re-Invoke) |
| 6 | 2026-10-18 | arXiv API | Q2 number of tools / catalog | 44 | 5 |
| 7 | 2026-10-18 | arXiv API | Q3 small LM + function calling | 138 | 5 |
| 8 | 2026-10-18 | arXiv API | Q4 MCP benchmark | 239 | 6 |
| 9 | 2026-10-18 | arXiv API | Q5 quantization + tool calling | 447 | 2 |
| 10 | 2026-10-18 | arXiv API | Q6 position bias + tool selection | 129 | 1 |
| 11 | 2026-10-18 | arXiv API | Q7 survey titles | 1 | 1 |
| 12 | 2026-10-18 | arXiv API | Q8 error analysis / failure modes | 93 | 1 |
| 13 | 2026-10-18 | Web search | "arXiv tool selection degradation number of tools small language models function calling" | 9 | 4 (LongFuncEval, Small Models Big Tasks, How Many Tools, ToolChoiceConfusion) |
| 14 | 2026-10-18 | Web search | "MCP tool overload many tools LLM accuracy drops benchmark arXiv 2025" | 9 | 2 (MCP-Bench, LiveMCPBench via snowball) |
| 15 | 2026-10-18 | Web search | "quantization effect on function calling tool use small LLM agents arXiv" | 9 | 6 (ACBench, Flat Score, quant recovery, 3 NeurIPS'26 workshop papers) |
| 16 | 2026-10-18 | Web search | "Rabinovich Anaby-Tavor function calling semantically similar tools robustness" | 10 | 2 (Robustness of Agentic FC, Beyond Fluent Generation) |
| 17 | 2026-10-18 | Web search | "tool retrieval benchmark ToolRet RAG-MCP Re-Invoke ToolGen COLT ..." | 9 | 2 (MCP-Zero; others already known) |
| 18 | 2026-10-18 | Web search | "position bias order of tools in prompt LLM tool selection study arXiv" | 9 | 2 (BiasBusters, Stability of tool learning) |
| 19 | 2026-10-18 | Web search | "\"Beyond Fluent Generation\" CPU reliability benchmark MCP ..." | 9 | 0 (confirmation only) |
| 20 | 2026-10-18 | Web search | "\"How Many Tools Should an LLM Agent See\" chance-corrected" | 9 | 0 (confirmation only) |
| 21 | 2026-10-18 | Web search | "tool learning survey 2025 large language models arXiv comprehensive" | 10 | 1 (Qu et al. survey) |
| 22 | 2026-10-18 | Web search | "\"Compression Breaks Discipline\" quantization missed tool calls ..." | 9 | 0 (confirmation) |
| 23 | 2026-10-18 | Web search | "small language models tool retrieval top-k tool catalog size scaling 1B 3B Ollama local ..." | 9 | 1 (PA-Tool / Don't Adapt SLMs) |
| 24 | 2026-10-18 | Web search | "Berkeley Function Calling Leaderboard BFCL paper ICML 2025 ..." | 9 | 1 (BFCL paper, PMLR) |
| 25 | 2026-10-18 | Web search | "Qwen3.5 technical report arXiv" | 9 | 0 (no general Qwen3.5 report exists; only Qwen3.5-Omni) |
| 26 | 2026-10-18 | Web search | Gorilla NeurIPS 2024; ToolACE ICLR 2025; Tool Learning w/ Foundation Models venue | 29 | 0 (venue verification only) |
| 27 | 2026-10-18 | Web search | "distractor functions BFCL small models 1B 3B accuracy vs number of candidate functions ..." | 10 | 1 (AgentWeave) |
| 28 | 2026-10-18 | ACL Anthology dump | 31 title look-ups | 15 matched | 14 used |
| 29 | 2026-10-18 | Known-item lookup (arXiv id_list) | foundational and tech-report papers (Toolformer, Gorilla, ToolLLM, API-Bank, ReAct, ToolACE, TinyAgent, Octopus v2, xLAM, APIGen, Hammer, Qwen2.5/3, Llama 3, BGE-M3, Qwen3-Embedding, ...) | 51 + 12 IDs resolved | ~25 |

**Totals.** About 1,156 arXiv API hits were reported, of which 281 were screened by title and abstract. With the web searches and known-item lookups added, about 350 records were screened, about 75 abstracts were read, and **57 verified references** were kept.

Some screened items were dropped as less relevant: τ-bench, AnyTool, Toolshed, MCPToolBench++, MCP-RADAR, MCPAgentBench, Tool-to-Agent Retrieval, *Tool Learning with Foundation Models*, the 350M OPT tool-calling paper, IFFC, REFLEX, and *The Bitter Lesson of Tool Calling*. All of them are verified and can be added back to the bib if needed (their metadata is in the review scratch files).

## 8. Threats to validity of the search

- Semantic Scholar keyword search and DBLP were both unavailable, so recall relies on the arXiv API plus web search. Re-run the S2 queries with an API key, and DBLP from a normal browser, before submission.
- Many of the closest works are 2026 preprints or workshop papers. Check them again for newer versions or venues just before submission.
- Title and abstract screening was done by a single reviewer, with no second screener.
