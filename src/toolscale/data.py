"""Load BFCL data and build tool catalogs of controlled size.

Design (see notes/lab_notebook.md, 2026-10-18 "Experiment design"):
  * Each BFCL `simple_python` query has exactly one gold function.
  * A *catalog* of size N is the toolbox an agent sees. For N > 1 we shuffle the
    queries (seeded), split them into blocks of N, and use the gold functions of
    a block as that block's catalog. Every query is therefore evaluated once per
    (N, seed), and its distractors are the real gold tools of other queries.
  * If a block has fewer than N distinct functions (duplicate names or the last
    short block), it is padded with random functions from the rest of the pool.
  * The catalog order is shuffled (seeded) and the gold position is recorded.
  * All queries in one block share the same catalog -> identical prompt prefix ->
    the Ollama KV cache can be reused, which makes N=64 feasible on a 4 GB GPU.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# BFCL's GORILLA_TO_OPENAPI type mapping (bfcl_eval/constants/type_mappings.py @ f7cf735)
GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number", "string": "string",
    "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array",
    "dict": "object", "object": "object", "tuple": "array", "any": "string",
    "byte": "integer", "short": "integer", "long": "integer", "double": "number",
    "char": "string", "ArrayList": "array", "Array": "array", "HashMap": "object",
}
PYTHON_HINT = " Note that the provided function is in Python 3 syntax."


def safe_name(name: str) -> str:
    """BFCL maps '.' -> '_' for OpenAI-style function-calling models."""
    return re.sub(r"\.", "_", name)


def _cast_to_openai_type(properties: dict) -> dict:
    """Port of bfcl_eval.model_handler.utils._cast_to_openai_type (GORILLA_TO_OPENAPI)."""
    for key, value in properties.items():
        if "type" not in value:
            value["type"] = "string"
        else:
            t = value["type"]
            if t == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
            value["type"] = GORILLA_TO_OPENAPI.get(t, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = _cast_to_openai_type(value["properties"])
            elif "items" in value:
                it = value["items"]
                it["type"] = GORILLA_TO_OPENAPI[it["type"]]
                if it["type"] == "array" and "items" in it:
                    it["items"]["type"] = GORILLA_TO_OPENAPI[it["items"]["type"]]
                elif it["type"] == "object" and "properties" in it:
                    it["properties"] = _cast_to_openai_type(it["properties"])
    return properties


def to_tool(func: dict) -> dict:
    """BFCL function doc -> OpenAI/Ollama tool schema (mirrors BFCL FC-mode preprocessing)."""
    f = copy.deepcopy(func)
    f["description"] = f.get("description", "") + PYTHON_HINT
    f["name"] = safe_name(f["name"])
    f["parameters"]["type"] = "object"
    f["parameters"]["properties"] = _cast_to_openai_type(f["parameters"]["properties"])
    return {"type": "function", "function": f}


def checker_description(func: dict) -> dict:
    """Function description for the BFCL checker: original types, model-visible name."""
    f = copy.deepcopy(func)
    f["name"] = safe_name(f["name"])
    return f


@dataclass
class Query:
    id: str
    question: str
    func: dict                      # original BFCL function doc (gold)
    answer: dict                    # {safe_name: {param: [acceptable values]}}

    @property
    def gold_name(self) -> str:
        return safe_name(self.func["name"])


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_queries(split: str = "simple_python") -> list[Query]:
    items = _read_jsonl(RAW / f"BFCL_v4_{split}.json")
    answers = {a["id"]: a for a in _read_jsonl(RAW / f"possible_answer_BFCL_v4_{split}.json")}
    out = []
    for it in items:
        assert len(it["function"]) == 1 and len(it["question"]) == 1, it["id"]
        turn = it["question"][0]
        user = [m for m in turn if m["role"] == "user"]
        assert len(user) == 1, it["id"]
        gt = answers[it["id"]]["ground_truth"]
        assert len(gt) == 1, it["id"]
        (gname, params), = gt[0].items()
        out.append(Query(it["id"], user[0]["content"], it["function"][0], {safe_name(gname): params}))
    return out


def function_pool(queries: list[Query]) -> dict[str, dict]:
    """Unique functions by model-visible name (first occurrence wins)."""
    pool: dict[str, dict] = {}
    for q in queries:
        pool.setdefault(q.gold_name, q.func)
    return pool


@dataclass
class Catalog:
    catalog_id: str
    names: list[str]                       # model-visible names, in prompt order
    query_ids: list[str] = field(default_factory=list)


def build_catalogs(queries: list[Query], n: int, seed: int) -> tuple[list[Catalog], dict[str, str]]:
    """Return catalogs and a map query_id -> catalog_id."""
    pool = function_pool(queries)
    rng = random.Random(f"catalog-{n}-{seed}")
    if n == 1:
        cats = [Catalog(f"N1-{q.id}", [q.gold_name], [q.id]) for q in queries]
        return cats, {q.id: c.catalog_id for q, c in zip(queries, cats)}
    order = list(queries)
    rng.shuffle(order)
    cats, q2c = [], {}
    all_names = sorted(pool)
    for b in range(0, len(order), n):
        block = order[b:b + n]
        names = list(dict.fromkeys(q.gold_name for q in block))
        if len(names) < n:
            rest = [x for x in all_names if x not in set(names)]
            names += rng.sample(rest, n - len(names))
        rng.shuffle(names)
        cid = f"N{n}-s{seed}-b{b // n:03d}"
        cats.append(Catalog(cid, names, [q.id for q in block]))
        for q in block:
            q2c[q.id] = cid
    return cats, q2c


def eval_subset(queries: list[Query], size: int | None, seed: int = 0) -> set[str]:
    """Fixed random subset of query ids that is evaluated in *every* condition.

    Catalogs are still built from all queries, so distractors stay realistic;
    only the evaluated queries are subsampled (small-compute "learning" version).
    """
    ids = sorted(q.id for q in queries)
    if not size or size >= len(ids):
        return set(ids)
    return set(random.Random(f"subsample-{seed}").sample(ids, size))
