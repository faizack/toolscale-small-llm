"""Embedding-based top-k tool retrieval (RQ3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ROOT, safe_name
from .llm import embed

CACHE = ROOT / "data" / "processed" / "embeddings"

# Qwen3-Embedding expects an instruction on the query side only (model card).
QUERY_TEMPLATES = {
    "qwen3-embedding:0.6b": "Instruct: Given a user request, retrieve the API function that can fulfill it\nQuery: {q}",
    "bge-m3:latest": "{q}",
}


def tool_text(func: dict) -> str:
    params = func.get("parameters", {}).get("properties", {})
    ps = "; ".join(f"{k}: {v.get('description', '')}" for k, v in params.items())
    return f"{safe_name(func['name'])}: {func.get('description', '')} Parameters: {ps}"


def _cached_embed(model: str, texts: list[str], tag: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(json.dumps([model, texts]).encode()).hexdigest()[:12]
    path = CACHE / f"{tag}_{model.replace(':', '_')}_{h}.npy"
    if path.exists():
        return np.load(path)
    v = np.asarray(embed(model, texts), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.save(path, v)
    return v


class ToolRetriever:
    def __init__(self, model: str, pool: dict[str, dict], questions: dict[str, str]):
        self.model = model
        self.names = sorted(pool)
        self.tool_vecs = _cached_embed(model, [tool_text(pool[n]) for n in self.names], "tools")
        tmpl = QUERY_TEMPLATES.get(model, "{q}")
        self.qids = sorted(questions)
        self.q_vecs = _cached_embed(model, [tmpl.format(q=questions[i]) for i in self.qids], "queries")
        self._ni = {n: i for i, n in enumerate(self.names)}
        self._qi = {q: i for i, q in enumerate(self.qids)}

    def top_k(self, qid: str, catalog: list[str], k: int) -> list[str]:
        """Top-k catalog tools by cosine similarity, most similar first."""
        idx = np.array([self._ni[n] for n in catalog])
        sims = self.tool_vecs[idx] @ self.q_vecs[self._qi[qid]]
        order = np.argsort(-sims, kind="stable")[:k]
        return [catalog[i] for i in order]
