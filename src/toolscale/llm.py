"""Thin wrapper around the Ollama /api/chat endpoint with native tool calling.

We call the HTTP API directly (not the `ollama` Python client) so we can request
`logprobs`: the generated tokens come back even when Ollama's tool-call parser
silently discards a malformed call. Without this, a call like
`{"name": air_quality, ...}` (unquoted name -> invalid JSON) shows up as an
empty response and would be mis-classified as "no call" (lab notebook, 2026-10-18).
"""

from __future__ import annotations

import json
import os
import time

import httpx

HOST = "http://127.0.0.1:11434"
_HTTP = httpx.Client(timeout=180)  # slowest normal call ~30 s; stalls are retried (lab notebook Entry 16)


def chat_with_tools(model: str, question: str | list[dict], tools: list[dict], *, num_ctx: int,
                    seed: int = 0, think: bool | None = None, num_predict: int = 512) -> dict:
    """One request. `question` is a user string or a full message list (multi-turn / system
    prompts). Returns a JSON-serialisable record of everything we need, including the
    assistant message (`message`) so callers can append it to the chat history."""
    messages = [{"role": "user", "content": question}] if isinstance(question, str) else question
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "logprobs": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    if think is not None:
        body["think"] = think
    t0 = time.perf_counter()
    r, restarted = None, False
    for attempt in (1, 2):
        try:
            resp = _HTTP.post(f"{HOST}/api/chat", json=body)
            resp.raise_for_status()
            r = resp.json()
            break
        except httpx.ReadTimeout as e:
            if attempt == 2:  # timed out again on a freshly restarted runner -> final "timeout"
                return {"error": f"ReadTimeout after watchdog restart: {e}", "latency_s": time.perf_counter() - t0,
                        "watchdog_restart": True}
            restart_model_runner(model)
            restarted = True
        except Exception as e:  # recorded, never silently dropped (retried on resume)
            return {"error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - t0}
    msg = r.get("message", {})
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"__unparsed__": args}
        calls.append({"name": fn.get("name"), "arguments": dict(args or {})})
    raw = "".join(t.get("token", "") for t in (r.get("logprobs") or []))
    return {
        "content": msg.get("content") or "",
        "thinking": msg.get("thinking") or "",
        "raw_output": raw,
        "tool_calls": calls,
        "message": {"role": "assistant", "content": msg.get("content") or "",
                    **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})},
        "prompt_eval_count": r.get("prompt_eval_count"),
        "eval_count": r.get("eval_count"),
        "prompt_eval_s": (r.get("prompt_eval_duration") or 0) / 1e9,
        "eval_s": (r.get("eval_duration") or 0) / 1e9,
        "load_s": (r.get("load_duration") or 0) / 1e9,
        "latency_s": time.perf_counter() - t0,
        "done_reason": r.get("done_reason"),
        "watchdog_restart": restarted,
    }


def _probe(model: str, timeout: float = 90) -> bool:
    """True if a 5-token request to `model` completes (loads the model if needed)."""
    try:
        r = httpx.post(f"{HOST}/api/chat", timeout=timeout, json={
            "model": model, "messages": [{"role": "user", "content": "Say hi"}], "stream": False,
            "think": False, "options": {"num_predict": 5}})
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _wait_api(wait_s: int = 60) -> None:
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{HOST}/api/version", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)


def restart_model_runner(model: str) -> None:
    """Watchdog (lab notebook Entry 18). Ollama 0.32.6 sometimes wedges: every chat request hangs
    while /api/version still answers. Step 1: kill the model's llama-server process. Step 2, if a
    probe request still hangs: restart the whole server via scripts/start_ollama_lowmem.cmd."""
    import subprocess
    from pathlib import Path
    win = os.name == "nt"
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"] if win else ["pkill", "-f", "llama-server"],
                   capture_output=True)
    _wait_api()
    if _probe(model):
        return
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"] if win else ["pkill", "-f", "ollama serve"],
                   capture_output=True)
    time.sleep(3)
    launcher = Path(__file__).resolve().parents[2] / "scripts" / "start_ollama_lowmem.cmd"
    if win:
        subprocess.run(["cmd", "/c", "start", "Ollama", "/min", str(launcher)], capture_output=True)
    else:
        subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_NUM_PARALLEL": "1",
                                                     "OLLAMA_MAX_LOADED_MODELS": "1"})
    _wait_api(90)
    _probe(model, timeout=120)


def embed(model: str, texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 32):
        r = _HTTP.post(f"{HOST}/api/embed", json={"model": model, "input": texts[i:i + 32],
                                                  "keep_alive": "10m"})
        r.raise_for_status()
        out += r.json()["embeddings"]
    return out
