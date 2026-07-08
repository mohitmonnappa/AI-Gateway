# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RouteWise is an AI gateway that routes prompts between two LLMs based on complexity: a Fast model (Groq Llama 3.1 8B Instant) for simple queries, and a Capable model (Google Gemini 2.5 Flash) for reasoning-heavy tasks. Routing itself is done by an LLM-as-router: the Fast model classifies each prompt (with heuristic signals attached) into `fast` or `capable` before the actual answer is generated.

## Commands

```bash
pip install -r requirements.txt

# Run the FastAPI gateway (requires GROQ_API_KEY and GEMINI_API_KEY in .env)
uvicorn gateway.main:app

# Evaluate routing accuracy/latency/cost against the labeled test suite
python poc.py
python poc.py --verbose                  # print per-prompt heuristic features
python poc.py --input path/to/file.json  # or .csv, same schema (id, prompt, label)

# Streamlit chat client (talks to the gateway at localhost:8000/chat)
streamlit run chat_ui.py

# Streamlit dashboard over logs/gateway_log.jsonl
streamlit run log_viewer/app.py
```

There is no test framework wired up (no pytest suite) — `poc.py` is the correctness/accuracy check for the routing model, and it exits with status 1 if accuracy drops below 75% against `test_suite.json`.

Required env vars (see `.env`): `GROQ_API_KEY`, `GEMINI_API_KEY`, optional `CACHE_THRESHOLD` (default 0.88). `poc.py` also reads optional cost-override env vars (`FAST_INPUT_COST_PER_1M`, `FAST_OUTPUT_COST_PER_1M`, `CAPABLE_INPUT_COST_PER_1M`, `CAPABLE_OUTPUT_COST_PER_1M`, `ROUTER_OUTPUT_TOKENS_EST`).

## Architecture

Request flow through `gateway/main.py` `POST /chat`:
1. `gateway/cache.py` `SemanticCache` checks for a near-duplicate prior prompt (normalized whitespace/case + `difflib.SequenceMatcher` ratio against every cached prompt — O(n) linear scan, not embedding-based despite "semantic" in the name). On hit, returns the cached response immediately with `cache_hit=True` and skips routing/generation entirely.
2. On a cache miss, `routing_model.route()` is called: it extracts heuristic features (`estimated_tokens`, `has_code_keywords`, `has_math_keywords`, `has_multi_step`) via `extract_features()`, sends the raw prompt + those signals to the Fast model (Groq) with a JSON-only system prompt, and parses back `{decision, confidence, reason}`. If confidence < `CONFIDENCE_FALLBACK_THRESHOLD` (0.6) or JSON parsing fails, the decision is forced to `capable` (fail-safe toward quality).
3. Based on `routing["decision"]`, `main.py` calls either `_call_fast_model` (Groq chat completions) or `_call_capable_model` (Gemini `generateContent`) directly via `requests` — `litellm` is in requirements but not currently used in `gateway/main.py`.
4. The response is written into the cache (`cache.put`) and appended as a `LogRecord` (JSON line) to `logs/gateway_log.jsonl`, which `log_viewer/app.py` reads to render metrics (cache hit rate, avg latency) via Streamlit/pandas.

Key modules:
- `gateway/main.py` — FastAPI app, direct HTTP calls to Groq/Gemini, cache + logging orchestration.
- `gateway/cache.py` — `SemanticCache` / `CacheEntry`, thread-safe via a single `Lock`.
- `gateway/models.py` — Pydantic schemas (`ChatRequest`, `ChatResponse`, `LogRecord`) shared between the API and the log format.
- `routing_model.py` — standalone, importable routing logic (`extract_features`, `route`) used by both `gateway/main.py` and `poc.py`; keep it free of FastAPI/Streamlit dependencies so it stays usable from the PoC script.
- `poc.py` — offline evaluator; estimates tokens/cost without calling the Capable model at all (cost comparison is simulated, not measured).
- `chat_ui.py` / `log_viewer/app.py` — two independent Streamlit apps (chat client vs. log dashboard), not part of the FastAPI process.

Note: `README.md` describes a `gateway/cache.py` as an in-memory semantic cache — accurate, but be aware "semantic" here means string-similarity, not embedding similarity (no `sentence-transformers` usage found despite it being in requirements.txt).
