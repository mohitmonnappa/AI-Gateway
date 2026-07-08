import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from gateway.cache import SemanticCache
from gateway.models import ChatRequest, ChatResponse, LogRecord
from routing_model import route

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

FAST_MODEL = "llama-3.1-8b-instant"
LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "gateway_log.jsonl"

app = FastAPI(title="RouteWise AI Gateway", version="0.1.0")
cache = SemanticCache(threshold=float(os.getenv("CACHE_THRESHOLD", "0.88")))


def _call_fast_model(prompt: str, api_key: str) -> str:
    payload = {
        "model": FAST_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=25)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_capable_model(prompt: str, api_key: str) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    response = requests.post(
        f"{GEMINI_API_URL}?key={api_key}",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return parts[0].get("text", "")


def _append_log(record: LogRecord) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()

    # 1) Cache check before any routing/model call.
    cached, similarity = cache.get(request.prompt)
    if cached is not None:
        total_latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response = ChatResponse(
            response=cached.response,
            model_used=cached.model_used,
            routing_reason=f"cache hit (similarity={similarity})",
            routing_confidence=1.0,
            routing_latency_ms=0.0,
            total_latency_ms=total_latency_ms,
            cache_hit=True,
            cache_similarity=similarity,
        )
        _append_log(
            LogRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                prompt=request.prompt,
                prompt_snippet=request.prompt[:120],
                model_used=cached.model_used,
                routing_reason=response.routing_reason,
                routing_confidence=response.routing_confidence,
                routing_latency_ms=response.routing_latency_ms,
                total_latency_ms=response.total_latency_ms,
                cache_hit=True,
                cache_similarity=similarity,
                response_preview=cached.response[:160],
            )
        )
        return response

    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not groq_key or not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="Missing GROQ_API_KEY or GEMINI_API_KEY. Set them in environment or .env.",
        )

    # 2) Route request to Fast or Capable model.
    try:
        routing = route(request.prompt, api_key=groq_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Routing model failed: {exc}") from exc

    model_used = routing["decision"]

    # 3) Generate answer using selected model.
    try:
        if model_used == "fast":
            answer = _call_fast_model(request.prompt, groq_key)
        else:
            answer = _call_capable_model(request.prompt, gemini_key)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

    cache.put(request.prompt, answer, model_used)

    total_latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response = ChatResponse(
        response=answer,
        model_used=model_used,
        routing_reason=routing["reason"],
        routing_confidence=float(routing["confidence"]),
        routing_latency_ms=float(routing["routing_latency_ms"]),
        total_latency_ms=total_latency_ms,
        cache_hit=False,
        cache_similarity=similarity,
    )

    _append_log(
        LogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=request.prompt,
            prompt_snippet=request.prompt[:120],
            model_used=model_used,
            routing_reason=response.routing_reason,
            routing_confidence=response.routing_confidence,
            routing_latency_ms=response.routing_latency_ms,
            total_latency_ms=response.total_latency_ms,
            cache_hit=False,
            cache_similarity=similarity,
            response_preview=answer[:160],
        )
    )

    return response
