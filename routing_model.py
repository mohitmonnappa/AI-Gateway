"""
RouteWise — Routing Model
-------------------------
Approach: LLM-as-router
The Fast model (Groq Llama 3.1 8B) classifies each incoming prompt
before any routing decision is made. It returns a binary decision
(fast / capable) plus a confidence score and one-line reasoning.

Design rationale
----------------
- Uses the fast model itself to self-classify, keeping the system
  to two models total (no third classifier model needed)
- Structured JSON output enforces a clean decision boundary
- Confidence score (0.0–1.0) is logged with every decision
- Latency of the routing call itself is measured and returned
  so the overhead cost can be evaluated in the PoC and PPT

Input features the model sees
------------------------------
- Full prompt text (raw)
- Token-count estimate (via simple split, no tiktoken dependency here)
- Heuristic flags injected into the system prompt context:
    * estimated_tokens   — rough size signal
    * has_code_keywords  — presence of "code", "implement", "debug", etc.
    * has_math_keywords  — presence of "calculate", "prove", "derive", etc.
    * has_multi_step     — presence of "step", "first...then", "plan", etc.
  These are surfaced to the router so it has explicit signals beyond
  just reading the raw prompt.

Decision logic
--------------
Route to CAPABLE when any of the following are likely true:
  - Multi-step reasoning required
  - Code generation or debugging
  - Mathematical derivation or proof
  - Analysis of ambiguous or open-ended problems
  - Response requires synthesis of multiple concepts

Route to FAST when:
  - Factual lookup / trivia
  - Short summarisation of provided text
  - Simple yes/no or definition questions
  - Translation of a short passage
  - Basic classification with obvious answer
"""

import json
import os
import time
import re
from typing import TypedDict

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
FAST_MODEL = "llama-3.1-8b-instant"          # Groq free tier
CONFIDENCE_FALLBACK_THRESHOLD = 0.6

ROUTING_SYSTEM_PROMPT = """You are a routing classifier for an AI gateway.
Your job is to read a user prompt and decide whether it should be handled by:
  - "fast"    : a small, cheap model (good for simple, factual, or short tasks)
  - "capable" : a larger, more powerful model (needed for reasoning, code, math, multi-step tasks)

You will receive the prompt plus some heuristic signals. Use all of them.

Respond ONLY with a valid JSON object — no prose, no markdown — exactly this shape:
{
  "decision": "fast" | "capable",
  "confidence": <float 0.0 to 1.0>,
  "reason": "<one concise sentence explaining the decision>"
}

Routing guidelines:
- Route to CAPABLE for: code generation/debugging, multi-step reasoning, math proofs,
  open-ended analysis, tasks requiring synthesis of multiple concepts.
- Route to FAST for: factual Q&A, definitions, simple yes/no, short translation,
  basic classification, straightforward summarisation of provided text.
- When in doubt and confidence < 0.6, prefer "capable" to avoid quality degradation.
"""


# ---------------------------------------------------------------------------
# Heuristic feature extraction
# ---------------------------------------------------------------------------

CODE_KEYWORDS = {
    "code", "implement", "function", "debug", "programme", "program",
    "algorithm", "class", "script", "bug", "fix", "refactor", "api",
    "sql", "query", "regex", "python", "javascript", "typescript",
}

MATH_KEYWORDS = {
    "calculate", "compute", "prove", "derive", "integral", "derivative",
    "equation", "formula", "theorem", "probability", "statistics",
    "matrix", "vector", "optimise", "optimize", "gradient",
}

MULTI_STEP_KEYWORDS = {
    "step", "steps", "first", "then", "finally", "plan", "strategy",
    "process", "workflow", "sequence", "pipeline", "how do i", "how to",
    "walk me through", "explain how",
}


def extract_features(prompt: str) -> dict:
    """Extract lightweight heuristic signals from a prompt."""
    lower = prompt.lower()
    words = set(re.findall(r"\b\w+\b", lower))
    multi_step_detected = (
        bool(words & MULTI_STEP_KEYWORDS)
        or any(phrase in lower for phrase in MULTI_STEP_KEYWORDS if " " in phrase)
    )
    return {
        "estimated_tokens": len(prompt.split()),
        "has_code_keywords": bool(words & CODE_KEYWORDS),
        "has_math_keywords": bool(words & MATH_KEYWORDS),
        "has_multi_step": multi_step_detected,
        "char_count": len(prompt),
    }


# ---------------------------------------------------------------------------
# Routing model
# ---------------------------------------------------------------------------

class RoutingDecision(TypedDict):
    decision: str          # "fast" or "capable"
    confidence: float      # 0.0 – 1.0
    reason: str            # one-line explanation
    routing_latency_ms: float   # time taken for this classification call
    features: dict         # heuristic signals that were passed to the router


def route(prompt: str, api_key: str | None = None) -> RoutingDecision:
    """
    Classify a prompt using Groq Llama 3.1 8B and return a routing decision.

    Parameters
    ----------
    prompt   : the user's raw input text
    api_key  : Groq API key; falls back to GROQ_API_KEY env var if omitted

    Returns
    -------
    RoutingDecision dict with decision, confidence, reason, latency, features
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set. Export it or pass api_key=.")

    features = extract_features(prompt)

    # Build the user message — give the router both the raw prompt and the
    # extracted heuristic signals so it has structured context to reason over.
    user_message = f"""Prompt to classify:
\"\"\"{prompt}\"\"\"

Heuristic signals:
- estimated_tokens  : {features['estimated_tokens']}
- has_code_keywords : {features['has_code_keywords']}
- has_math_keywords : {features['has_math_keywords']}
- has_multi_step    : {features['has_multi_step']}
- char_count        : {features['char_count']}

Return your JSON decision now."""

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": FAST_MODEL,
        "messages": [
            {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.0,   # deterministic classification
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }

    t0 = time.perf_counter()
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=15)
    latency_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(raw)
        decision = parsed.get("decision", "capable").lower()
        if decision not in ("fast", "capable"):
            decision = "capable"
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reason = parsed.get("reason", "no reason returned")
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: if parsing fails, default to capable (safe)
        decision = "capable"
        confidence = 0.5
        reason = f"parse error — raw response: {raw[:120]}"

    if confidence < CONFIDENCE_FALLBACK_THRESHOLD:
        decision = "capable"
        reason = (
            f"low confidence ({confidence:.2f}) fallback to capable — {reason}"
        )

    return RoutingDecision(
        decision=decision,
        confidence=confidence,
        reason=reason,
        routing_latency_ms=round(latency_ms, 1),
        features=features,
    )
