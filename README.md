# RouteWise AI Gateway

RouteWise is an AI gateway that routes prompts between two LLMs based on complexity, optimizing for cost and latency without sacrificing quality on hard tasks.

## Models Used

1. **Fast Model (Groq — Llama 3.1 8B Instant)**
   - **Handles:** Simple factual queries, short summaries, definitions, and basic classification.
   - **Why:** Extremely low latency and cost-effective for tasks that don't require deep reasoning. This model also serves as the internal routing engine that classifies prompts before they are executed.

2. **Capable Model (Google Gemini 2.5 Flash)**
   - **Handles:** Complex problem-solving, code generation, multi-step reasoning, mathematical derivations, and open-ended analysis.
   - **Why:** Provides robust reasoning and higher-quality outputs for tasks where the Fast Model would degrade quality.

Routing itself is done by an **LLM-as-router**: the Fast model classifies each prompt — with heuristic signals (estimated tokens, code/math keywords, multi-step phrasing) attached — into `fast` or `capable` before the actual answer is generated. If the router's confidence comes back below 0.6, or its response fails to parse, the decision falls back to `capable` to avoid a quality hit.

## Setup Instructions

```bash
git clone <repo-url> && cd ai_gateway
pip install -r requirements.txt
cp .env.example .env          # fill in GROQ_API_KEY and GEMINI_API_KEY
uvicorn gateway.main:app      # start the FastAPI gateway server
```

Required environment variables (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq API key, used for the Fast model and the router |
| `GEMINI_API_KEY` | yes | Google Gemini API key, used for the Capable model |
| `CACHE_THRESHOLD` | no (default `0.88`) | Similarity threshold (0–1) above which a prompt is served from cache |

## System Components

- **Gateway Server (`gateway/main.py`)** — FastAPI app exposing `POST /chat` and `GET /health`. Checks the cache, invokes the routing model on a miss, calls the selected LLM, then caches and logs the result.
- **Routing Model (`routing_model.py`)** — LLM-as-router: extracts heuristic features (token count, code/math keywords, multi-step phrasing) and asks the Fast model to classify the prompt as `fast` or `capable`.
- **Cache Layer (`gateway/cache.py`)** — In-memory near-duplicate cache. Prompts are normalized (lowercased, whitespace-collapsed) and compared against every cached prompt using `difflib.SequenceMatcher` string similarity — this is **not** an embedding-based semantic cache, just fast/cheap text-similarity matching.
- **Chat Client (`chat_ui.py`)** — Streamlit app that talks to the gateway's `/chat` endpoint and shows routing metadata (model used, confidence, latency, cache hit) alongside each reply.
- **Log Viewer (`log_viewer/app.py`)** — Streamlit dashboard over `logs/gateway_log.jsonl`, showing request volume, cache hit rate, and average latency.

## Proof of Concept (PoC) Evaluator

`poc.py` is a standalone, offline evaluator that measures the routing model's accuracy, latency, and estimated cost impact against a labeled test suite (`test_suite.json`) — it calls the router (Groq) for real but never calls the Capable model, so cost comparisons are simulated from token estimates, not measured.

```bash
export GROQ_API_KEY=your_key_here

python poc.py                            # run against test_suite.json
python poc.py --verbose                  # also print per-prompt heuristic features
python poc.py --input path/to/file.json  # or .csv with the same id/prompt/label schema
```

The evaluator exits with status 1 if routing accuracy falls below 75%.

## Running the Chat Client and Log Viewer

With the gateway running (`uvicorn gateway.main:app`), start either Streamlit app in a separate terminal:

```bash
streamlit run chat_ui.py             # interactive chat client
streamlit run log_viewer/app.py      # dashboard over logged requests
```
