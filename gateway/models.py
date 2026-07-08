from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt text")


class ChatResponse(BaseModel):
    response: str
    model_used: str
    routing_reason: str
    routing_confidence: float
    routing_latency_ms: float
    total_latency_ms: float
    cache_hit: bool
    cache_similarity: float


class LogRecord(BaseModel):
    timestamp: str
    prompt: str
    prompt_snippet: str
    model_used: str
    routing_reason: str
    routing_confidence: float
    routing_latency_ms: float
    total_latency_ms: float
    cache_hit: bool
    cache_similarity: float
    response_preview: str
