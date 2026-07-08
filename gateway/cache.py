import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from threading import Lock


@dataclass
class CacheEntry:
    prompt: str
    response: str
    model_used: str


class SemanticCache:
    """Lightweight in-memory near-duplicate cache using normalized text similarity."""

    def __init__(self, threshold: float = 0.88) -> None:
        self.threshold = threshold
        self._store: dict[str, CacheEntry] = {}
        self._lock = Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def get(self, prompt: str) -> tuple[CacheEntry | None, float]:
        normalized = self._normalize(prompt)
        best_score = 0.0
        best_entry: CacheEntry | None = None

        with self._lock:
            for cached_prompt, entry in self._store.items():
                score = SequenceMatcher(None, normalized, cached_prompt).ratio()
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry and best_score >= self.threshold:
            return best_entry, round(best_score, 3)
        return None, round(best_score, 3)

    def put(self, prompt: str, response: str, model_used: str) -> None:
        normalized = self._normalize(prompt)
        entry = CacheEntry(prompt=prompt, response=response, model_used=model_used)
        with self._lock:
            self._store[normalized] = entry
