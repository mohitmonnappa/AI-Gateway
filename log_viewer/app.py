import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "gateway_log.jsonl"

st.set_page_config(page_title="RouteWise Log Viewer", layout="wide")
st.title("RouteWise Gateway Logs")

if not LOG_FILE.exists():
    st.warning("No logs yet. Run the gateway and call /chat first.")
    st.stop()

rows = []
with open(LOG_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    st.warning("Log file is empty.")
    st.stop()

df = pd.DataFrame(rows)
df = df.sort_values("timestamp", ascending=False)

st.metric("Total Requests", len(df))
st.metric("Cache Hit Rate", f"{(df['cache_hit'].mean() * 100):.1f}%")
st.metric("Avg Total Latency", f"{df['total_latency_ms'].mean():.1f} ms")

st.dataframe(
    df[
        [
            "timestamp",
            "prompt_snippet",
            "model_used",
            "routing_reason",
            "routing_confidence",
            "routing_latency_ms",
            "total_latency_ms",
            "cache_hit",
            "cache_similarity",
        ]
    ],
    use_container_width=True,
)
