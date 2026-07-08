"""
RouteWise — PoC Evaluator (poc.py)
-----------------------------------
Standalone proof-of-concept for the routing model.

Usage
-----
    # Set your Groq API key first:
    export GROQ_API_KEY=your_key_here

    # Run against the default test suite:
    python poc.py

    # Run against a custom JSON or CSV file:
    python poc.py --input my_prompts.json
    python poc.py --input my_prompts.csv

    # Verbose mode (prints per-prompt features and full reason):
    python poc.py --verbose

Input format
------------
JSON:  list of objects with keys: "id", "prompt", "label"  (label: "fast" | "capable")
CSV:   columns: id, prompt, label

Output
------
Per prompt:
    [CORRECT / WRONG] #{id} | GT: {label} | Decision: {decision} | Conf: {conf:.2f} | {reason}

Summary:
    Total | Correct | Accuracy | FP rate | FN rate | Avg routing latency
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from routing_model import extract_features, route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_suite(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    if p.suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # normalise
        for r in rows:
            r["id"] = int(r.get("id", 0))
        return rows

    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data


def print_separator(char="─", width=80):
    print(char * width)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate from word count.
    Keeps PoC fully offline and dependency-light.
    """
    return max(1, int(len(text.split()) * 1.25))


def estimate_completion_tokens(prompt: str) -> int:
    """Estimate answer length from prompt complexity signals."""
    features = extract_features(prompt)
    base = max(35, int(features["estimated_tokens"] * 0.7))
    if features["has_multi_step"]:
        base += 100
    if features["has_code_keywords"]:
        base += 80
    if features["has_math_keywords"]:
        base += 70
    return min(420, base)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(test_suite: list[dict], api_key: str, verbose: bool = False) -> None:
    total     = len(test_suite)
    correct   = 0
    fp        = 0   # false positive: simple prompt sent to capable
    fn        = 0   # false negative: complex prompt sent to fast
    latencies = []
    misroutes = []
    errors    = []
    routed_decisions: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Notional costs (USD per 1M tokens). Can be overridden via env vars.
    fast_in = float(os.getenv("FAST_INPUT_COST_PER_1M", "0.05"))
    fast_out = float(os.getenv("FAST_OUTPUT_COST_PER_1M", "0.08"))
    capable_in = float(os.getenv("CAPABLE_INPUT_COST_PER_1M", "0.075"))
    capable_out = float(os.getenv("CAPABLE_OUTPUT_COST_PER_1M", "0.30"))
    router_out_tokens = int(os.getenv("ROUTER_OUTPUT_TOKENS_EST", "20"))

    print_separator("═")
    print(" RouteWise — Routing Model PoC Evaluation")
    print(f" Model: LLM-as-router via Groq Llama 3.1 8B")
    print(f" Prompts: {total}")
    print_separator("═")
    print()

    for item in test_suite:
        pid     = item.get("id", "?")
        prompt  = item["prompt"]
        gt      = item["label"].lower().strip()   # "fast" or "capable"
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_completion_tokens(prompt)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        try:
            result = route(prompt, api_key=api_key)
        except Exception as e:
            errors.append((pid, str(e)))
            routed_decisions.append("capable")
            print(f"  [ERROR] #{pid} — {e}")
            continue

        decision   = result["decision"]
        confidence = result["confidence"]
        reason     = result["reason"]
        latency    = result["routing_latency_ms"]
        features   = result["features"]
        routed_decisions.append(decision)

        latencies.append(latency)
        is_correct = decision == gt

        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG  "
            if gt == "fast" and decision == "capable":
                fp += 1   # simple → capable (over-routing)
            elif gt == "capable" and decision == "fast":
                fn += 1   # complex → fast  (under-routing, worse outcome)
            misroutes.append(
                {
                    "id": pid,
                    "ground_truth": gt,
                    "decision": decision,
                    "confidence": confidence,
                    "reason": reason,
                    "rationale": item.get("rationale", ""),
                }
            )

        # Per-prompt line
        conf_str = f"{confidence:.2f}"
        label_str = f"GT: {gt:>7}  →  Decision: {decision:>7}"
        latency_str = f"{latency:>6.0f}ms"
        print(f"  [{status}]  #{str(pid):<3}  {label_str}  Conf: {conf_str}  {latency_str}")
        print(f"           Reason: {reason}")

        if verbose:
            print(f"           Features: tokens={features['estimated_tokens']}, "
                  f"code={features['has_code_keywords']}, "
                  f"math={features['has_math_keywords']}, "
                  f"multi_step={features['has_multi_step']}")
        print()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    evaluated = total - len(errors)
    accuracy  = correct / evaluated if evaluated else 0.0
    fp_rate   = fp / evaluated if evaluated else 0.0
    fn_rate   = fn / evaluated if evaluated else 0.0
    avg_lat   = sum(latencies) / len(latencies) if latencies else 0.0
    p50_lat   = sorted(latencies)[len(latencies) // 2] if latencies else 0.0

    # Cost model:
    # - Baseline: all prompts to capable model.
    # - Smart routing: every prompt incurs router cost + selected model cost.
    # Completion token estimates are identical between strategies (task-driven),
    # while per-model token rates differ.
    baseline_cost = (
        (total_prompt_tokens * capable_in) + (total_completion_tokens * capable_out)
    ) / 1_000_000

    smart_cost = 0.0
    for item, routed in zip(test_suite, routed_decisions):
        prompt = item["prompt"]
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_completion_tokens(prompt)

        # Router call itself (always fast model)
        router_input_tokens = prompt_tokens + 40
        smart_cost += ((router_input_tokens * fast_in) + (router_out_tokens * fast_out)) / 1_000_000

        if routed == "fast":
            smart_cost += ((prompt_tokens * fast_in) + (completion_tokens * fast_out)) / 1_000_000
        else:
            smart_cost += ((prompt_tokens * capable_in) + (completion_tokens * capable_out)) / 1_000_000

    cost_reduction = ((baseline_cost - smart_cost) / baseline_cost) if baseline_cost else 0.0

    print_separator("═")
    print(" SUMMARY")
    print_separator()
    print(f"  Prompts evaluated    : {evaluated} / {total}")
    print(f"  Correct              : {correct}")
    print(f"  Accuracy             : {accuracy:.1%}")
    print()
    print(f"  False positives (FP) : {fp}   — simple prompt over-routed to Capable")
    print(f"    FP rate            : {fp_rate:.1%}")
    print()
    print(f"  False negatives (FN) : {fn}   — complex prompt under-routed to Fast")
    print(f"    FN rate            : {fn_rate:.1%}")
    print()
    print(f"  Avg routing latency  : {avg_lat:.0f}ms")
    print(f"  p50 routing latency  : {p50_lat:.0f}ms")

    if latencies:
        print(f"  Min / Max latency    : {min(latencies):.0f}ms / {max(latencies):.0f}ms")

    if errors:
        print()
        print(f"  Errors               : {len(errors)}")
        for eid, emsg in errors:
            print(f"    #{eid}: {emsg}")

    print()
    print(" COST COMPARISON (estimated)")
    print_separator()
    print(f"  Prompt tokens total             : {total_prompt_tokens}")
    print(f"  Completion tokens total (est)   : {total_completion_tokens}")
    print(f"  Always-capable estimated cost   : ${baseline_cost:.6f}")
    print(f"  Smart-routing estimated cost    : ${smart_cost:.6f}")
    print(f"  Estimated cost reduction        : {cost_reduction:.1%}")

    if misroutes:
        print()
        print(" HARDEST MIS-ROUTES (top 3 by confidence)")
        print_separator()
        for m in sorted(misroutes, key=lambda x: x["confidence"], reverse=True)[:3]:
            print(
                f"  #{m['id']} GT={m['ground_truth']} → decision={m['decision']} "
                f"(conf={m['confidence']:.2f})"
            )
            print(f"     Router reason : {m['reason']}")
            if m["rationale"]:
                print(f"     Ground truth  : {m['rationale']}")

    print_separator("═")

    # Exit code — fail if below 75% accuracy target
    target = 0.75
    if accuracy < target:
        print(f"\n  ⚠  Accuracy {accuracy:.1%} is below the target of {target:.0%}")
        sys.exit(1)
    else:
        print(f"\n  ✓  Accuracy {accuracy:.1%} meets the >{target:.0%} target")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RouteWise PoC — evaluate the routing model against a labelled test suite"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(Path(__file__).parent / "test_suite.json"),
        help="Path to JSON or CSV test suite (default: test_suite.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extracted heuristic features for each prompt"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[ERROR] GROQ_API_KEY environment variable is not set.")
        print("        Export it: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    test_suite = load_test_suite(args.input)
    evaluate(test_suite, api_key=api_key, verbose=args.verbose)


if __name__ == "__main__":
    main()
