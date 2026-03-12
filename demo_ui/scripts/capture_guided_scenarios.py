#!/usr/bin/env python3
"""
Capture Guided Demo Scenarios

Calls /compare for all 8 guided-demo combinations (4 scenarios x 2 methods)
and saves the results as guided-demo-data.json for offline playback.

Usage:
    python capture_guided_scenarios.py [--backend-url URL]
    python capture_guided_scenarios.py --scenario improve_frontier_self_consistency

Requirements:
    - Backend server must be running (default: http://localhost:8000)
    - Models must be configured and accessible (gpt-4o, gpt-4.1-nano,
      gpt-4.1, llama-3.2-3b)
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ============================================================
# Scenario + method configurations
# ============================================================

GUIDED_CONFIGS = [
    # --- improve_frontier ---
    # GPT-4.1 Nano: small model that makes frequent errors on moderate math.
    # SC majority voting recovers the correct answer.
    # Retry: keep recapturing until baseline is wrong but ITS is right.
    {
        "key": "improve_frontier_self_consistency",
        "scenario_id": "improve_frontier",
        "use_case": "improve_model",
        "algorithm": "self_consistency",
        "model_id": "gpt-4.1-nano",
        "budget": 8,
        "question": "A palindrome is a number that reads the same forwards and backwards. How many 5-digit palindromes are divisible by 3?",
        "expected_answer": "300",
        "question_type": "math",
        "require_improvement": True,
    },
    {
        "key": "improve_frontier_best_of_n",
        "scenario_id": "improve_frontier",
        "use_case": "improve_model",
        "algorithm": "best_of_n",
        "model_id": "gpt-4.1-nano",
        "budget": 4,
        "question": "An investment of $10,000 earns 8% annual interest compounded quarterly. After 3 years, how much total interest has been earned? Round to the nearest cent.",
        "question_type": "general",
        "judge_criterion": "Score 1-10 strictly: The correct formula is A = P(1 + r/n)^(nt) = 10000(1.02)^12 = $12,682.42. Interest earned = $2,682.42. Deduct 5 points if final answer is wrong by more than $1. Deduct 3 points if formula is wrong. Deduct 2 points if work is unclear or steps are missing. Only 9-10 if answer is exactly $2,682.42 with clear work shown.",
        "require_improvement": True,
    },
    # --- improve_opensource ---
    # Llama 3.2 3B: 3B open-source model that struggles with probability.
    # SC majority voting recovers the correct answer.
    {
        "key": "improve_opensource_self_consistency",
        "scenario_id": "improve_opensource",
        "use_case": "improve_model",
        "algorithm": "self_consistency",
        "model_id": "llama-3.2-3b",
        "budget": 8,
        "question": "In how many ways can 5 letters be placed in 5 addressed envelopes so that no letter is in its correct envelope?",
        "expected_answer": "44",
        "question_type": "math",
        "require_improvement": True,
    },
    {
        "key": "improve_opensource_best_of_n",
        "scenario_id": "improve_opensource",
        "use_case": "improve_model",
        "algorithm": "best_of_n",
        "model_id": "llama-3.2-3b",
        "budget": 4,
        "question": "A store buys shirts for $15 each and sells them for $25 each. Last month they sold 400 shirts. This month, they offered a 10% discount and sold 500 shirts. Calculate: (1) last month's profit, (2) this month's profit, (3) which month was more profitable and by how much.",
        "question_type": "general",
        "judge_criterion": "Score 1-10 strictly: Last month profit = 400 x (25-15) = $4000. This month sale price = $22.50, profit per shirt = $7.50, total = 500 x 7.50 = $3750. Last month more profitable by $250. Deduct 3 points per wrong calculation. Deduct 2 points if conclusion contradicts the numbers. Only 9-10 if all three parts calculated correctly with clear work.",
        "require_improvement": True,
    },
    # --- match_same_family ---
    # GPT-4.1-nano vs GPT-4.1: modular arithmetic is error-prone for small models.
    # ITS costs less than the frontier model while matching quality.
    {
        "key": "match_same_family_self_consistency",
        "scenario_id": "match_same_family",
        "use_case": "match_frontier",
        "algorithm": "self_consistency",
        "model_id": "gpt-4.1-nano",
        "frontier_model_id": "gpt-4.1",
        "budget": 8,
        "question": "A palindrome is a number that reads the same forwards and backwards. How many 5-digit palindromes are divisible by 3?",
        "expected_answer": "300",
        "question_type": "math",
        "require_improvement": True,
    },
    {
        "key": "match_same_family_best_of_n",
        "scenario_id": "match_same_family",
        "use_case": "match_frontier",
        "algorithm": "best_of_n",
        "model_id": "gpt-4.1-nano",
        "frontier_model_id": "gpt-4.1",
        "budget": 8,
        "question": "An investment of $10,000 earns 8% annual interest compounded quarterly. After 3 years, how much total interest has been earned? Round to the nearest cent.",
        "question_type": "general",
        "judge_criterion": "Score 1-10 strictly: The correct formula is A = P(1 + r/n)^(nt) = 10000(1.02)^12 = $12,682.42. Interest earned = $2,682.42. Deduct 5 points if final answer is wrong by more than $1. Deduct 3 points if formula is wrong. Deduct 2 points if work is unclear or steps are missing. Only 9-10 if answer is exactly $2,682.42 with clear work shown.",
    },
    # --- match_cross_family ---
    # Llama 3.2 3B vs GPT-4o: dramatic cost savings (~20x cheaper).
    # ITS costs ~$0.0005 for Llama+ITS vs ~$0.01 for GPT-4o.
    {
        "key": "match_cross_family_self_consistency",
        "scenario_id": "match_cross_family",
        "use_case": "match_frontier",
        "algorithm": "self_consistency",
        "model_id": "llama-3.2-3b",
        "frontier_model_id": "gpt-4o",
        "budget": 8,
        "question": "In how many ways can 5 letters be placed in 5 addressed envelopes so that no letter is in its correct envelope?",
        "expected_answer": "44",
        "question_type": "math",
        "require_improvement": True,
    },
    {
        "key": "match_cross_family_best_of_n",
        "scenario_id": "match_cross_family",
        "use_case": "match_frontier",
        "algorithm": "best_of_n",
        "model_id": "llama-3.2-3b",
        "frontier_model_id": "gpt-4o",
        "budget": 8,
        "question": "A store buys shirts for $15 each and sells them for $25 each. Last month they sold 400 shirts. This month, they offered a 10% discount and sold 500 shirts. Calculate: (1) last month's profit, (2) this month's profit, (3) which month was more profitable and by how much.",
        "question_type": "general",
        "judge_criterion": "Score 1-10 strictly: Last month profit = 400 x (25-15) = $4000. This month sale price = $22.50, profit per shirt = $7.50, total = 500 x 7.50 = $3750. Last month more profitable by $250. Deduct 3 points per wrong calculation. Deduct 2 points if conclusion contradicts the numbers. Only 9-10 if all three parts calculated correctly with clear work.",
    },
]


# ============================================================
# Vote-count key simplification
# ============================================================

def _unwrap_tuple_key(key: str) -> str:
    """Unwrap vote_counts keys from Python tuple repr format.

    The backend's self-consistency algorithm stores vote keys as Python
    tuple reprs, e.g. ``"('\\\\frac{29',)"`` or ``"('2',)"``.
    This extracts the inner string value.
    """
    # Match ('value',) or (None,) patterns
    m = re.match(r"^\('?(.*?)'?,\)$", key.strip())
    if m:
        inner = m.group(1)
        # Unescape doubled backslashes from JSON encoding
        return inner.replace("\\\\", "\\")
    return key


def _simplify_latex(text: str) -> str:
    """Convert LaTeX math fragments to readable labels.

    E.g. ``\\frac{29}{44}`` → ``29/44``, ``\\frac{11}{850}`` → ``11/850``.
    """
    # \frac{a}{b} → a/b
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    # Preserve common math symbols as Unicode before stripping commands
    text = text.replace("\\pi", "π")
    text = text.replace("\\sqrt", "√")
    text = text.replace("\\times", "×")
    text = text.replace("\\cdot", "·")
    # Strip remaining LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Clean up braces and whitespace
    text = text.replace("{", "").replace("}", "").strip()
    return text


def simplify_vote_counts(vote_counts: dict[str, int], is_math: bool) -> dict[str, int]:
    """Simplify verbose vote_counts keys to concise labels.

    The backend stores vote keys as Python tuple reprs of extracted answers.
    This function unwraps them and converts LaTeX to human-readable text.
    """
    simplified: dict[str, int] = {}
    for raw_key, count in vote_counts.items():
        inner = _unwrap_tuple_key(raw_key)
        if inner == "None":
            label = "(no answer)"
        elif is_math:
            label = _simplify_latex(inner)
        else:
            label = inner.strip().split("\n")[0][:80]
            if len(inner.strip().split("\n")[0]) > 80:
                label += "..."

        # Merge counts if labels collide after simplification
        simplified[label] = simplified.get(label, 0) + count

    return simplified


# ============================================================
# API call + response transformation
# ============================================================

async def capture_one(
    client: httpx.AsyncClient,
    backend_url: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Call /compare and transform to guided-demo format."""
    key = config["key"]
    print(f"\n{'='*60}")
    print(f"Capturing: {key}")
    print(f"  model={config['model_id']}  algorithm={config['algorithm']}  budget={config['budget']}")
    print(f"{'='*60}")

    request_body: dict[str, Any] = {
        "question": config["question"],
        "model_id": config["model_id"],
        "algorithm": config["algorithm"],
        "budget": config["budget"],
        "use_case": config["use_case"],
        "question_type": config.get("question_type", "auto"),
    }
    if config.get("frontier_model_id"):
        request_body["frontier_model_id"] = config["frontier_model_id"]
    if config.get("expected_answer"):
        request_body["expected_answer"] = config["expected_answer"]
    if config.get("judge_criterion"):
        request_body["judge_criterion"] = config["judge_criterion"]

    try:
        resp = await client.post(
            f"{backend_url}/compare",
            json=request_body,
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        print(f"  HTTP error: {e}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    is_match = config["use_case"] == "match_frontier"
    is_math = config.get("question_type") == "math"

    # --- Build result dict in guided-demo format ---

    def _detail(rd: dict) -> dict:
        return {
            "response": rd["answer"],
            "latency_ms": rd["latency_ms"],
            "input_tokens": rd.get("input_tokens", 0) or 0,
            "output_tokens": rd.get("output_tokens", 0) or 0,
            "cost_usd": rd.get("cost_usd", 0) or 0,
        }

    if is_match:
        # match_frontier: API small_baseline → guided baseline,
        #                 API its            → guided its,
        #                 API baseline       → guided frontier
        result = {
            "baseline": _detail(data["small_baseline"]),
            "its": _detail(data["its"]),
            "frontier": _detail(data["baseline"]),
        }
    else:
        # improve_model: API baseline → guided baseline, API its → guided its
        result = {
            "baseline": _detail(data["baseline"]),
            "its": _detail(data["its"]),
        }

    # --- Trace ---
    trace_raw = data["its"].get("trace")
    if trace_raw:
        trace = trace_raw if isinstance(trace_raw, dict) else json.loads(trace_raw)

        if trace.get("algorithm") == "self_consistency" and trace.get("vote_counts"):
            trace["vote_counts"] = simplify_vote_counts(trace["vote_counts"], is_math)

        result["trace"] = trace

    result["question"] = config["question"]
    if config.get("expected_answer"):
        result["expected_answer"] = config["expected_answer"]

    print(f"  Baseline: {result['baseline']['latency_ms']}ms, "
          f"{result['baseline']['output_tokens']} tokens")
    print(f"  ITS:      {result['its']['latency_ms']}ms, "
          f"{result['its']['output_tokens']} tokens")
    if is_match:
        print(f"  Frontier: {result['frontier']['latency_ms']}ms, "
              f"{result['frontier']['output_tokens']} tokens")

    return result


def _responses_differ(result: dict[str, Any], algo: str) -> bool:
    """Check whether the ITS response is meaningfully different from baseline.

    For SC: the baseline and ITS final answers should differ (baseline wrong,
    ITS right via majority vote).
    For BoN: the ITS response should be longer or structurally different from
    baseline (judge selected a better candidate).
    """
    b_resp = result["baseline"]["response"]
    i_resp = result["its"]["response"]

    if algo == "self_consistency":
        # For math SC: check if the two answers are textually different
        # (one is wrong, the other is right via voting)
        b_short = b_resp[:200].strip()
        i_short = i_resp[:200].strip()
        return b_short != i_short

    # For BoN: responses should be different text (judge picked a different candidate)
    return b_resp.strip() != i_resp.strip()


# ============================================================
# Main
# ============================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture guided demo scenarios from the ITS backend"
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: ../frontend/guided-demo-data.json)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Capture a single scenario by key (e.g. improve_frontier_self_consistency). "
             "Merges into existing output file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "frontend" / "guided-demo-data.json"
    )

    # Filter configs if --scenario is specified
    if args.scenario:
        configs = [c for c in GUIDED_CONFIGS if c["key"] == args.scenario]
        if not configs:
            valid_keys = [c["key"] for c in GUIDED_CONFIGS]
            print(f"Unknown scenario: {args.scenario}")
            print(f"Valid keys: {', '.join(valid_keys)}")
            sys.exit(1)
    else:
        configs = GUIDED_CONFIGS

    print(f"Backend URL: {args.backend_url}")
    print(f"Output file: {output_path}")
    print(f"Scenarios:   {len(configs)}")

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            h = await client.get(f"{args.backend_url}/health", timeout=5.0)
            h.raise_for_status()
            print("Backend is reachable")
        except Exception as e:
            print(f"Backend not reachable: {e}")
            print("\nStart the backend first, e.g.:")
            print("  cd demo_ui && uvicorn backend.main:app --port 8000")
            sys.exit(1)

    # Load existing data if merging a single scenario
    if args.scenario and output_path.exists():
        results: dict[str, Any] = json.loads(output_path.read_text())
        print(f"Loaded {len(results)} existing scenarios from {output_path}")
    else:
        results = {}

    # Set up capture log directory
    log_dir = Path(__file__).resolve().parent / "capture_logs"
    log_dir.mkdir(exist_ok=True)

    # Capture combinations
    # For configs with require_improvement=True, retry up to MAX_RETRIES
    # times until the baseline and ITS responses visibly differ.
    MAX_RETRIES = 5
    async with httpx.AsyncClient() as client:
        for config in configs:
            need_improvement = config.get("require_improvement", False)
            algo = config["algorithm"]
            best_result = None

            attempts = MAX_RETRIES if need_improvement else 1
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    print(f"  Retry {attempt}/{attempts} — looking for baseline != ITS ...")
                result = await capture_one(client, args.backend_url, config)
                if not result:
                    continue

                if not need_improvement or _responses_differ(result, algo):
                    best_result = result
                    if need_improvement:
                        print(f"  Found improvement on attempt {attempt}")
                    break
                else:
                    print(f"  Baseline and ITS are too similar, retrying...")
                    best_result = result  # keep as fallback
                    await asyncio.sleep(1)

            if best_result:
                results[config["key"]] = best_result
                # Write capture log
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"{config['key']}_{timestamp}.json"
                log_file.write_text(json.dumps(best_result, indent=2, ensure_ascii=False) + "\n")
                print(f"  Log saved: {log_file.name}")
            else:
                print(f"  SKIPPED (failed): {config['key']}")
            await asyncio.sleep(1)

    if not results:
        print("\nNo scenarios captured successfully.")
        sys.exit(1)

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully captured {len(results)}/{len(GUIDED_CONFIGS)} scenarios")
    print(f"Written to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Open the frontend and test all 4 scenarios x 2 methods")
    print(f"  2. Verify no '[Placeholder]' text remains")
    print(f"  3. Commit guided-demo-data.json")


if __name__ == "__main__":
    asyncio.run(main())
