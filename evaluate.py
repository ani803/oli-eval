#!/usr/bin/env python3
"""Evaluate an assistant transcript against expected Oli tool behaviour.

Input JSON format:
{
  "case_id": "TC01",
  "assistant_text": "Final assistant response...",
  "tool_calls": [{"name":"check_offer","arguments":{},"result":"...","success":true}],
  "observations": {
    "offer_checked": true,
    "persistent_write_authorized": true
  },
  "critical_failures": [],
  "notes": "optional"
}

This is a lightweight rubric evaluator, not an LLM judge. It checks logged evidence
and explicit annotations; it cannot infer hidden tool calls from prose.
"""
import argparse
import json
import sys
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_case(case, transcript):
    calls = transcript.get("tool_calls", [])
    names = [c.get("name", "") for c in calls]
    observations = transcript.get("observations", {})
    assistant_text = transcript.get("assistant_text", "")
    critical_failures = transcript.get("critical_failures", [])
    expected = case.get("expected_tools", [])
    results = []

    # Tool selection: expected calls must appear in order; empty expected list means
    # the case is evaluated on the refusal/no-write behaviour, not absence of all reads.
    if expected:
        cursor = 0
        for expected_name in expected:
            try:
                cursor = names.index(expected_name, cursor) + 1
            except ValueError:
                break
        else:
            results.append(("tool_selection", 2, "Expected tool sequence observed."))
        if not results or results[-1][0] != "tool_selection":
            results.append(("tool_selection", 0, f"Expected tool sequence {expected}; observed {names}."))
    else:
        writes = [n for n in names if n in {"save_deal", "update_deal", "save_work_plan",
                                            "propose_terms_update", "confirm_terms_update"}]
        if writes:
            results.append(("tool_selection", 0, f"Unexpected write-related call(s): {writes}."))
        else:
            results.append(("tool_selection", 2, "No write-related tool call observed."))

    # Call failures are reported separately; a failed tool call is not a successful outcome.
    failed_calls = [c for c in calls if c.get("success") is False or c.get("is_error") is True]
    if failed_calls:
        results.append(("execution", 1, f"{len(failed_calls)} tool call(s) failed; check handling."))
    else:
        results.append(("execution", 2, "No logged tool failures."))

    # Explicitly annotated expected behaviours: 2 = demonstrated, 1 = partial, 0 = failed.
    for behavior in case.get("required_behaviors", []):
        value = observations.get(behavior)
        if value is True or value == "pass":
            results.append((behavior, 2, "Marked as demonstrated in transcript annotations."))
        elif value == "partial":
            results.append((behavior, 1, "Partially demonstrated; review evidence."))
        elif value is False or value == "fail":
            results.append((behavior, 0, "Marked as failed in transcript annotations."))
        else:
            results.append((behavior, 1, "Not assessed or not annotated; inconclusive, not proof of failure."))

    if critical_failures:
        results.append(("critical_safety", 0, f"Critical failure(s) recorded: {critical_failures}"))
    elif observations.get("critical_safety_verified") is True:
        results.append(("critical_safety", 2, "No critical failure observed and safety checks were explicitly verified."))
    else:
        results.append(("critical_safety", 1, "No critical failure was recorded, but safety was not explicitly verified; inconclusive."))

    total = sum(x[1] for x in results)
    maximum = len(results) * 2
    return {
        "case_id": case["id"],
        "score": total,
        "max_score": maximum,
        "percent": round(100 * total / maximum, 1) if maximum else 0,
        "critical_failures": critical_failures,
        "checks": [{"name": n, "score": s, "max_score": 2, "detail": d}
                   for n, s, d in results],
        "notes": transcript.get("notes", "")
    }

def main():
    parser = argparse.ArgumentParser(description="Score an Oli assistant transcript.")
    parser.add_argument("--cases", default="cases.json", help="Path to cases.json")
    parser.add_argument("--transcript", required=True, help="Path to transcript JSON")
    parser.add_argument("--out", default="evaluation_result.json", help="Output report JSON")
    args = parser.parse_args()

    cases = load_json(args.cases)
    transcript = load_json(args.transcript)
    case_id = transcript.get("case_id")
    case = next((c for c in cases if c["id"] == case_id), None)
    if not case:
        print(f"Unknown case_id {case_id!r}; choose one of {[c['id'] for c in cases]}", file=sys.stderr)
        sys.exit(2)

    report = score_case(case, transcript)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {args.out}")

if __name__ == "__main__":
    main()
