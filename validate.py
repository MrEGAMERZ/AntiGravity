"""
AntiGravity — Pre-Submission Validator
Run this before submitting to verify everything works end-to-end.

Usage:
    python3 validate.py [--url http://localhost:7860]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import httpx

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    icon = PASS if passed else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{suffix}")


def post(base: str, path: str, body: dict) -> dict:
    r = httpx.post(f"{base}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def get(base: str, path: str) -> dict:
    r = httpx.get(f"{base}{path}", timeout=15)
    r.raise_for_status()
    return r.json()


def run_validation(base_url: str) -> None:
    base = base_url.rstrip("/")
    print(f"\n{'='*60}")
    print(f"  AntiGravity Pre-Submission Validator")
    print(f"  Target: {base}")
    print(f"{'='*60}\n")

    # ── 1. Health Check ───────────────────────────────────────────
    print("[ 1/6 ] Health Check")
    try:
        h = get(base, "/health")
        check("GET /health returns 200", True, str(h))
        check("Status is 'ok'", h.get("status") == "ok")
    except Exception as e:
        check("GET /health", False, str(e))
        print("\n  ERROR: Cannot reach server. Aborting.\n")
        sys.exit(1)

    # ── 2. Root endpoint ──────────────────────────────────────────
    print("\n[ 2/6 ] Root Endpoint")
    try:
        root = get(base, "/")
        check("GET / returns 200", True)
        check("Has 'name' field", "name" in root)
        check("Has 'endpoints' field", "endpoints" in root)
    except Exception as e:
        check("GET /", False, str(e))

    # ── 3. Easy task ──────────────────────────────────────────────
    print("\n[ 3/6 ] Easy Task (Single Email Label)")
    try:
        obs = post(base, "/reset", {"task_level": "easy", "seed": 42})
        check("POST /reset easy returns 200", True)
        check("Observation has 'emails'", "emails" in obs)
        check("Observation has 'task_level'", obs.get("task_level") == "easy")
        check("Observation has 'instructions'", bool(obs.get("instructions")))
        check("Has exactly 1 email", len(obs["emails"]) == 1)

        email = obs["emails"][0]
        check("Email has id/sender/subject/body", all(k in email for k in ["id", "sender", "subject", "body"]))

        result = post(base, "/step", {
            "action_type": "label",
            "labels": {email["id"]: "important"},
        })
        check("POST /step returns 200", True)
        check("StepResult has 'reward'", "reward" in result)
        check("Reward in [0.0, 1.0]", 0.0 <= result["reward"] <= 1.0, f"reward={result['reward']}")
        check("StepResult has 'done'", "done" in result)
    except Exception as e:
        check("Easy task", False, str(e))

    # ── 4. Medium task ────────────────────────────────────────────
    print("\n[ 4/6 ] Medium Task (Inbox Priority Sort)")
    try:
        obs = post(base, "/reset", {"task_level": "medium", "seed": 42})
        check("POST /reset medium returns 200", True)
        check("Has 10 emails", len(obs["emails"]) == 10, f"got {len(obs['emails'])}")

        ids = [e["id"] for e in obs["emails"]]
        result = post(base, "/step", {
            "action_type": "rank",
            "ranking": ids,
        })
        check("POST /step rank returns 200", True)
        check("Reward in [0.0, 1.0]", 0.0 <= result["reward"] <= 1.0, f"reward={result['reward']}")
    except Exception as e:
        check("Medium task", False, str(e))

    # ── 5. Hard task ──────────────────────────────────────────────
    print("\n[ 5/6 ] Hard Task (Triage + Reply + Archive)")
    try:
        obs = post(base, "/reset", {"task_level": "hard", "seed": 42})
        check("POST /reset hard returns 200", True)
        check("Has emails", len(obs["emails"]) > 0)

        emails = obs["emails"]
        labels = {e["id"]: "important" for e in emails}
        result = post(base, "/step", {
            "action_type": "triage",
            "labels": labels,
            "urgent_id": emails[0]["id"],
            "reply_text": "Thank you for reaching out. I have received your message and will handle it right away.",
        })
        check("POST /step triage returns 200", True)
        check("Reward in [0.0, 1.0]", 0.0 <= result["reward"] <= 1.0, f"reward={result['reward']}")
    except Exception as e:
        check("Hard task", False, str(e))

    # ── 6. State endpoint + Determinism ───────────────────────────
    print("\n[ 6/6 ] State Endpoint & Determinism")
    try:
        state = get(base, "/state")
        check("GET /state returns 200", True)
        check("State has task fields", "task_level" in state)

        # Determinism check: same seed → same first email
        obs_a = post(base, "/reset", {"task_level": "easy", "seed": 99})
        obs_b = post(base, "/reset", {"task_level": "easy", "seed": 99})
        check("Deterministic with same seed", obs_a["emails"][0]["id"] == obs_b["emails"][0]["id"])
    except Exception as e:
        check("State / determinism", False, str(e))

    # ── Summary ───────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} checks passed")

    failed = [(n, d) for n, ok, d in results if not ok]
    if failed:
        print(f"\n  {FAIL} Failed checks:")
        for name, detail in failed:
            print(f"    - {name}" + (f": {detail}" if detail else ""))
        print(f"\n{'='*60}\n")
        sys.exit(1)
    else:
        print(f"\n  {PASS} ALL CHECKS PASSED — ready to submit! 🚀")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AntiGravity pre-submission validator")
    parser.add_argument("--url", default="http://localhost:7860", help="Environment base URL")
    args = parser.parse_args()
    run_validation(args.url)
