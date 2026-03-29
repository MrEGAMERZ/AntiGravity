"""
AntiGravity — inference.py
Baseline inference script that runs an AI agent against all 3 tasks.
Uses Groq's free API (OpenAI-compatible) to drive the agent.

Required env vars:
  API_BASE_URL   e.g. https://your-space.hf.space  (or http://localhost:7860)
  MODEL_NAME     e.g. llama-3.3-70b-versatile  (default)
  GROQ_API_KEY   your free Groq API key (console.groq.com)
  HF_TOKEN       (optional) for HF Spaces auth
"""
from __future__ import annotations

import json
import os
import sys
import time

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq is OpenAI-compatible — just swap the base_url
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{API_BASE_URL}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict:
    r = httpx.get(f"{API_BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _llm(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


# ─── Task runners ────────────────────────────────────────────────────────────

def run_easy() -> float:
    obs = _post("/reset", {"task_level": "easy"})
    email = obs["emails"][0]
    sys_prompt = (
        "You classify emails. Reply with ONLY a JSON object: "
        '{"action_type": "label", "labels": {"<email_id>": "<category>"}}\n'
        "Categories: spam, promo, important, newsletter"
    )
    user_msg = (
        f"Email ID: {email['id']}\n"
        f"From: {email['sender']}\n"
        f"Subject: {email['subject']}\n"
        f"Body:\n{email['body']}"
    )
    raw = _llm(sys_prompt, user_msg)
    action_dict = json.loads(raw.strip().strip("```json").strip("```"))
    result = _post("/step", action_dict)
    print(f"[EASY] reward={result['reward']} done={result['done']}")
    return result["reward"]


def run_medium() -> float:
    obs = _post("/reset", {"task_level": "medium"})
    emails = obs["emails"]
    email_list = "\n".join(
        f"{i+1}. ID={e['id']} | From={e['sender']} | Subject={e['subject']} | Time={e['timestamp']}"
        for i, e in enumerate(emails)
    )
    sys_prompt = (
        "You rank emails by urgency. Reply with ONLY a JSON object: "
        '{"action_type": "rank", "ranking": ["id1", "id2", ...]}\n'
        "Most urgent first. Consider: important > newsletter > promo > spam."
    )
    raw = _llm(sys_prompt, f"Emails:\n{email_list}")
    action_dict = json.loads(raw.strip().strip("```json").strip("```"))
    result = _post("/step", action_dict)
    print(f"[MEDIUM] reward={result['reward']} done={result['done']}")
    return result["reward"]


def run_hard() -> float:
    obs = _post("/reset", {"task_level": "hard"})
    emails = obs["emails"]
    email_details = "\n\n".join(
        f"ID={e['id']}\nFrom={e['sender']}\nSubject={e['subject']}\nBody={e['body'][:200]}"
        for e in emails
    )
    sys_prompt = (
        "You are an expert email triager. Reply with ONLY a JSON object:\n"
        '{"action_type": "triage", '
        '"labels": {"id": "category", ...}, '
        '"urgent_id": "id_of_urgent_email", '
        '"reply_text": "short professional reply"}\n'
        "Categories: spam, promo, important, newsletter"
    )
    raw = _llm(sys_prompt, f"Inbox:\n\n{email_details}")
    action_dict = json.loads(raw.strip().strip("```json").strip("```"))
    result = _post("/step", action_dict)
    print(f"[HARD] reward={result['reward']} done={result['done']}")
    return result["reward"]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"AntiGravity Inference — target: {API_BASE_URL}, model: {MODEL_NAME} (via Groq)")
    print("=" * 60)

    # Health check
    try:
        health = _get("/health")
        print(f"Health: {health}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {API_BASE_URL}: {e}")
        sys.exit(1)

    scores = {}
    for task, runner in [("easy", run_easy), ("medium", run_medium), ("hard", run_hard)]:
        try:
            score = runner()
            scores[task] = score
        except Exception as e:
            print(f"[{task.upper()}] FAILED: {e}")
            scores[task] = 0.0
        time.sleep(1)

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:8s}: {score:.4f}  {bar}")
    avg = sum(scores.values()) / 3
    print(f"  {'AVERAGE':8s}: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
