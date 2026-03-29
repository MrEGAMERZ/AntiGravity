"""
AntiGravity — inference.py  (v2: Chain-of-Thought Prompting)
Baseline inference script that runs an AI agent against all 3 tasks.
Uses Groq's free API (OpenAI-compatible) to drive the agent.

Required env vars:
  API_BASE_URL   e.g. https://mregamerz-antigravity.hf.space (or http://localhost:7860)
  MODEL_NAME     e.g. llama-3.3-70b-versatile  (default)
  GROQ_API_KEY   your free Groq API key (console.groq.com)
  HF_TOKEN       (optional) for HF Spaces auth
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
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
            {"role": "user",   "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a string (handles CoT reasoning prefix)."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Strip markdown code block
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find the first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in LLM response:\n{raw[:500]}")


# ─── Task runners ────────────────────────────────────────────────────────────

def run_easy() -> float:
    obs  = _post("/reset", {"task_level": "easy"})
    email = obs["emails"][0]

    sys_prompt = (
        "You are an expert email classifier. Your job is to assign a single label to an email.\n\n"
        "Available labels:\n"
        "  - spam       : unsolicited or fraudulent emails (fake prizes, phishing, scams)\n"
        "  - promo      : marketing, sales, discounts, product promotions\n"
        "  - newsletter : regular digests, updates, publications (not trying to sell something directly)\n"
        "  - important  : emails requiring action, urgent responses, meetings, contracts, incidents\n\n"
        "Think step by step:\n"
        "1. Who is the sender? Is the domain suspicious?\n"
        "2. What is the subject line signalling?\n"
        "3. Does the body require action from me?\n"
        "4. Choose the single best label.\n\n"
        "Output ONLY this JSON (no other text):\n"
        '{"action_type": "label", "labels": {"<email_id>": "<category>"}}'
    )
    user_msg = (
        f"Email ID: {email['id']}\n"
        f"From: {email['sender']}\n"
        f"Subject: {email['subject']}\n"
        f"Body:\n{email['body']}"
    )
    raw = _llm(sys_prompt, user_msg)
    action_dict = _extract_json(raw)
    result = _post("/step", action_dict)
    print(f"[EASY]   reward={result['reward']} done={result['done']}")
    return result["reward"]


def run_medium() -> float:
    obs    = _post("/reset", {"task_level": "medium"})
    emails = obs["emails"]

    email_list = "\n".join(
        f"{i+1}. ID={e['id']} | From={e['sender']} | Subject={e['subject']} | Sent={e['timestamp']}"
        for i, e in enumerate(emails)
    )

    sys_prompt = (
        "You are an expert email triage specialist. Rank emails from MOST to LEAST urgent.\n\n"
        "Urgency rules (apply in order):\n"
        "  1. IMPORTANT emails (action required, incidents, deadlines, contracts) — always highest\n"
        "  2. Within 'important': more recent = more urgent\n"
        "  3. NEWSLETTER: regular digests, lower priority\n"
        "  4. PROMO: marketing/sales, low priority\n"
        "  5. SPAM: fraudulent/unsolicited, always lowest\n\n"
        "Red flags for SPAM: suspicious domains (.xyz, .win), all-caps subjects, prize claims.\n"
        "Red flags for IMPORTANT: words like 'urgent', 'required', 'incident', 'deadline', 'overdue'.\n\n"
        "Think step by step — classify each email's category, then rank them.\n\n"
        "Output ONLY this JSON:\n"
        '{"action_type": "rank", "ranking": ["id1", "id2", ...]}'
    )
    raw = _llm(sys_prompt, f"Emails to rank:\n{email_list}")
    action_dict = _extract_json(raw)
    result = _post("/step", action_dict)
    print(f"[MEDIUM] reward={result['reward']} done={result['done']}")
    return result["reward"]


def run_hard() -> float:
    obs    = _post("/reset", {"task_level": "hard"})
    emails = obs["emails"]

    email_details = "\n\n".join(
        f"ID: {e['id']}\nFrom: {e['sender']}\nSubject: {e['subject']}\n"
        f"Sent: {e['timestamp']}\nHas Attachment: {e.get('has_attachment', False)}\n"
        f"Body:\n{e['body'][:300]}"
        for e in emails
    )

    sys_prompt = (
        "You are a senior executive assistant performing full inbox triage.\n\n"
        "You must:\n"
        "  1. Label EVERY email: spam | promo | important | newsletter\n"
        "  2. Identify the ONE email needing an urgent reply (urgent_id)\n"
        "  3. Write a concise, professional reply (20–80 words)\n\n"
        "Label definitions:\n"
        "  spam       — phishing, scams, unsolicited junk\n"
        "  promo      — sales, discounts, marketing\n"
        "  newsletter — digests, updates, publications\n"
        "  important  — action required, incidents, deadlines, contracts\n\n"
        "For the reply:\n"
        "  - Address the sender professionally\n"
        "  - Acknowledge receipt and your intended action\n"
        "  - Keep it under 80 words, no URLs\n"
        "  - Use professional language: 'noted', 'will handle', 'acknowledged', 'right away'\n\n"
        "Think step by step. Then output ONLY this JSON (no other text):\n"
        "{\n"
        '  "action_type": "triage",\n'
        '  "labels": {"<id>": "<category>", ...},\n'
        '  "urgent_id": "<most_urgent_email_id>",\n'
        '  "reply_text": "<your professional reply>"\n'
        "}"
    )
    raw = _llm(sys_prompt, f"Full inbox ({len(emails)} emails):\n\n{email_details}")
    action_dict = _extract_json(raw)
    result = _post("/step", action_dict)
    print(f"[HARD]   reward={result['reward']} done={result['done']}")
    return result["reward"]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"AntiGravity Inference (v2 CoT) — target: {API_BASE_URL}, model: {MODEL_NAME} (via Groq)")
    print("=" * 65)

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

    print("\n" + "=" * 65)
    print("FINAL SCORES")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:8s}: {score:.4f}  {bar}")
    avg = sum(scores.values()) / 3
    print(f"  {'AVERAGE':8s}: {avg:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
