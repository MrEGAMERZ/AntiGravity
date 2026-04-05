"""
AntiGravity — inference.py (Baseline Agent)
=========================================
MANDATORY COMPLIANCE:
- API_BASE_URL : LLM endpoint (e.g. https://api.groq.com/openai/v1)
- MODEL_NAME   : Model identifier (e.g. llama-3.3-70b-versatile)
- HF_TOKEN     : API Key (Hugging Face / Groq / OpenAI)

This script runs the AntiGravity email triage agent using Chain-of-Thought
reasoning to solve Easy, Medium, and Hard tasks.
"""

import os
import json
import re
import sys
import time
import httpx
from openai import OpenAI

# ─── Environment Configuration ───────────────────────────────────────────────

# LLM Config (Mandatory names as per hackathon spec)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = HF_TOKEN

# AntiGravity Environment URL (The target to benchmark)
# Default points to the live deployment for convenience
OPENENV_URL  = os.getenv("OPENENV_URL", "https://mregamerz-antigravity.hf.space").rstrip("/")

if not API_KEY:
    print("WARNING: HF_TOKEN (API_KEY) not set. Inference may fail.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _llm_call(system_prompt: str, user_prompt: str) -> str:
    """Standardized OpenAI client call as per mandatory requirements."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        print(f"ERROR calling LLM: {e}")
        return ""

def _post(path: str, body: dict) -> dict:
    """Helper for OpenEnv API interaction."""
    r = httpx.post(f"{OPENENV_URL}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()

def _extract_json(raw: str) -> dict:
    """Extracts JSON from CoT thinking or markdown blocks."""
    try:
        # Standard cleaning
        text = raw.strip()
        # Look for JSON block in markdown
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
        # Fallback: look for any { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Could not extract JSON: {e}\nRaw start: {raw[:100]}...")


# ─── Task Implementation ──────────────────────────────────────────────────────

def run_easy() -> float:
    """Task: Single Email Labeling."""
    obs = _post("/reset", {"task_level": "easy"})
    email = obs["emails"][0]
    
    sys = (
        "Classify the email as: spam, promo, newsletter, or important. "
        "Think step-by-step then return ONLY this JSON: "
        '{"action_type": "label", "labels": {"<id>": "<category>"}}'
    )
    user = f"ID: {email['id']}\nSubject: {email['subject']}\nBody: {email['body']}"
    
    raw = _llm_call(sys, user)
    res = _post("/step", _extract_json(raw))
    print(f"[STEP] Easy run complete. Reward: {res['reward']:.2f}")
    return res["reward"]

def run_medium() -> float:
    """Task: Inbox Priority Ranking (Kendall's Tau)."""
    obs = _post("/reset", {"task_level": "medium"})
    emails = obs["emails"]
    
    # Give the model full subject + sender + timestamp
    email_str = "\n".join([
        f"- ID:{e['id']} | Sent:{e['timestamp']} | From:{e['sender']} | Subject:{e['subject']}"
        for e in emails
    ])

    sys = (
        "You are an expert email prioritizer.\n"
        "Rank these emails from MOST to LEAST urgent using this strict hierarchy:\n"
        "  1. important (action required, deadlines, from real people)\n"
        "  2. newsletter (informational, no action needed)\n"
        "  3. promo (marketing, offers, discounts)\n"
        "  4. spam (unsolicited, suspicious, irrelevant)\n\n"
        "Within the same category, rank strictly by Sent date (OLDEST timestamps FIRST).\n"
        "CRITICAL: The 'ranking' list must contain ONLY the alphanumeric IDs provided.\n"
        "Do NOT include email addresses, explanations, or quotes inside the list.\n"
        "Return ONLY valid JSON:\n"
        '{"action_type": "rank", "ranking": ["id1", "id2", ...]}'
    )
    raw = _llm_call(sys, f"Rank these emails:\n{email_str}")
    res = _post("/step", _extract_json(raw))
    print(f"[STEP] Medium run complete. Reward: {res['reward']:.2f}")
    return res["reward"]


def run_hard() -> float:
    """Task: Full Triage (Label + Urgent ID + Reply)."""
    obs = _post("/reset", {"task_level": "hard"})
    emails = obs["emails"]

    # Build full email context — include timestamp
    email_blocks = "\n\n".join([
        f"ID: {e['id']}\n"
        f"Sent: {e['timestamp']}\n"
        f"From: {e['sender']}\n"
        f"Subject: {e['subject']}\n"
        f"Body: {e['body']}"
        for e in emails
    ])

    sys = (
        "You are a professional executive assistant triaging an inbox.\n\n"
        "STEP 1 — LABEL every email as exactly one of: spam, promo, newsletter, important\n"
        "  - important: requires action, has deadline, from a real person or client\n"
        "  - newsletter: informational digest, no action needed\n"
        "  - promo: discount, offer, marketing email\n"
        "  - spam: unsolicited, suspicious, irrelevant\n\n"
        "STEP 2 — URGENT ID: pick the ONE email that needs an immediate reply.\n"
        "  - Must be labeled 'important'. Use ONLY the alphanumeric ID.\n"
        "  - If there are multiple 'important' emails, pick the OLDEST ONE (earliest Sent date).\n\n"
        "STEP 3 — REPLY: Write a professional reply to that urgent email.\n"
        "  - 20 to 60 words, no more. Address them professionally.\n"
        "  - Acknowledge receipt, confirm you are on it, give a timeframe\n"
        "  - Use words like: received, understood, will handle, on it, follow up\n"
        "  - No URLs, no ALL CAPS, no filler\n\n"
        "Return ONLY this JSON, no explanation:\n"
        "{\n"
        '  "action_type": "triage",\n'
        '  "labels": {"<id>": "<category>", ...},\n'
        '  "urgent_id": "<id>",\n'
        '  "reply_text": "<text>"\n'
        "}"
    )

    raw = _llm_call(sys, f"Here are the emails:\n\n{email_blocks}")
    res = _post("/step", _extract_json(raw))
    print(f"[STEP] Hard run complete. Reward: {res['reward']:.2f}")
    return res["reward"]


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    print(f"[START] AntiGravity Baseline — Endpoint: {OPENENV_URL}")
    print(f"[START] Using Model: {MODEL_NAME}")
    
    scores = {}
    for task, runner in [("easy", run_easy), ("medium", run_medium), ("hard", run_hard)]:
        try:
            scores[task] = runner()
        except Exception as e:
            print(f"[STEP] {task.upper()} Failed: {e}")
            scores[task] = 0.0
            
    avg = sum(scores.values()) / 3
    print(f"[END] FINAL MEAN SCORE: {avg:.4f}")

if __name__ == "__main__":
    main()
