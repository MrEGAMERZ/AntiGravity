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
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

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
            temperature=0,
            max_tokens=800,
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
        # Strip markdown syntax
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
        # Find first { block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(cleaned)
    except Exception:
        raise ValueError(f"Could not extract JSON from model response.\nRaw: {raw[:200]}...")


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
    print(f"[EASY]   Reward: {res['reward']:.2f}")
    return res["reward"]

def run_medium() -> float:
    """Task: Inbox Priority Ranking (Kendall's Tau)."""
    obs = _post("/reset", {"task_level": "medium"})
    email_str = "\n".join([f"- {e['id']}: {e['subject']}" for e in obs["emails"]])
    
    sys = (
        "Rank emails by priority (important > newsletter > promo > spam). "
        "Return ONLY this JSON: "
        '{"action_type": "rank", "ranking": ["id1", "id2", ...]}'
    )
    raw = _llm_call(sys, f"Emails:\n{email_str}")
    res = _post("/step", _extract_json(raw))
    print(f"[MEDIUM] Reward: {res['reward']:.2f}")
    return res["reward"]

def run_hard() -> float:
    """Task: Full Triage (Label + Urgent ID + Reply)."""
    obs = _post("/reset", {"task_level": "hard"})
    emails = obs["emails"]
    
    sys = (
        "Identify the ONE urgent email, write a professional reply (<80 words), and label all emails. "
        "Return ONLY this JSON:\n"
        '{"action_type": "triage", "labels": {...}, "urgent_id": "...", "reply_text": "..."}'
    )
    user = "\n\n".join([f"ID: {e['id']}\nFrom: {e['sender']}\nSub: {e['subject']}\nBody: {e['body'][:200]}" for e in emails])
    
    raw = _llm_call(sys, user)
    res = _post("/step", _extract_json(raw))
    print(f"[HARD]   Reward: {res['reward']:.2f}")
    return res["reward"]


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    print(f"AntiGravity Baseline — Endpoint: {OPENENV_URL}")
    print(f"Using Model: {MODEL_NAME}")
    print("-" * 50)
    
    scores = {}
    for task, runner in [("easy", run_easy), ("medium", run_medium), ("hard", run_hard)]:
        try:
            scores[task] = runner()
        except Exception as e:
            print(f"[{task.upper()}] Failed: {e}")
            scores[task] = 0.0
            
    avg = sum(scores.values()) / 3
    print("-" * 50)
    print(f"FINAL MEAN SCORE: {avg:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
