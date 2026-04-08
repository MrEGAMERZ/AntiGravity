"""
AntiGravity — inference.py (Baseline Agent)
=============================================
MANDATORY COMPLIANCE:
- API_BASE_URL    : LLM endpoint (e.g. https://api.groq.com/openai/v1)
- MODEL_NAME      : Model identifier (e.g. llama-3.3-70b-versatile)
- HF_TOKEN        : API Key (Hugging Face / Groq / OpenAI)
- LOCAL_IMAGE_NAME: Name of the local Docker image (if using from_docker_image())

STDOUT FORMAT — strictly required:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import json
import re
import sys
import httpx
from openai import OpenAI

# ─── Environment Configuration ────────────────────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = HF_TOKEN

# Target environment URL
OPENENV_URL = os.getenv("OPENENV_URL", "https://mregamerz-antigravity.hf.space").rstrip("/")

if not API_KEY:
    print("WARNING: HF_TOKEN not set. LLM calls may fail.", file=sys.stderr)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "sk-no-key-set")


# ─── Structured Log Helpers ───────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

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
        return ""

def _post(path: str, body: dict) -> dict:
    """HTTP POST to the OpenEnv environment."""
    r = httpx.post(f"{OPENENV_URL}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()

def _extract_json(raw: str) -> dict:
    """Extracts JSON from CoT thinking or markdown blocks."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


# ─── Task Runners ─────────────────────────────────────────────────────────────

def run_easy() -> tuple[float, int]:
    """Task 1: Single Email Labeling."""
    obs = _post("/reset", {"task_level": "easy"})
    email = obs["emails"][0]

    system = (
        "Classify the email as: spam, promo, newsletter, or important. "
        "Think step-by-step then return ONLY this JSON: "
        '{"action_type": "label", "labels": {"<id>": "<category>"}}'
    )
    user = f"ID: {email['id']}\nSubject: {email['subject']}\nBody: {email['body']}"

    raw = _llm_call(system, user)
    try:
        action_dict = _extract_json(raw)
    except Exception as e:
        action_dict = {"action_type": "label", "labels": {email["id"]: "important"}}

    action_str = f"label({email['id']})"
    res = _post("/step", action_dict)
    reward = float(res.get("reward", 0.05))
    done = bool(res.get("done", True))

    step_count = 1
    log_step(step=step_count, action=action_str, reward=reward, done=done, error=None)
    return reward, step_count


def run_medium() -> tuple[float, int]:
    """Task 2: Inbox Priority Ranking (Kendall's Tau)."""
    obs = _post("/reset", {"task_level": "medium"})
    emails = obs["emails"]

    email_str = "\n".join([
        f"- ID:{e['id']} | Sent:{e['timestamp']} | From:{e['sender']} | Subject:{e['subject']}"
        for e in emails
    ])

    system = (
        "You are an expert email prioritizer.\n"
        "Rank these emails from MOST to LEAST urgent using this strict hierarchy:\n"
        "  1. important (action required, deadlines, from real people)\n"
        "  2. newsletter (informational, no action needed)\n"
        "  3. promo (marketing, offers, discounts)\n"
        "  4. spam (unsolicited, suspicious, irrelevant)\n\n"
        "Within the same category, rank strictly by Sent date (OLDEST timestamps FIRST).\n"
        "CRITICAL: The 'ranking' list must contain ONLY the alphanumeric IDs provided.\n"
        "Return ONLY valid JSON:\n"
        '{"action_type": "rank", "ranking": ["id1", "id2", ...]}'
    )
    raw = _llm_call(system, f"Rank these emails:\n{email_str}")
    try:
        action_dict = _extract_json(raw)
    except Exception:
        action_dict = {"action_type": "rank", "ranking": [e["id"] for e in emails]}

    action_str = f"rank({len(emails)}_emails)"
    res = _post("/step", action_dict)
    reward = float(res.get("reward", 0.05))
    done = bool(res.get("done", True))

    step_count = 1
    log_step(step=step_count, action=action_str, reward=reward, done=done, error=None)
    return reward, step_count


def run_hard() -> tuple[float, int]:
    """Task 3: Full Triage (Label + Urgent ID + Reply)."""
    obs = _post("/reset", {"task_level": "hard"})
    emails = obs["emails"]

    email_blocks = "\n\n".join([
        f"ID: {e['id']}\nSent: {e['timestamp']}\nFrom: {e['sender']}\nSubject: {e['subject']}\nBody: {e['body']}"
        for e in emails
    ])

    system = (
        "You are a professional executive assistant triaging an inbox.\n\n"
        "STEP 1 — LABEL every email as exactly one of: spam, promo, newsletter, important\n"
        "STEP 2 — URGENT ID: pick the ONE email that needs an immediate reply (must be 'important').\n"
        "STEP 3 — REPLY: Write a professional reply (20-60 words). Use words like: "
        "received, understood, will handle, on it, follow up. No URLs, no ALL CAPS.\n\n"
        "Return ONLY this JSON:\n"
        '{"action_type": "triage", "labels": {"<id>": "<category>", ...}, "urgent_id": "<id>", "reply_text": "<text>"}'
    )

    raw = _llm_call(system, f"Here are the emails:\n\n{email_blocks}")
    try:
        action_dict = _extract_json(raw)
    except Exception:
        first_id = emails[0]["id"] if emails else "unknown"
        action_dict = {
            "action_type": "triage",
            "labels": {e["id"]: "important" for e in emails},
            "urgent_id": first_id,
            "reply_text": "Thank you for reaching out. I have received your message and will handle it right away. I will follow up shortly."
        }

    action_str = f"triage({len(emails)}_emails)"
    res = _post("/step", action_dict)
    reward = float(res.get("reward", 0.05))
    done = bool(res.get("done", True))

    step_count = 1
    log_step(step=step_count, action=action_str, reward=reward, done=done, error=None)
    return reward, step_count


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    task_mappings = [
        ("easy_label", run_easy),
        ("medium_rank", run_medium),
        ("hard_triage", run_hard)
    ]

    for task_id, runner in task_mappings:
        log_start(task=task_id, env="antigravity", model=MODEL_NAME)
        task_rewards = []
        try:
            reward, steps = runner()
            task_rewards.append(reward)
            success = reward > 0.5
            log_end(success=success, steps=steps, score=reward, rewards=task_rewards)
        except Exception as e:
            log_step(step=1, action=f"{task_id}_failed", reward=0.05, done=True, error=str(e)[:80])
            log_end(success=False, steps=1, score=0.05, rewards=[0.05])
        print("")  # Add newline separator between tasks


if __name__ == "__main__":
    main()
