"""
AntiGravity — Pydantic v2 models
Defines the typed data structures for the OpenEnv Email Triage environment.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ─── Email object ────────────────────────────────────────────────────────────

import random as _random

class Email(BaseModel):
    id: str = Field(default_factory=lambda: "".join(_random.choices("0123456789abcdef", k=8)))

    sender: str
    subject: str
    body: str
    timestamp: str                          # ISO-8601
    has_attachment: bool = False


# ─── Observation (what the agent sees) ───────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_level: Literal["easy", "medium", "hard"]
    emails: List[Email]
    step_count: int = 0
    instructions: str                       # natural-language goal
    max_steps: int = 3


# ─── Action (what the agent can do) ──────────────────────────────────────────

class Action(BaseModel):
    action_type: Literal["label", "rank", "triage"]
    # Task 1 – single label
    labels: Optional[Dict[str, str]] = None   # email_id → category
    # Task 2 – ordered ranking
    ranking: Optional[List[str]] = None       # email IDs in priority order
    # Task 3 – full triage
    reply_text: Optional[str] = None          # draft reply body
    urgent_id: Optional[str] = None           # ID of urgent email


# ─── Step result ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool = False
    info: Dict = Field(default_factory=dict)


# ─── State (internal snapshot) ───────────────────────────────────────────────

class InboxState(BaseModel):
    task_id: str
    task_level: Literal["easy", "medium", "hard"]
    emails: List[Email]
    ground_truth_labels: Dict[str, str]       # email_id → category
    ground_truth_ranking: List[str]           # ordered email IDs (most urgent first)
    urgent_email_id: str                      # the one that needs a reply
    expected_reply_keywords: List[str]        # keywords grader checks for
    step_count: int = 0
    done: bool = False
    cumulative_reward: float = 0.0


# ─── Category constants ───────────────────────────────────────────────────────

VALID_LABELS = {"spam", "promo", "important", "newsletter"}

ADJACENT_CATEGORIES = {
    "promo": {"newsletter"},
    "newsletter": {"promo"},
    "spam": set(),
    "important": set(),
}
