"""
AntiGravity — Deterministic Graders
All graders are pure functions: same input → same output, always.
Scores are in [0.0, 1.0].
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from models import Action, InboxState, ADJACENT_CATEGORIES


# ─── Utility ──────────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─── Task 1: Single Email Label ───────────────────────────────────────────────

def grade_label(action: Action, state: InboxState) -> float:
    """
    Reward 1.0 for exact match, 0.5 if adjacent category, 0.0 otherwise.
    Multiple emails → average score per email.
    """
    if not action.labels or not state.ground_truth_labels:
        return 0.0

    total = 0.0
    count = len(state.ground_truth_labels)

    for email_id, true_cat in state.ground_truth_labels.items():
        pred_cat = (action.labels or {}).get(email_id, "")
        if pred_cat == true_cat:
            total += 1.0
        elif pred_cat in ADJACENT_CATEGORIES.get(true_cat, set()):
            total += 0.5
        else:
            total += 0.0

    return _clamp(total / count if count > 0 else 0.0)


# ─── Task 2: Inbox Priority Sort (Kendall's Tau) ─────────────────────────────

def _kendall_tau(pred: List[str], truth: List[str]) -> float:
    """
    Kendall's Tau correlation normalised to [0, 1].
    tau = 1.0 means perfect agreement, 0.0 means perfect inversion.
    """
    n = len(truth)
    if n <= 1:
        return 1.0

    # Map item → rank in ground truth
    truth_rank = {item: i for i, item in enumerate(truth)}

    # Work only with items that appear in both lists
    pred_filtered = [p for p in pred if p in truth_rank]
    if not pred_filtered:
        return 0.0

    concordant = 0
    discordant = 0
    for i in range(len(pred_filtered)):
        for j in range(i + 1, len(pred_filtered)):
            a, b = pred_filtered[i], pred_filtered[j]
            pred_order = 1 if i < j else -1
            truth_order = 1 if truth_rank[a] < truth_rank[b] else -1
            if pred_order == truth_order:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0

    tau = (concordant - discordant) / total   # in [-1, 1]
    return _clamp((tau + 1) / 2)              # shift to [0, 1]


def grade_ranking(action: Action, state: InboxState) -> float:
    """Reward is Kendall's Tau correlation, normalised to [0, 1]."""
    if not action.ranking or not state.ground_truth_ranking:
        return 0.0
    return _kendall_tau(action.ranking, state.ground_truth_ranking)


# ─── Task 3: Triage + Reply + Archive ────────────────────────────────────────

def _score_reply(reply_text: Optional[str], keywords: List[str]) -> float:
    """Simple keyword-overlap + length heuristic."""
    if not reply_text:
        return 0.0

    text_lower = reply_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    keyword_score = min(matches / max(len(keywords), 1), 1.0)

    # Length bonus: replies between 20-300 chars get full score
    length = len(reply_text.strip())
    if 20 <= length <= 300:
        length_score = 1.0
    elif length < 20:
        length_score = length / 20
    else:
        length_score = max(0.0, 1.0 - (length - 300) / 300)

    return _clamp(0.6 * keyword_score + 0.4 * length_score)


def grade_triage(action: Action, state: InboxState) -> float:
    """
    Composite score:
      0.30 × label accuracy (per email)
      0.30 × correct urgent email identified
      0.40 × reply quality (keyword + length)
    """
    label_score = grade_label(action, state)

    # Urgency identification
    urgency_score = 1.0 if action.urgent_id == state.urgent_email_id else 0.0

    # Reply quality
    reply_score = _score_reply(action.reply_text, state.expected_reply_keywords)

    composite = (
        0.30 * label_score
        + 0.30 * urgency_score
        + 0.40 * reply_score
    )
    return _clamp(composite)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def compute_reward(action: Action, state: InboxState) -> float:
    if state.task_level == "easy":
        return grade_label(action, state)
    elif state.task_level == "medium":
        return grade_ranking(action, state)
    elif state.task_level == "hard":
        return grade_triage(action, state)
    return 0.0
