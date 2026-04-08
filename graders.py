"""
AntiGravity — Deterministic Graders  (v3)
All graders are pure functions: same input → same output, always.
Scores are STRICTLY in (0, 1) — never 0.0 or 1.0 exactly.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from models import Action, InboxState, ADJACENT_CATEGORIES


# ─── Utility ──────────────────────────────────────────────────────────────────

def _strict(v: float, task_id: str = "default") -> float:
    """
    Maps any float to the open interval (0.01, 0.99).
    Uses a deterministic task_id-based offset so it remains a pure function.
    EVERY return value from any grader MUST go through this function.
    """
    h = int(hashlib.md5(task_id.encode()).hexdigest(), 16)
    # Deterministic small jiggle in [0.01, 0.02]
    jiggle = 0.01 + (h % 100) / 10000.0
    lo = jiggle
    hi = 1.0 - jiggle
    return round(max(lo, min(hi, float(v))), 4)


# ─── Task 1: Single Email Label ───────────────────────────────────────────────

def grade_label(action: Action, state: InboxState) -> float:
    """
    Reward for exact match = 0.98, adjacent category = 0.49, wrong = 0.01.
    Multiple emails → macro-average score per email.
    All return paths go through _strict().
    """
    if not action.labels or not state.ground_truth_labels:
        return _strict(0.05, state.task_id)  # no submission — minimal score, not 0.0

    total = 0.0
    count = len(state.ground_truth_labels)

    for email_id, true_cat in state.ground_truth_labels.items():
        pred_cat = (action.labels or {}).get(email_id, "")
        if pred_cat == true_cat:
            total += 0.98   # correct — never 1.0
        elif pred_cat in ADJACENT_CATEGORIES.get(true_cat, set()):
            total += 0.49   # adjacent category partial credit
        else:
            total += 0.05   # wrong — never 0.0

    raw = total / count if count > 0 else 0.05
    return _strict(raw, state.task_id)


# ─── Task 2: Inbox Priority Sort (Kendall's Tau) ─────────────────────────────

def _kendall_tau_raw(pred: List[str], truth: List[str]) -> float:
    """
    Kendall's Tau correlation normalised to [0, 1].
    Returns a raw float — caller must pass through _strict().
    """
    n = len(truth)
    if n <= 1:
        return 0.97  # trivially sorted, return near-perfect but not 1.0

    truth_rank = {item: i for i, item in enumerate(truth)}
    pred_filtered = [p for p in pred if p in truth_rank]

    if not pred_filtered:
        return 0.05  # nothing matched, minimal score not 0.0

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

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.5

    tau = (concordant - discordant) / total_pairs  # in [-1, 1]
    return (tau + 1) / 2                           # shift to [0, 1]


def grade_ranking(action: Action, state: InboxState) -> float:
    """Reward is Kendall's Tau correlation, strictly in (0, 1)."""
    if not action.ranking or not state.ground_truth_ranking:
        return _strict(0.05, state.task_id)
    raw = _kendall_tau_raw(action.ranking, state.ground_truth_ranking)
    return _strict(raw, state.task_id)


# ─── Task 3: Triage + Reply + Archive ────────────────────────────────────────

_REPLY_SYNONYMS: Dict[str, List[str]] = {
    "received":    ["received", "got it", "have received", "gotten"],
    "on it":       ["on it", "working on it", "looking into", "investigating"],
    "will handle": ["will handle", "will take care", "will address", "taking care"],
    "understood":  ["understood", "noted", "noted that", "i understand", "we understand"],
    "confirmed":   ["confirmed", "confirm", "can confirm", "have confirmed"],
    "acknowledge": ["acknowledge", "acknowledging", "acknowledged"],
    "asap":        ["asap", "as soon as possible", "right away", "immediately", "urgently"],
    "right away":  ["right away", "straightaway", "without delay"],
    "sorry":       ["sorry", "apologies", "apologize", "regret"],
    "thank":       ["thank", "thanks", "grateful", "appreciate"],
    "help":        ["help", "assist", "support", "resolve"],
    "follow up":   ["follow up", "follow-up", "get back", "update you"],
}

_ALL_POSITIVE_TOKENS = {
    token
    for synonyms in _REPLY_SYNONYMS.values()
    for token in synonyms
}


def _score_reply_raw(reply_text: Optional[str], keywords: List[str]) -> float:
    """
    Reply scoring — returns raw float in approximately [0, 1].
    Caller must pass through _strict().
    """
    if not reply_text or not reply_text.strip():
        return 0.05  # empty reply — never 0.0

    text = reply_text.strip()
    text_lower = text.lower()

    # Keyword score (synonym-aware)
    expanded_kws: set[str] = set()
    for kw in keywords:
        expanded_kws.add(kw.lower())
        for synonyms in _REPLY_SYNONYMS.values():
            if kw.lower() in synonyms:
                expanded_kws.update(synonyms)
    expanded_kws.update(_ALL_POSITIVE_TOKENS)

    matched = sum(1 for token in expanded_kws if token in text_lower)
    keyword_score = min(matched / 3.0, 0.97)  # cap at 0.97, not 1.0

    # Coherence / length score
    word_count = len(text.split())
    if 5 <= word_count <= 80:
        coherence_score = 0.97
    elif word_count < 5:
        coherence_score = max(0.05, word_count / 5.0 * 0.97)
    else:
        coherence_score = max(0.05, 0.97 - (word_count - 80) / 80.0)

    # Professional tone score
    caps_ratio = sum(1 for w in text.split() if w.isupper() and len(w) > 2) / max(len(text.split()), 1)
    has_url = bool(re.search(r"https?://", text))
    tone_score = 0.97
    if caps_ratio > 0.3:
        tone_score -= 0.4
    if has_url:
        tone_score -= 0.2
    tone_score = max(0.05, tone_score)

    return 0.60 * keyword_score + 0.25 * coherence_score + 0.15 * tone_score


def grade_triage(action: Action, state: InboxState) -> float:
    """
    Composite score:
      0.30 × label accuracy
      0.30 × correct urgent email identified
      0.40 × reply quality
    All component scores and final score pass through _strict().
    """
    label_score = grade_label(action, state)

    # Urgency: use 0.97/0.05 instead of 1.0/0.0
    urgency_raw = 0.97 if action.urgent_id == state.urgent_email_id else 0.05
    urgency_score = _strict(urgency_raw, state.task_id)

    reply_raw = _score_reply_raw(action.reply_text, state.expected_reply_keywords)
    reply_score = _strict(reply_raw, state.task_id)

    composite = (
        0.30 * label_score
        + 0.30 * urgency_score
        + 0.40 * reply_score
    )
    return _strict(composite, state.task_id)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def compute_reward(action: Action, state: InboxState) -> float:
    """Route to the correct grader. All paths return strictly (0, 1)."""
    if state.task_level == "easy":
        return grade_label(action, state)
    elif state.task_level == "medium":
        return grade_ranking(action, state)
    elif state.task_level == "hard":
        return grade_triage(action, state)
    # Fallback: never 0.0
    return _strict(0.1, "fallback")
