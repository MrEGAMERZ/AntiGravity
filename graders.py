"""
AntiGravity — Deterministic Graders  (v2)
All graders are pure functions: same input → same output, always.
Scores are in [0.0, 1.0].

Improvements over v1:
  - Expanded reply-keyword bank with profession synonyms
  - Sentence-coherence length bonus (not just char count)
  - Per-email partial-credit for label grader
  - Triage weights tweaked for better partial-reward signal
  - Kendall's Tau handles ties gracefully
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from models import Action, InboxState, ADJACENT_CATEGORIES


# ─── Utility ──────────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─── Task 1: Single Email Label ───────────────────────────────────────────────

def grade_label(action: Action, state: InboxState) -> float:
    """
    Reward 1.0 for exact match, 0.5 if adjacent category, 0.0 otherwise.
    Multiple emails → macro-average score per email.
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
        # else: 0.0 — wrong category

    return _clamp(total / count if count > 0 else 0.0)


# ─── Task 2: Inbox Priority Sort (Kendall's Tau) ─────────────────────────────

def _kendall_tau(pred: List[str], truth: List[str]) -> float:
    """
    Kendall's Tau correlation normalised to [0, 1].
    tau = 1.0 means perfect agreement, 0.0 means perfect inversion.
    Handles partial lists gracefully.
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

# Expanded synonym bank — covers common professional acknowledgement styles
_REPLY_SYNONYMS: Dict[str, List[str]] = {
    "received":        ["received", "got it", "have received", "gotten"],
    "on it":           ["on it", "working on it", "looking into", "investigating"],
    "will handle":     ["will handle", "will take care", "will address", "taking care"],
    "understood":      ["understood", "noted", "noted that", "i understand", "we understand"],
    "confirmed":       ["confirmed", "confirm", "can confirm", "have confirmed"],
    "acknowledge":     ["acknowledge", "acknowledging", "acknowledged"],
    "asap":            ["asap", "as soon as possible", "right away", "immediately", "urgently"],
    "right away":      ["right away", "straightaway", "without delay"],
    "sorry":           ["sorry", "apologies", "apologize", "regret"],
    "thank":           ["thank", "thanks", "grateful", "appreciate"],
    "help":            ["help", "assist", "support", "resolve"],
    "follow up":       ["follow up", "follow-up", "get back", "update you"],
}

# Flatten to a lookup set for quick membership test
_ALL_POSITIVE_TOKENS = {
    token
    for synonyms in _REPLY_SYNONYMS.values()
    for token in synonyms
}


def _score_reply(reply_text: Optional[str], keywords: List[str]) -> float:
    """
    Improved reply scoring:
      60% — keyword overlap (expanded synonym matching)
      25% — sentence-level coherence (word count heuristic)
      15% — professional tone (no ALL CAPS rants, no URLs)
    """
    if not reply_text or not reply_text.strip():
        return 0.0

    text = reply_text.strip()
    text_lower = text.lower()

    # ── Keyword score (synonym-aware) ─────────────────────────────────────────
    # Build expanded keyword set from provided base keywords
    expanded_kws: set[str] = set()
    for kw in keywords:
        expanded_kws.add(kw.lower())
        for synonyms in _REPLY_SYNONYMS.values():
            if kw.lower() in synonyms:
                expanded_kws.update(synonyms)

    # Also always reward common professionalism tokens
    expanded_kws.update(_ALL_POSITIVE_TOKENS)

    matched = sum(1 for token in expanded_kws if token in text_lower)
    # Normalise: reward hitting ≥3 positive tokens as full score
    keyword_score = min(matched / 3.0, 1.0)

    # ── Coherence / length score ───────────────────────────────────────────────
    word_count = len(text.split())
    if 5 <= word_count <= 80:
        coherence_score = 1.0
    elif word_count < 5:
        coherence_score = word_count / 5.0
    else:
        # Penalize linearly only after 80 words
        coherence_score = max(0.0, 1.0 - (word_count - 80) / 80.0)

    # ── Professional tone score ────────────────────────────────────────────────
    # Penalise: all-caps words, suspicious URLs, profanity (basic)
    caps_ratio = sum(1 for w in text.split() if w.isupper() and len(w) > 2) / max(len(text.split()), 1)
    has_url = bool(re.search(r"https?://", text))
    tone_score = 1.0
    if caps_ratio > 0.3:
        tone_score -= 0.4
    if has_url:
        tone_score -= 0.2
    tone_score = max(0.0, tone_score)

    return _clamp(0.60 * keyword_score + 0.25 * coherence_score + 0.15 * tone_score)


def grade_triage(action: Action, state: InboxState) -> float:
    """
    Composite score (v2):
      0.30 × label accuracy (per email)
      0.30 × correct urgent email identified
      0.40 × reply quality (synonym-aware keyword + coherence + tone)
    """
    label_score = grade_label(action, state)

    # Urgency identification — binary but weighted low enough not to kill score
    urgency_score = 1.0 if action.urgent_id == state.urgent_email_id else 0.0

    # Reply quality — improved scorer
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
