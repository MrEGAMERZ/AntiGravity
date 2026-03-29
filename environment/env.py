"""
AntiGravity — Core OpenEnv-compatible Environment
Implements reset(), step(), state() per the OpenEnv contract.
"""
from __future__ import annotations

import uuid
import hashlib
from typing import Optional

from models import (
    Action, Email, InboxState, Observation, StepResult,
)
from data_gen import generate_easy_inbox, generate_medium_inbox, generate_hard_inbox
from graders import compute_reward


_INSTRUCTIONS = {
    "easy": (
        "You are given ONE email. Classify it with a single label: "
        "'spam', 'promo', 'important', or 'newsletter'. "
        "Submit action_type='label' with labels={email_id: category}."
    ),
    "medium": (
        "You are given 10 emails. Rank them by urgency from most to least urgent "
        "(index 0 = most urgent). Consider sender tone, keywords, time-sensitivity. "
        "Submit action_type='rank' with ranking=[email_id, ...]."
    ),
    "hard": (
        "You are given a full inbox. You must: "
        "(1) label every email as spam/promo/important/newsletter, "
        "(2) identify the single email that requires an urgent reply (urgent_id), "
        "(3) draft a concise, professional reply (reply_text, 20-300 chars). "
        "Submit action_type='triage' with all fields filled."
    ),
}

_MAX_STEPS = {"easy": 2, "medium": 2, "hard": 3}


class AntiGravityEnv:
    """
    OpenEnv-compliant Email Triage Environment.

    Usage:
        env = AntiGravityEnv()
        obs = env.reset(task_level="easy")
        result = env.step(action)
        snapshot = env.state()
    """

    def __init__(self) -> None:
        self._state: Optional[InboxState] = None

    # ─── reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_level: str = "easy",
        seed: Optional[int] = None,
    ) -> Observation:
        """Generate a fresh inbox and return the initial observation."""
        level = task_level.lower()
        assert level in ("easy", "medium", "hard"), f"Invalid task_level: {level}"

        if level == "easy":
            emails, gt_labels, urgent_id, reply_kws = generate_easy_inbox(seed)
            gt_ranking = [urgent_id]
        elif level == "medium":
            emails, gt_labels, gt_ranking, urgent_id, reply_kws = generate_medium_inbox(seed)
        else:
            emails, gt_labels, gt_ranking, urgent_id, reply_kws = generate_hard_inbox(seed)

        # Derive a deterministic task_id from seed (reproducible episodes)
        if seed is not None:
            task_id = hashlib.md5(f"{level}-{seed}".encode()).hexdigest()[:16]
        else:
            task_id = str(uuid.uuid4())

        self._state = InboxState(
            task_id=task_id,
            task_level=level,
            emails=emails,
            ground_truth_labels=gt_labels,
            ground_truth_ranking=gt_ranking,
            urgent_email_id=urgent_id,
            expected_reply_keywords=reply_kws,
        )

        return self._build_observation()

    # ─── step ─────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """Apply an agent action and return (observation, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        reward = compute_reward(action, self._state)
        self._state.step_count += 1
        self._state.cumulative_reward += reward

        done = (
            self._state.step_count >= _MAX_STEPS[self._state.task_level]
            or reward >= 0.9          # early termination on near-perfect score
        )
        self._state.done = done

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "step_count": self._state.step_count,
                "cumulative_reward": round(self._state.cumulative_reward, 4),
                "task_level": self._state.task_level,
            },
        )

    # ─── state ────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        """Return the full internal state (for debugging / evaluation)."""
        if self._state is None:
            return {"status": "not_started"}
        return self._state.model_dump()

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        assert self._state is not None
        return Observation(
            task_id=self._state.task_id,
            task_level=self._state.task_level,
            emails=self._state.emails,
            step_count=self._state.step_count,
            instructions=_INSTRUCTIONS[self._state.task_level],
            max_steps=_MAX_STEPS[self._state.task_level],
        )
