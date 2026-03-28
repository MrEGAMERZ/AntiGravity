"""
AntiGravity — Synthetic Email Generator
Generates realistic email inboxes with embedded ground-truth labels,
urgency rankings, and expected reply keywords.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from faker import Faker

from models import Email, VALID_LABELS

_fake = Faker()


# ─── Templates per category ──────────────────────────────────────────────────

_TEMPLATES = {
    "spam": [
        {
            "subjects": [
                "YOU WON $1,000,000 — Claim Now!",
                "Congratulations! You've been selected",
                "FREE Gift — Act Immediately",
                "Urgent: Your account needs verification",
                "Make $$$ from home — Guaranteed",
            ],
            "bodies": [
                "Dear valued customer,\n\nYou have been selected as today's lucky winner. Click here to claim your prize: http://totally-legit-prize.win\n\nDo NOT miss this opportunity!",
                "Hi there,\n\nWe noticed suspicious activity. Reply with your SSN and password to unlock your account.\n\nSecurity Team",
                "EARN $5,000 per week from HOME. No experience needed. Limited slots. Reply YES to join!",
            ],
        }
    ],
    "promo": [
        {
            "subjects": [
                "🎉 Flash Sale — 50% off everything today only!",
                "Your exclusive discount code inside",
                "Weekend deals you don't want to miss",
                "Last chance: Save 30% before midnight",
                "New arrivals just dropped — shop now",
            ],
            "bodies": [
                "Hi {name},\n\nThis weekend only, enjoy 50% off sitewide. Use code SAVE50 at checkout.\n\nShop now: https://shop.example.com\n\nBest,\nThe Team",
                "Hey {name}! 👋\n\nWe've got new arrivals you'll love. Check them out before they sell out.\n\nTap here to browse.",
                "Dear {name},\n\nAs a valued member, you get early access to our biggest sale of the year. Don't miss it!",
            ],
        }
    ],
    "newsletter": [
        {
            "subjects": [
                "Weekly Digest: Top stories this week",
                "Your monthly product update",
                "The AI Insider — Issue #42",
                "Tech roundup: What you missed",
                "Community newsletter — March 2026",
            ],
            "bodies": [
                "Hello {name},\n\nHere's your weekly digest of the top stories:\n\n1. AI breakthroughs in 2026\n2. New Python 3.14 features\n3. Open source spotlight\n\nRead more at: https://newsletter.example.com",
                "Hi {name},\n\nThis month we launched 3 new features, fixed 47 bugs, and onboarded 200 new users. Here's what's new...",
                "Dear {name},\n\nCommunity highlights:\n- 1,200 new members joined\n- Best thread of the week: 'How to scale RAG pipelines'\n\nSee you next week!",
            ],
        }
    ],
    "important": [
        {
            "subjects": [
                "URGENT: Action required on your account",
                "Interview scheduled for tomorrow at 10 AM",
                "Contract requires your signature by EOD",
                "Server outage — immediate response needed",
                "Your invoice #INV-2026-044 is overdue",
            ],
            "bodies": [
                "Hi {name},\n\nWe need you to sign the attached contract by end of day today. Please review and confirm receipt.\n\nBest regards,\n{sender_name}",
                "Dear {name},\n\nYour interview has been confirmed for tomorrow at 10:00 AM. Please join via: https://meet.example.com/interview\n\nGood luck!",
                "Hi {name},\n\nOur production server is experiencing a critical outage. Please join the incident channel immediately.\n\n— On-call",
                "Hello,\n\nYour invoice #INV-2026-044 for $4,200 is now 15 days overdue. Please process payment ASAP.\n\nFinance Team",
            ],
        }
    ],
}

# Keywords considered good reply signals for urgent emails
_REPLY_KEYWORDS_MAP = {
    "important": [
        "received", "on it", "will handle", "understood", "confirmed",
        "looking into", "acknowledge", "asap", "right away", "noted",
    ]
}


def _now_minus(hours: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_email(category: str, hours_ago: int) -> Email:
    tmpl = random.choice(_TEMPLATES[category][0]["subjects"])
    subject = tmpl
    body_tmpl = random.choice(_TEMPLATES[category][0]["bodies"])
    name = _fake.first_name()
    sender_name = _fake.name()
    body = body_tmpl.format(name=name, sender_name=sender_name)
    sender_domain = (
        "spam-domain.xyz" if category == "spam"
        else _fake.free_email_domain()
    )
    sender = f"{_fake.user_name()}@{sender_domain}"
    return Email(
        sender=sender,
        subject=subject,
        body=body,
        timestamp=_now_minus(hours_ago),
        has_attachment=(category == "important" and random.random() < 0.4),
    )


# ─── Public generators ────────────────────────────────────────────────────────

def generate_easy_inbox(seed: int | None = None) -> Tuple[List[Email], Dict[str, str], str, List[str]]:
    """
    Returns: (emails, ground_truth_labels, urgent_id, expected_reply_keywords)
    Easy: single email, classify it.
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    category = random.choice(list(VALID_LABELS))
    email = _make_email(category, hours_ago=random.randint(1, 48))
    ground_truth = {email.id: category}
    return (
        [email],
        ground_truth,
        email.id,
        _REPLY_KEYWORDS_MAP.get(category, ["received"]),
    )


def generate_medium_inbox(seed: int | None = None) -> Tuple[List[Email], Dict[str, str], List[str], str, List[str]]:
    """
    Returns: (emails, ground_truth_labels, ground_truth_ranking, urgent_id, reply_keywords)
    Medium: 10 emails; rank by urgency.
    Urgency order: important (oldest first) > newsletter > promo > spam
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    categories = (
        ["important"] * 3
        + ["newsletter"] * 2
        + ["promo"] * 3
        + ["spam"] * 2
    )
    random.shuffle(categories)

    hours = list(range(1, 50, 5))
    random.shuffle(hours)

    emails: List[Email] = []
    gt_labels: Dict[str, str] = {}
    for cat, h in zip(categories, hours):
        e = _make_email(cat, h)
        emails.append(e)
        gt_labels[e.id] = cat

    # Rank: important first (sort by recency within category), then newsletter, promo, spam
    _priority = {"important": 0, "newsletter": 1, "promo": 2, "spam": 3}
    ranked = sorted(emails, key=lambda e: (_priority[gt_labels[e.id]], e.timestamp))
    gt_ranking = [e.id for e in ranked]

    urgent_id = gt_ranking[0]  # most urgent
    return (
        emails,
        gt_labels,
        gt_ranking,
        urgent_id,
        _REPLY_KEYWORDS_MAP.get(gt_labels[urgent_id], ["received"]),
    )


def generate_hard_inbox(seed: int | None = None) -> Tuple[List[Email], Dict[str, str], List[str], str, List[str]]:
    """
    Returns: (emails, ground_truth_labels, ground_truth_ranking, urgent_id, reply_keywords)
    Hard: mixed inbox; label all + identify urgent + draft reply.
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    categories = (
        ["important"] * 2
        + ["newsletter"] * 2
        + ["promo"] * 3
        + ["spam"] * 3
    )
    random.shuffle(categories)

    emails: List[Email] = []
    gt_labels: Dict[str, str] = {}
    for i, cat in enumerate(categories):
        e = _make_email(cat, hours_ago=random.randint(1, 72))
        emails.append(e)
        gt_labels[e.id] = cat

    _priority = {"important": 0, "newsletter": 1, "promo": 2, "spam": 3}
    ranked = sorted(emails, key=lambda e: (_priority[gt_labels[e.id]], e.timestamp))
    gt_ranking = [e.id for e in ranked]

    urgent_id = gt_ranking[0]
    return (
        emails,
        gt_labels,
        gt_ranking,
        urgent_id,
        _REPLY_KEYWORDS_MAP.get("important", ["received", "on it", "will handle"]),
    )
