"""
AntiGravity — Synthetic Email Generator  (v2)
Generates realistic email inboxes with embedded ground-truth labels,
urgency rankings, and expected reply keywords.

Improvements over v1:
  - 3× more email templates per category (more variety for judges / evaluators)
  - Realistic corporate/domain senders (not just free-email domains)
  - Richer body templates with named placeholders
  - context-sensitive reply keywords per email type
  - Added `thread_id` context to some emails for multi-turn realism
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from faker import Faker

from models import Email, VALID_LABELS

_fake = Faker()


# ─── Extended Templates per category ─────────────────────────────────────────

_TEMPLATES = {
    "spam": {
        "subjects": [
            "YOU WON $1,000,000 — Claim Now!",
            "Congratulations! You've been selected",
            "FREE Gift — Act Immediately",
            "Urgent: Your account needs verification",
            "Make $$$ from home — Guaranteed",
            "LAST WARNING: Your package is on hold",
            "Your crypto wallet has been credited",
            "Exclusive opportunity — act in 24 hours",
            "Nigerian Prince needs your help",
            "You're our 1,000,000th visitor!",
        ],
        "bodies": [
            "Dear valued customer,\n\nYou have been selected as today's lucky winner. Click here to claim your prize: http://totally-legit-prize.win\n\nDo NOT miss this opportunity!",
            "Hi there,\n\nWe noticed suspicious activity. Reply with your SSN and password to unlock your account.\n\nSecurity Team",
            "EARN $5,000 per week from HOME. No experience needed. Limited slots. Reply YES to join!",
            "Dear Friend,\n\nI am Prince Adebayo from Nigeria. I need your help to transfer $18,000,000. You will receive 40% commission. Reply urgently.\n\nWith regards,\nPrince Adebayo",
            "CONGRATULATIONS!! Your email has been chosen in our annual customer lottery. To claim your $50,000 prize, click: http://prize-winner.xyz/claim\n\nThis offer expires in 24 HOURS.",
            "Your Microsoft account has been compromised! Click here IMMEDIATELY to secure it: http://microsoft-security-alert.xyz\n\nDo not ignore this message.",
            "Hi,\n\nYour Amazon package #AM-291847 is on hold due to failed delivery. Pay $2.99 to reschedule: http://amaz0n-delivery.xyz",
        ],
        "senders": ["noreply@prize-zone.xyz", "winner@lottery-global.win", "security@microsoft-alert.xyz",
                    "support@amaz0n-help.xyz", "prince@royalfoundation.ng"],
    },
    "promo": {
        "subjects": [
            "🎉 Flash Sale — 50% off everything today only!",
            "Your exclusive discount code inside",
            "Weekend deals you don't want to miss",
            "Last chance: Save 30% before midnight",
            "New arrivals just dropped — shop now",
            "Members-only early access starts NOW",
            "We miss you — here's 25% off to come back",
            "Your cart is waiting — complete your purchase",
            "Introducing our summer collection 🌞",
            "Free shipping on all orders this weekend",
        ],
        "bodies": [
            "Hi {name},\n\nThis weekend only, enjoy 50% off sitewide. Use code SAVE50 at checkout.\n\nShop now: https://shop.example.com\n\nBest,\nThe Team",
            "Hey {name}! 👋\n\nWe've got new arrivals you'll love. Check them out before they sell out.\n\nTap here to browse.",
            "Dear {name},\n\nAs a valued member, you get early access to our biggest sale of the year. Don't miss it!",
            "Hi {name},\n\nYou left something behind! Your cart has items waiting. Complete your order now and get free shipping.\n\nShop: https://shop.example.com/cart",
            "Hello {name},\n\nWe haven't seen you in a while! Come back and enjoy 25% off your next purchase with code MISSYOU25.\n\nValid until end of month.",
            "Hi {name},\n\nOur summer collection just launched and it's 🔥. New styles, new colours, same quality you love.\n\nFree shipping on orders over $50 this weekend only.",
            "Dear {name},\n\nFlash sale starts NOW! 40% off our best-selling items for the next 6 hours only. Don't miss out!\n\nCode: FLASH40",
        ],
        "senders": ["deals@shopify-store.com", "offers@brand.co", "noreply@promo.example.com",
                    "hello@weekendsale.io", "marketing@fashionbrand.com"],
    },
    "newsletter": {
        "subjects": [
            "Weekly Digest: Top stories this week",
            "Your monthly product update",
            "The AI Insider — Issue #42",
            "Tech roundup: What you missed",
            "Community newsletter — March 2026",
            "🚀 This week in open source",
            "The Founder's Brief — Sunday Edition",
            "Design Matters: Issue #17",
            "Hacker News Weekly Digest",
            "Your personalized reading list is ready",
        ],
        "bodies": [
            "Hello {name},\n\nHere's your weekly digest of the top stories:\n\n1. AI breakthroughs in 2026\n2. New Python 3.14 features\n3. Open source spotlight\n\nRead more at: https://newsletter.example.com",
            "Hi {name},\n\nThis month we launched 3 new features, fixed 47 bugs, and onboarded 200 new users. Here's what's new...",
            "Dear {name},\n\nCommunity highlights:\n- 1,200 new members joined\n- Best thread of the week: 'How to scale RAG pipelines'\n\nSee you next week!",
            "Hey {name},\n\nThis week in AI:\n• Meta released Llama 4 with 400B parameters\n• Google DeepMind's AlphaChemist discovered 3 new materials\n• OpenAI launched GPT-5 to all paid users\n\nFull recap: https://aiinsider.news",
            "Hi {name},\n\nThe Founder's Brief is here. This week:\n- How Figma rebuilt their rendering engine\n- Lessons from 100 failed startups\n- The 5 metrics every SaaS founder should track\n\nHappy reading!",
            "Hello {name},\n\nYour personalized reading list for this week is ready based on your interests:\n1. 'The Future of Agent-Based AI' — 8 min read\n2. 'Building Reliable Distributed Systems' — 12 min read\n3. 'Why Rust is winning' — 5 min read",
            "Dear {name},\n\nDesign Matters Issue #17 is here!\n\nThis week we feature:\n- Interview with the Figma design team\n- 10 UI patterns that increase conversions\n- Free resource: Icon pack with 500 icons\n\nEnjoy!",
        ],
        "senders": ["digest@weeklytech.io", "newsletter@aiinsider.news", "hello@founders-brief.com",
                    "updates@opensourcedigest.dev", "noreply@designmatters.io"],
    },
    "important": {
        "subjects": [
            "URGENT: Action required on your account",
            "Interview scheduled for tomorrow at 10 AM",
            "Contract requires your signature by EOD",
            "Server outage — immediate response needed",
            "Your invoice #INV-2026-044 is overdue",
            "Critical security patch — deploy by Friday",
            "Board meeting moved to Monday — attendance required",
            "Client escalation — needs response within 2 hours",
            "Performance review scheduled — please confirm",
            "Production incident P1 — all hands on deck",
        ],
        "bodies": [
            "Hi {name},\n\nWe need you to sign the attached contract by end of day today. Please review and confirm receipt.\n\nBest regards,\n{sender_name}",
            "Dear {name},\n\nYour interview has been confirmed for tomorrow at 10:00 AM. Please join via: https://meet.example.com/interview\n\nGood luck!",
            "Hi {name},\n\nOur production server is experiencing a critical outage. Please join the incident channel immediately.\n\n— On-call",
            "Hello,\n\nYour invoice #INV-2026-044 for $4,200 is now 15 days overdue. Please process payment ASAP.\n\nFinance Team",
            "Hi {name},\n\nA critical security vulnerability (CVE-2026-1337) has been identified in our auth layer. We need to deploy the patch before Friday 5 PM. Please coordinate with your team.\n\nThanks,\n{sender_name}",
            "Dear {name},\n\nThe board meeting has been rescheduled from Wednesday to Monday 9 AM. Your attendance is mandatory. Please confirm by EOD.\n\n{sender_name}",
            "Hi {name},\n\nWe have a P1 production incident affecting 40% of users. All engineers are needed in the war room NOW. Join: https://meet.example.com/incident\n\n— Engineering Lead",
            "Dear {name},\n\nWe have a client escalation from Acme Corp — they've reported data inconsistencies in the Q1 report. This needs a response within 2 hours.\n\n{sender_name}, Account Manager",
            "Hi {name},\n\nYour performance review is scheduled for Thursday 3 PM. Please prepare your self-assessment doc and share it with your manager before then.\n\nHR Team",
            "Hello {name},\n\nThis is a reminder that your project deadline is tomorrow. The client is expecting the delivery by 9 AM sharp. Please confirm the status.\n\nProject Manager",
        ],
        "senders": [
            "hr@company.com", "manager@corp.io", "oncall@engineering.co",
            "finance@accounting.com", "security@infosec.io", "pm@projects.co"
        ],
    },
}

# Context-rich reply keywords per email scenario
_REPLY_KEYWORDS_MAP = {
    "important": [
        "received", "on it", "will handle", "understood", "confirmed",
        "looking into", "acknowledge", "asap", "right away", "noted",
        "will update", "have noted", "will respond", "thank you for", "apologies",
    ],
    "spam": ["received"],
    "promo": ["received"],
    "newsletter": ["received"],
}


def _now_minus(hours: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_email(category: str, hours_ago: int) -> Email:
    tmpl = _TEMPLATES[category]
    subject = random.choice(tmpl["subjects"])
    body_tmpl = random.choice(tmpl["bodies"])
    name = _fake.first_name()
    sender_name = _fake.name()

    # Use category-specific sender or generate a realistic one
    if random.random() < 0.6 and tmpl.get("senders"):
        sender = random.choice(tmpl["senders"])
    else:
        domain = "spam-domain.xyz" if category == "spam" else _fake.free_email_domain()
        sender = f"{_fake.user_name()}@{domain}"

    try:
        body = body_tmpl.format(name=name, sender_name=sender_name)
    except KeyError:
        body = body_tmpl  # Template has no placeholders

    return Email(
        sender=sender,
        subject=subject,
        body=body,
        timestamp=_now_minus(hours_ago),
        has_attachment=(category == "important" and random.random() < 0.5),
    )


# ─── Public generators ────────────────────────────────────────────────────────

def generate_easy_inbox(
    seed: int | None = None,
) -> Tuple[List[Email], Dict[str, str], str, List[str]]:
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


def generate_medium_inbox(
    seed: int | None = None,
) -> Tuple[List[Email], Dict[str, str], List[str], str, List[str]]:
    """
    Returns: (emails, ground_truth_labels, ground_truth_ranking, urgent_id, reply_keywords)
    Medium: 10 emails; rank from most to least urgent.
    Urgency order: important (most recent) > newsletter > promo > spam
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

    # Rank: important first (most recent first within category), then newsletter, promo, spam
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


def generate_hard_inbox(
    seed: int | None = None,
) -> Tuple[List[Email], Dict[str, str], List[str], str, List[str]]:
    """
    Returns: (emails, ground_truth_labels, ground_truth_ranking, urgent_id, reply_keywords)
    Hard: mixed inbox; label all + identify urgent + draft professional reply.
    10 emails, more balanced distribution.
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
