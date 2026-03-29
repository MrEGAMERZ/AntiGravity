---
title: AntiGravity
emoji: 📧
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AntiGravity 📧

> **An OpenEnv-compliant AI environment for intelligent email triage**
> Built for the **OpenEnv Hackathon · Meta × Scaler · April 2026**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-7c5cfc)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://docker.com)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live-yellow)](https://huggingface.co/spaces/mregamerz/AntiGravity)
[![Score](https://img.shields.io/badge/Baseline%20Score-0.91%20avg-brightgreen)](https://huggingface.co/spaces/mregamerz/AntiGravity)

---

## 🧠 What is AntiGravity?

AntiGravity simulates a **realistic professional email inbox** where an AI agent learns to:
- **Classify** emails (spam / promo / newsletter / important)
- **Prioritize** a full inbox by urgency
- **Triage** an inbox end-to-end: label + identify what's urgent + draft a professional reply

No human uses this environment directly — an AI agent "plays" in it, takes actions, and receives **deterministic scored feedback**. Researchers and companies can use it to **benchmark how well a model reasons about real-world email tasks**.

> **Why Email?** Email triage is one of the most universal, high-stakes knowledge-work tasks. It requires reading comprehension, contextual reasoning, urgency assessment, and drafting skill — exactly what separates good LLMs from great ones. No standard OpenEnv benchmark existed for this. **AntiGravity fills that gap.**

---

## 🎯 Tasks

| # | Level  | Task | Grader |
|---|--------|------|--------|
| 1 | 🟢 Easy | **Single Email Label** — Classify one email as `spam`, `promo`, `newsletter`, or `important` | Exact match: 1.0, adjacent category: 0.5, wrong: 0.0 |
| 2 | 🟡 Medium | **Inbox Priority Sort** — Rank 10 emails by urgency, most urgent first | Kendall's Tau correlation ∈ [0, 1] |
| 3 | 🔴 Hard | **Triage + Reply + Archive** — Label all emails, identify the urgent one, draft a professional reply | Composite: 0.3×labels + 0.3×urgency + 0.4×reply quality |

All graders are **deterministic** — same input, same seed, always produces the same score.

---

## 🏆 Baseline Scores (v2 — Chain-of-Thought Agent)

| Task       | GPT-4o-mini (v1) | **Llama-3.3-70b via Groq (v2)** |
|------------|------------------|---------------------------------|
| Easy       | 0.92             | **1.00** ✅                      |
| Medium     | 0.74             | **0.89** ✅                      |
| Hard       | 0.61             | **0.83** ✅ (+0.22!)             |
| **Average**| **0.76**         | **🏆 0.91**                      |

*Tested live against `https://mregamerz-antigravity.hf.space` using v2 Chain-of-Thought prompting.*
*Run: `GROQ_API_KEY=<key> API_BASE_URL=https://mregamerz-antigravity.hf.space python3 inference.py`*

---

## 🔭 Observation Space

```json
{
  "task_id": "string",
  "task_level": "easy | medium | hard",
  "emails": [
    {
      "id": "string",
      "sender": "string",
      "subject": "string",
      "body": "string",
      "timestamp": "ISO-8601",
      "has_attachment": true
    }
  ],
  "step_count": 0,
  "instructions": "natural language goal",
  "max_steps": 3
}
```

## ⚡ Action Space

```json
{
  "action_type": "label | rank | triage",
  "labels":   {"email_id": "spam | promo | important | newsletter"},
  "ranking":  ["email_id_1", "email_id_2", "..."],
  "urgent_id": "email_id",
  "reply_text": "short professional reply (20–80 words)"
}
```

---

## 🌐 API Reference

| Method | Path       | Description |
|--------|------------|-------------|
| `GET`  | `/`        | Environment info + endpoint list |
| `GET`  | `/health`  | Health check (`{"status": "ok"}`) |
| `GET`  | `/metrics` | Aggregate reward stats across sessions |
| `GET`  | `/play`    | **Interactive Visualizer Dashboard** 🌟 |
| `POST` | `/reset`   | Start new episode (`task_level`, `seed?`) |
| `POST` | `/step`    | Submit action → `(obs, reward, done, info)` |
| `GET`  | `/state`   | Full internal state snapshot (debug/eval) |
| `GET`  | `/docs`    | Interactive Swagger API explorer |

---

## 🚀 Setup & Local Run

### Prerequisites
- Python 3.11+
- pip

### Install

```bash
cd antigravity
pip install -e .
```

### Run server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Server is now live at `http://localhost:7860`

### Quick test

```bash
# Health check
curl http://localhost:7860/health

# Start an easy task (seed for reproducibility)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy", "seed": 42}'

# Submit a label action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "label", "labels": {"<email_id>": "important"}}'

# View internal state
curl http://localhost:7860/state
```

---

## 🐳 Docker

```bash
# Build
docker build -t antigravity .

# Run
docker run -p 7860:7860 antigravity
```

---

## 🤖 Run Baseline Inference (Free — via Groq)

```bash
export API_BASE_URL="https://mregamerz-antigravity.hf.space"
export GROQ_API_KEY="your_groq_key_from_console.groq.com"

python3 inference.py
```

The baseline agent uses **Chain-of-Thought prompting** with `llama-3.3-70b-versatile` via Groq's free API.

---

## ✅ Pre-Submission Validator

```bash
# Validate against the live Space
python3 validate.py --url https://mregamerz-antigravity.hf.space

# Or validate locally
python3 validate.py --url http://localhost:7860
```

Checks all 6 requirement areas: health, root, easy/medium/hard tasks, state, and determinism.

---

## 📁 Project Structure

```
antigravity/
├── __init__.py              ← Public exports
├── models.py                ← Pydantic models (Email, Observation, Action, StepResult)
├── data_gen.py              ← Synthetic email generator (100+ templates, 4 categories)
├── graders.py               ← Deterministic reward functions (v2 with synonym expansion)
├── client.py                ← Async + sync HTTP client
├── inference.py             ← Baseline agent (Chain-of-Thought, Groq/Llama-3.3)
├── validate.py              ← Pre-submission validator (6 checks)
├── openenv.yaml             ← Environment manifest
├── pyproject.toml           ← Dependencies
├── Dockerfile               ← Container (HF Spaces Docker SDK)
├── .dockerignore
├── environment/
│   └── env.py               ← AntiGravityEnv (reset/step/state)
└── server/
    ├── app.py               ← FastAPI server (+ /metrics endpoint)
    └── requirements.txt     ← Docker deps
```

---

## 🏅 Judging Criteria Alignment

| Criterion | Weight | Our Approach |
|-----------|--------|--------------|
| **Real-world Utility** | 30% | Email triage is universal, high-stakes, and requires multi-step reasoning — a novel OpenEnv domain |
| **Task & Grader Quality** | 25% | 3 tasks, escalating difficulty, fully deterministic graders, partial credit on all tasks |
| **Environment Design** | 20% | Clean `reset()` / `step()` / `state()`, typed Pydantic models, partial rewards, early termination |
| **Code Quality** | 15% | Docker-ready, FastAPI `/docs`, `validate.py` script, HF Space deployed & live |

**Live URL:** `https://mregamerz-antigravity.hf.space`

---

## 📄 License

MIT © AntiGravity Team · OpenEnv Hackathon 2026
