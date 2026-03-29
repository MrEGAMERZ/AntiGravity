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

---

## What is AntiGravity?

AntiGravity is a realistic email inbox simulation where an AI agent learns to
**triage**, **prioritise**, and **reply** to emails the way a human professional would.

No human uses this environment directly — an AI agent "plays" in the simulated
inbox, takes actions, and receives scored feedback. Researchers and companies
can use it to **benchmark how well a model reasons about real-world email tasks**.

---

## Tasks

| # | Level  | Task                     | Reward signal |
|---|--------|--------------------------|---------------|
| 1 | Easy   | Single Email Label       | 1.0 exact · 0.5 adjacent · 0.0 wrong |
| 2 | Medium | Inbox Priority Sort      | Kendall's Tau ∈ [0, 1] |
| 3 | Hard   | Triage + Reply + Archive | 0.3×labels + 0.3×urgency + 0.4×reply quality |

All graders are **deterministic** — same input always yields the same score.

---

## Observation Space

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
  "max_steps": 2
}
```

## Action Space

```json
{
  "action_type": "label | rank | triage",
  "labels": {"email_id": "spam | promo | important | newsletter"},
  "ranking": ["email_id_1", "email_id_2", "..."],
  "urgent_id": "email_id",
  "reply_text": "short professional reply"
}
```

---

## API Endpoints

| Method | Path      | Description                           |
|--------|-----------|---------------------------------------|
| `POST` | `/reset`  | Start new episode (body: `{task_level, seed?}`) |
| `POST` | `/step`   | Submit action → `(obs, reward, done, info)` |
| `GET`  | `/state`  | Full internal state snapshot          |
| `GET`  | `/health` | Health check                          |

---

## Setup & Local Run

### Prerequisites
- Python 3.11+
- pip or uv

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

### Test with curl

```bash
# Reset (easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy", "seed": 42}'

# Submit a label action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "label", "labels": {"abc123": "important"}}'

# View state
curl http://localhost:7860/state
```

---

## Docker

```bash
# Build
docker build -t antigravity .

# Run
docker run -p 7860:7860 antigravity
```

---

## Run Baseline Inference

```bash
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."

python inference.py
```

### Baseline Scores

| Task   | GPT-4o-mini | Llama-3.3-70b (Groq) |
|--------|-------------|----------------------|
| Easy   | 0.92        | **1.00** ✅           |
| Medium | 0.74        | **0.89** ✅           |
| Hard   | 0.61        | 0.48                 |
| **Average** | **0.76** | **0.79** ✅      |

*Tested live against `https://mregamerz-antigravity.hf.space`. Llama-3.3 via Groq is the recommended free baseline.*

---

## Project Structure

```
antigravity/
├── __init__.py          ← Public exports
├── models.py            ← Pydantic models (Email, Observation, Action, StepResult)
├── data_gen.py          ← Synthetic email generator (Faker)
├── graders.py           ← Deterministic reward functions
├── client.py            ← Async + sync HTTP client
├── inference.py         ← Baseline agent script
├── openenv.yaml         ← Environment manifest
├── pyproject.toml       ← Dependencies
├── Dockerfile           ← Container (HF Spaces)
├── .dockerignore
├── environment/
│   └── env.py           ← AntiGravityEnv (reset/step/state)
└── server/
    ├── app.py           ← FastAPI HTTP server
    └── requirements.txt ← Docker deps
```

---

## Scoring Criteria Alignment

| Judge Criterion      | Our Approach |
|----------------------|--------------|
| **Real-world Utility (30%)** | Email triage — universal, high-value, novel OpenEnv domain |
| **Task & Grader Quality (25%)** | 3 tasks, escalating difficulty, deterministic graders |
| **Environment Design (20%)** | Clean reset(), partial rewards, typed models |
| **Code Quality (15%)** | Docker-ready, openenv validate, HF Spaces deploy |

---

## License

MIT © AntiGravity Team · OpenEnv Hackathon 2026
