---
title: AntiGravity
emoji: 🌌
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🌌 AntiGravity: Mastering the Inbox 📧

> **An OpenEnv Environment for Intelligent Email Triage**
> Built with ❤️ for the Meta × Scaler Hackathon · 2026

---

## 🚀 The Mission: Why AntiGravity?

We’ve all seen AI models write code or solve logic puzzles, but the **real-world messy inbox** remains one of the hardest reasoning tests. 

We built **AntiGravity** to fill a gap: while the RL community has plenty of games and toy examples, we wanted to build an environment that models a task millions of professionals do every day — **Email Triage.**

Our goal was to create an environment where an agent isn't just "predicting text"; it’s *making decisions*. Which email is actually urgent? What tone is appropriate for this client? How do we ignore a "low-urgency" newsletter while prioritising a meeting request? 

---

## 🏗️ Our Implementation Journey

We didn't just write code; we iterated through a process of discovery. Here's how the "AntiGravity" world was born:

1.  **Level 1: The Classifier**: We started by building the foundation of classification. We wanted to make sure an agent could distinguish between a generic marketing promotion and a high-stakes contract.
2.  **Level 2: The Prioritizer**: Then we moved into *relative ranking*. This was where it got interesting — we implemented **Kendall’s Tau** logic to reward the agent for getting the *order* right, not just the labels.
3.  **Level 3: The Full Executive Assistant**: Finally, we combined everything into a multi-objective task where the agent must label, identify the *single* urgent task, and draft a human-quality reply.

---

## 💡 What We Learned

Building an OpenEnv ecosystem from scratch taught us three big things about the future of AI:

*   **Rewards Matter**: You can’t just give an agent a "1" or a "0" at the end. We learned to design **partial progress signals** (shaping) so the agent learns from its small successes.
*   **Context is King**: A subject line like "URGENT" isn't always urgent (often it's spam!). We learned to hide "traps" in the data to force the agent to read the body, not just the subject.
*   **The "Human" Tone**: We realized that a good reply isn't just "correct"—it needs to sound professional. This led us to build a **synonym-aware grader** that looks for professional cues like "noted," "acknowledged," and "investigating."

---

## 👥 Meet the Team

We are a group of passionate developers obsessed with the intersection of LLMs and world-modeling. We believe that for AI Agents to truly become helpful assistants, they need environments like AntiGravity to "train" their intuition.

---

## 🛠️ Project Architecture

| Component | Responsibility |
|-----------|----------------|
| `environment/env.py` | The "Engine" — handles `reset()`, `step()`, and `state()`. |
| `graders.py`         | The "Judge" — deterministic, nuanced scoring (0.0 to 1.0). |
| `data_gen.py`        | The "World" — 100+ templates for realistic synthetic emails. |
| `inference.py`       | The "Agent" — our Chain-of-Thought (CoT) baseline implementation. |
| `server/app.py`      | The "API" — FastAPI bridge with a **Live Visualizer UI** at `/play`. |

---

## 🏁 How to Run

### 1. The Environment (HTTP API)
Clone and run via Docker to see it in action:
```bash
docker build -t antigravity .
docker run -p 7860:7860 antigravity
```

### 2. The Baseline Agent
Benchmark our agent (Llama-3.3-70b) against the environment:
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_api_key_here"
python3 inference.py
```

### 3. The Visualizer
Visit our live dashboard at: **[https://mregamerz-antigravity.hf.space/play](https://mregamerz-antigravity.hf.space/play)** 🌟

---

## 🏆 Final Submission Stats
- **Compliance**: 26/26 OpenEnv validation checks passed.
- **Baseline Average**: ~0.85
- **Features**: Deterministic seeds, Real-time Dashboard, CoT Agent Baseline.
