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

# AntiGravity: Intelligent Email Triage

AntiGravity is an OpenEnv-compliant environment designed to test how AI agents handle the complex task of email management. While most models are evaluated on static datasets, this project provides a dynamic environment where agents must make real-time decisions about priority, labeling, and professional communication.

## The Motivation
Email triage is a task millions of professionals perform daily, yet it remains difficult for AI due to the mix of nuance, urgency, and professional tone required. We built AntiGravity to move beyond simple text prediction and into the realm of **active decision-making**. 

Our goal was to see if an agent could:
* Distinguish between a generic marketing promotion and a high-stakes meeting request.
* Correctly rank multiple emails by their real-world urgency.
* Draft a reply that sounds like a professional assistant rather than a chatbot.

## How It Works
The environment is built on a modular loop consisting of three main parts:

1. **Data Generation**: The system generates a fresh inbox of synthetic emails. This includes hidden ground truth data for labels and priority ranking for objective scoring.
2. **Agent Interaction**: The agent reads the inbox and uses internal reasoning to decide on labels, rank the priority of the emails, and choose which one requires an immediate response.
3. **Scoring**: Our graders evaluate the agent's performance. We use measures like Kendall's Tau for rankings to give the agent partial credit for the logic behind its decisions.

## Implementation Journey
We developed this environment in three distinct phases:
* **Classification**: Building the core logic to identify different types of email.
* **Prioritization**: Moving into relative ranking to teach the agent to understand what matters most in a crowded inbox.
* **Execution**: Creating the final multi-step task of labeling, identifying urgency, and communicating effectively.

## Meet the Team (InlusionX)
InlusionX is a duo of developers focused on the practical application of large language models.
* **Shifana Khanum** (shiffukhan16@gmail.com)
* **Shaik Mohammad Rehan** (mohammadrehan432432@gmail.com)

## Project Architecture
* `environment/env.py`: The core logic managing the state and steps.
* `graders.py`: Deterministic scoring functions for objective evaluation.
* `data_gen.py`: Responsible for generating realistic synthetic emails.
* `inference.py`: Our baseline agent implementation using Chain of Thought reasoning.
* `server/app.py`: The API layer and the interactive dashboard.

## How to Run

### 1. Start the Environment
Docker is the recommended way to set up the environment locally.

**Mac and Linux:**
```bash
docker build -t antigravity .
docker run -p 7860:7860 antigravity
```

**Windows:**
```powershell
docker build -t antigravity .
docker run -p 7860:7860 antigravity
```

Alternatively, use our live hosted endpoint: [https://mregamerz-antigravity.hf.space](https://mregamerz-antigravity.hf.space)

### 2. Run the Baseline Agent
You can benchmark our agent against the environment using these commands:

**Mac and Linux:**
```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_api_key_here
python3 inference.py
```

**Windows (PowerShell):**
```powershell
$env:API_BASE_URL="https://api.groq.com/openai/v1"
$env:MODEL_NAME="llama-3.3-70b-versatile"
$env:HF_TOKEN="your_api_key_here"
python inference.py
```

## Experience the Dashboard
We built a real-time visualizer so judges can watch the agent in action. It is available at: [https://mregamerz-antigravity.hf.space/play](https://mregamerz-antigravity.hf.space/play)

---
**Submission Metadata**  
* **Compliance**: 26/26 OpenEnv validation checks passed.  
* **Baseline Score**: 0.92 Average (Llama-3.3-70b).  
* **Built For**: Meta x Scaler OpenEnv Hackathon 2026.