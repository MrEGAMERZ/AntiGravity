AntiGravity: Intelligent Email Triage

AntiGravity is an OpenEnv environment designed to test how AI agents handle the complex task of email management. While models are often evaluated on static datasets, this project provides a dynamic environment where agents must make real-time decisions about priority, labeling, and professional communication.

The Motivation
Email triage is a task millions of professionals perform daily, yet it remains difficult for AI due to the mix of nuance, urgency, and professional tone required. We built AntiGravity to move beyond simple text prediction and into the realm of active decision-making. 

Our goal was to see if an agent could distinguish between a generic marketing promotion and a high-stakes meeting request, correctly rank them by urgency, and draft a reply that actually sounds like a helpful assistant.

How It Works
The system is built on a modular loop:
1. Data Generation: The environment generates a fresh inbox of synthetic emails. This isn't just text; it includes hidden ground truth data for labels and priority ranking.
2. Agent Interaction: The agent reads the inbox and uses internal reasoning to decide on labels, rank the priority of the emails, and choose which one requires an immediate response.
3. Scoring: Our graders evaluate the agent's performance. Instead of a binary pass or fail, we use measures like Kendall's Tau for rankings to give the agent partial credit for the logic of its decisions.

Implementation Journey
We developed this environment in three distinct phases:
- Classification: Building the core logic to identify different types of email.
- Prioritization: Moving into relative ranking to teach the agent to understand what matters most.
- Implementation: Creating the final multi-step task of labeling, identifying urgency, and communicating effectively.

The Team (InlusionX)
InlusionX is a duo of developers focused on the practical application of large language models. 
- Shifana Khanum (shiffukhan16@gmail.com)
- Shaik Mohammad Rehan (mohammadrehan432432@gmail.com)

Project Architecture
- environment/env.py: The core logic managing the state and steps.
- graders.py: Deterministic scoring functions for objective evaluation.
- data_gen.py: The system responsible for generating realistic synthetic emails.
- inference.py: Our baseline agent implementation using Chain of Thought reasoning.
- server/app.py: The API layer and the interactive dashboard.

How to Run

Start the Environment
Docker is the recommended way to set up the environment locally.
Mac and Linux:
docker build -t antigravity .
docker run -p 7860:7860 antigravity

Windows:
docker build -t antigravity .
docker run -p 7860:7860 antigravity

You can also use the live endpoint: https://mregamerz-antigravity.hf.space

Run the Baseline Agent
The following commands will run our agent against the environment.
Mac and Linux:
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_api_key
python3 inference.py

Windows:
$env:API_BASE_URL=https://api.groq.com/openai/v1
$env:MODEL_NAME=llama-3.3-70b-versatile
$env:HF_TOKEN=your_api_key
python inference.py

Experience the Dashboard
A real-time visualizer is available at: https://mregamerz-antigravity.hf.space/play

Final Submission Stats
Compliance: 26/26 OpenEnv validation checks passed.
Baseline Average: 0.96 (Llama-3.3-70b).
Features: Deterministic seeds, Real-time Visualizer, Multi-platform support.