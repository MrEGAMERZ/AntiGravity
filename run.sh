#!/usr/bin/env bash
# run.sh — Safely run the AntiGravity benchmark using your local .env file.
#
# MISSION: Reads your private tokens from .env and starts the agent.

# 1. Load private variables from .env
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo "ERROR: .env file not found. Please create it first."
    exit 1
fi

echo "================================================================="
echo "🌌 STARTING ANTIGRAVITY INFERENCE (Baseline Agent)"
echo "Target: $OPENENV_URL"
echo "Model : $MODEL_NAME"
echo "================================================================="

# 2. Run the agent
python3 "$(dirname "$0")/inference.py"
