"""
AntiGravity — FastAPI Server  v2  (OpenEnv HTTP Interface)
Exposes POST /reset, POST /step, GET /state, GET /health

New in v2:
  - POST /reset returns proper Observation model
  - GET /metrics endpoint (aggregate stats across sessions)
  - Richer root metadata
  - Better error handling
"""
from __future__ import annotations

import sys
import os
import time
from collections import defaultdict

# Make sure the antigravity package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from models import Action, Observation, StepResult
from environment.env import AntiGravityEnv


# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AntiGravity — Email Triage OpenEnv",
    description=(
        "**AntiGravity** is an OpenEnv-compliant HTTP server for email triage. "
        "An AI agent interacts with a simulated inbox via `reset()` → `step()` loops, "
        "receiving scored feedback (0.0–1.0) on classification, ranking, and reply quality.\n\n"
        "**Tasks:**\n"
        "- `easy` — Classify a single email (spam/promo/newsletter/important)\n"
        "- `medium` — Rank 10 emails by urgency (Kendall's Tau grader)\n"
        "- `hard` — Full triage: label all emails + identify urgent + draft reply\n\n"
        "Built for the **OpenEnv Hackathon · Meta × Scaler · 2026**"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per-process)
_env = AntiGravityEnv()

# Session-level metrics
_metrics: dict = defaultdict(lambda: {"total_episodes": 0, "total_reward": 0.0, "step_calls": 0})
_server_start = time.time()


# ─── Request schemas ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_level: str = "easy"   # "easy" | "medium" | "hard"
    seed: Optional[int] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"], summary="Environment info & Visualizer")
def root(request: Request):
    """
    Smart endpoint: 
    - Use in browser to see the Interactive Visualizer.
    - Use via API/OpenEnv to get environment metadata.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return play_ui()
    
    return {
        "name": "AntiGravity",
        "version": "2.0.0",
        "description": "Email triage OpenEnv environment",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/metrics", "/docs", "/play"],
        "grader_type": "deterministic",
        "reward_range": [0.0, 1.0],
    }


@app.get("/health", tags=["meta"], summary="Health check")
def health():
    """Returns {status: ok} when server is running."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _server_start)}


@app.get("/metrics", tags=["meta"], summary="Aggregate session metrics")
def metrics():
    """Returns aggregate performance metrics across all sessions."""
    total_eps = sum(v["total_episodes"] for v in _metrics.values())
    total_reward = sum(v["total_reward"] for v in _metrics.values())
    return {
        "total_episodes": total_eps,
        "total_step_calls": sum(v["step_calls"] for v in _metrics.values()),
        "mean_reward_all_time": round(total_reward / max(total_eps, 1), 4),
        "per_task": {
            level: {
                "episodes": _metrics[level]["total_episodes"],
                "mean_reward": round(
                    _metrics[level]["total_reward"] / max(_metrics[level]["total_episodes"], 1),
                    4,
                ),
            }
            for level in ["easy", "medium", "hard"]
        },
        "uptime_seconds": round(time.time() - _server_start),
    }


@app.post("/reset", response_model=Observation, tags=["env"], summary="Start a new episode")
def reset(req: ResetRequest):
    """
    Reset the environment. Returns the initial Observation.
    - **task_level**: `easy` | `medium` | `hard`
    - **seed**: optional integer for reproducible episodes
    """
    if req.task_level not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_level '{req.task_level}'. Must be easy, medium, or hard.",
        )
    obs = _env.reset(task_level=req.task_level, seed=req.seed)
    _metrics[req.task_level]["total_episodes"] += 1
    return obs


@app.get("/play", response_class=HTMLResponse, tags=["ui"], summary="Interactive Visualizer")
def play_ui():
    """Returns the Interactive Visualizer UI for judges."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Visualizer UI not found")


@app.post("/step", response_model=StepResult, tags=["env"], summary="Submit an agent action")
def step(action: Action):
    """
    Submit an agent action. Returns `(observation, reward, done, info)`.

    **action_type**: `label` | `rank` | `triage`
    """
    try:
        result = _env.step(action)
        _metrics[_env.state().get("task_level", "easy")]["step_calls"] += 1
        _metrics[_env.state().get("task_level", "easy")]["total_reward"] += result.reward
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", tags=["env"], summary="Inspect internal environment state")
def state():
    """
    Returns the full internal environment state snapshot.
    Useful for debugging, evaluation, or building agent visualisations.
    """
    return _env.state()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
