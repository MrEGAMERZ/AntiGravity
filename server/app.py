"""
AntiGravity — FastAPI Server (OpenEnv HTTP Interface)
Exposes POST /reset, POST /step, GET /state
"""
from __future__ import annotations

import sys
import os

# Make sure the antigravity package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import Action, Observation, StepResult
from environment.env import AntiGravityEnv


# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AntiGravity Email Triage Environment",
    description=(
        "OpenEnv-compliant HTTP server for the AntiGravity email triage environment. "
        "Exposes reset(), step(), and state() endpoints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per-process)
_env = AntiGravityEnv()


# ─── Request schemas ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_level: str = "easy"   # "easy" | "medium" | "hard"
    seed: Optional[int] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "name": "AntiGravity",
        "description": "Email triage OpenEnv environment",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation, tags=["env"])
def reset(req: ResetRequest):
    """
    Reset the environment and return the initial observation.
    - task_level: "easy" | "medium" | "hard"
    - seed: optional integer for reproducibility
    """
    if req.task_level not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_level '{req.task_level}'. Must be easy, medium, or hard.",
        )
    obs = _env.reset(task_level=req.task_level, seed=req.seed)
    return obs


@app.post("/step", response_model=StepResult, tags=["env"])
def step(action: Action):
    """
    Submit an agent action and receive (observation, reward, done, info).
    """
    try:
        result = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", tags=["env"])
def state():
    """
    Return the full internal environment state (for debugging/evaluation).
    """
    return _env.state()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
