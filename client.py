"""
AntiGravity — OpenEnv Client
Provides AntiGravityEnvClient (async) and AntiGravityEnvClientSync.
"""
from __future__ import annotations

from typing import Optional
import httpx

from models import Action, Observation, StepResult


BASE_URL_DEFAULT = "http://localhost:7860"


class AntiGravityEnvClient:
    """Async client for the AntiGravity environment HTTP API."""

    def __init__(self, base_url: str = BASE_URL_DEFAULT) -> None:
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_level: str = "easy", seed: Optional[int] = None) -> Observation:
        assert self._client
        resp = await self._client.post(
            "/reset", json={"task_level": task_level, "seed": seed}
        )
        resp.raise_for_status()
        return Observation(**resp.json())

    async def step(self, action: Action) -> StepResult:
        assert self._client
        resp = await self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult(**resp.json())

    async def state(self) -> dict:
        assert self._client
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def sync(self) -> "AntiGravityEnvClientSync":
        return AntiGravityEnvClientSync(self.base_url)


class AntiGravityEnvClientSync:
    """Synchronous client for the AntiGravity environment HTTP API."""

    def __init__(self, base_url: str = BASE_URL_DEFAULT) -> None:
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None

    def __enter__(self):
        self._client = httpx.Client(base_url=self.base_url, timeout=30)
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()

    def reset(self, task_level: str = "easy", seed: Optional[int] = None) -> Observation:
        assert self._client
        resp = self._client.post(
            "/reset", json={"task_level": task_level, "seed": seed}
        )
        resp.raise_for_status()
        return Observation(**resp.json())

    def step(self, action: Action) -> StepResult:
        assert self._client
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> dict:
        assert self._client
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()
