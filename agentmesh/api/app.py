"""FastAPI wrapper over the Orchestrator.

Single long-lived Orchestrator instance per process. Sessions are provided
per-request (so the same process can serve many isolated users).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentmesh import __version__
from agentmesh.config import get_settings
from agentmesh.orchestrator.core import Orchestrator
from agentmesh.utils.logging import get_logger, setup_logging
from agentmesh.utils.types import AgentResult

log = get_logger(__name__)


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="Natural-language task")
    session_id: str = Field(default="default", description="Memory scope key")
    critic_enabled: bool | None = Field(
        default=None,
        description="Override the server-default critic setting for this call",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    planner_model: str
    critic_model: str
    tools: list[str]


_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _orchestrator
    setup_logging()
    settings = get_settings()
    _orchestrator = Orchestrator.from_settings(settings)
    await _orchestrator.initialize()
    log.info("api.ready", version=__version__)
    try:
        yield
    finally:
        if _orchestrator is not None:
            await _orchestrator.close()
            _orchestrator = None


app = FastAPI(
    title="AgentMesh",
    version=__version__,
    description="MCP-native multi-agent orchestrator with hierarchical memory.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _orchestrator is None:
        raise HTTPException(503, "orchestrator not initialised")
    s = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        planner_model=s.planner_model,
        critic_model=s.critic_model,
        tools=_orchestrator.tools.list_tool_names(),
    )


@app.post("/run", response_model=AgentResult)
async def run(req: RunRequest) -> AgentResult:
    if _orchestrator is None:
        raise HTTPException(503, "orchestrator not initialised")

    if req.critic_enabled is not None:
        previous = _orchestrator.critic_enabled
        _orchestrator.critic_enabled = req.critic_enabled
        try:
            return await _orchestrator.run(task=req.task, session_id=req.session_id)
        finally:
            _orchestrator.critic_enabled = previous

    return await _orchestrator.run(task=req.task, session_id=req.session_id)
