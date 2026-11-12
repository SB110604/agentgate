"""FastAPI app exposing runs, human approvals, traces and metrics for an :class:`Orchestrator`."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import ApprovalError, ApprovalStatus, Orchestrator
from .planner import PlanError


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


class Decision(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


def create_app(orch: Orchestrator, *, api_key: str | None = None, title: str = "agentgate") -> FastAPI:
    """Build the HTTP API. If ``api_key`` is set, every request except ``/health`` needs header ``X-API-Key``."""
    app = FastAPI(
        title=title, version="0.1.0", description="Multi-agent runs with human approval gates, traces and metrics."
    )

    def auth(key: str | None) -> None:
        if api_key and key != api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "agents": list(orch.agents), "tools": [t.name for t in orch.registry]}

    @app.post("/runs", status_code=201)
    def start_run(body: RunRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        try:
            return orch.run(body.task).to_dict()
        except PlanError as e:
            raise HTTPException(422, f"could not plan task: {e}") from e

    @app.get("/runs")
    def list_runs(x_api_key: str | None = Header(default=None)) -> list[dict[str, Any]]:
        auth(x_api_key)
        return [
            {"id": r.id, "task": r.task, "status": r.status.value, "created_at": r.created_at}
            for r in sorted(orch.runs.values(), key=lambda r: -r.created_at)
        ]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        run = orch.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return run.to_dict()

    @app.get("/runs/{run_id}/trace")
    def trace(run_id: str, x_api_key: str | None = Header(default=None)) -> list[dict[str, Any]]:
        auth(x_api_key)
        if run_id not in orch.runs:
            raise HTTPException(404, "run not found")
        return orch.tracer.events(run_id)

    @app.get("/approvals")
    def approvals(
        status: ApprovalStatus | None = ApprovalStatus.PENDING, x_api_key: str | None = Header(default=None)
    ) -> list[dict[str, Any]]:
        auth(x_api_key)
        return [asdict(a) for a in orch.approvals.values() if status is None or a.status is status]

    def _decide(approval_id: str, approve: bool, body: Decision, user: str | None) -> dict[str, Any]:
        try:
            return orch.decide(approval_id, approve, by=user or "api", reason=body.reason).to_dict()
        except ApprovalError as e:
            raise HTTPException(404 if "unknown" in str(e) else 409, str(e)) from e

    @app.post("/approvals/{approval_id}/approve")
    def approve(
        approval_id: str,
        body: Decision | None = None,
        x_api_key: str | None = Header(default=None),
        x_user: str | None = Header(default=None),
    ) -> dict[str, Any]:
        auth(x_api_key)
        return _decide(approval_id, True, body or Decision(), x_user)

    @app.post("/approvals/{approval_id}/reject")
    def reject(
        approval_id: str,
        body: Decision,
        x_api_key: str | None = Header(default=None),
        x_user: str | None = Header(default=None),
    ) -> dict[str, Any]:
        auth(x_api_key)
        return _decide(approval_id, False, body, x_user)

    @app.get("/metrics")
    def metrics(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_api_key)
        return orch.metrics.snapshot()

    return app
