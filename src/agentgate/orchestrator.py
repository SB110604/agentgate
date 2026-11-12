"""Agents and the orchestrator: plans a task, runs steps in parallel by dependency, pauses for human approval."""

from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .memory import LongTermMemory
from .monitoring import Metrics, Tracer
from .planner import Plan, PlanError, Planner
from .tools import Tool, ToolError, ToolRegistry


@dataclass
class Agent:
    name: str
    role: str
    tools: list[Tool]

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


class StepStatus(str, Enum):
    PENDING = "pending"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class Step:
    id: str
    agent: str
    tool: str
    args: dict[str, Any]
    depends_on: list[str]
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error: str | None = None
    attempts: int = 0
    approved: bool = False
    resolved_args: dict[str, Any] | None = None


@dataclass
class Run:
    id: str
    task: str
    planner: str
    steps: dict[str, Step]
    status: RunStatus = RunStatus.RUNNING
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "planner": self.planner,
            "status": self.status.value,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "steps": [
                {
                    "id": s.id,
                    "agent": s.agent,
                    "tool": s.tool,
                    "args": s.resolved_args or s.args,
                    "depends_on": s.depends_on,
                    "status": s.status.value,
                    "output": s.output,
                    "error": s.error,
                    "attempts": s.attempts,
                }
                for s in self.steps.values()
            ],
        }


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class Approval:
    id: str
    run_id: str
    step_id: str
    agent: str
    tool: str
    args: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    reason: str | None = None
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None


class ApprovalError(Exception):
    pass


_REF = re.compile(r"\$\{([A-Za-z0-9_-]+)(?:\.([A-Za-z0-9_.-]+))?\}")


def _lookup(value: Any, path: str | None) -> Any:
    for part in path.split(".") if path else []:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def resolve_args(args: Any, outputs: dict[str, Any]) -> Any:
    """Replace ``${step}`` / ``${step.field}`` references with earlier step outputs (recursively)."""
    if isinstance(args, dict):
        return {k: resolve_args(v, outputs) for k, v in args.items()}
    if isinstance(args, list):
        return [resolve_args(v, outputs) for v in args]
    if isinstance(args, str):
        full = _REF.fullmatch(args)
        if full:
            return _lookup(outputs[full.group(1)], full.group(2))
        return _REF.sub(lambda m: str(_lookup(outputs[m.group(1)], m.group(2))), args)
    return args


class Orchestrator:
    def __init__(
        self,
        agents: list[Agent],
        planner: Planner,
        *,
        memory: LongTermMemory | None = None,
        tracer: Tracer | None = None,
        metrics: Metrics | None = None,
        default_retries: int = 2,
        default_timeout: float = 30.0,
        backoff: float = 0.2,
        max_parallel: int = 4,
    ) -> None:
        self.agents = {a.name: a for a in agents}
        self.registry = ToolRegistry()
        for a in agents:
            for t in a.tools:
                if t.name not in self.registry:
                    self.registry.register(t)
        self.planner, self.memory = planner, memory
        self.tracer, self.metrics = tracer or Tracer(), metrics or Metrics()
        self.default_retries, self.default_timeout, self.backoff = default_retries, default_timeout, backoff
        self.max_parallel = max_parallel
        self.runs: dict[str, Run] = {}
        self.approvals: dict[str, Approval] = {}
        self._lock = threading.RLock()

    # ---- public API -------------------------------------------------------------------------------------------
    def run(self, task: str) -> Run:
        run_id = uuid.uuid4().hex[:12]
        self.metrics.inc("runs_started")
        try:
            plan: Plan = self.planner.plan(task, self.agents, self.registry.schemas())
        except PlanError:
            self.metrics.inc("plans_failed")
            raise
        steps = {s.id: Step(s.id, s.agent, s.tool, s.args, s.depends_on) for s in plan.steps}
        run = Run(run_id, task, plan.planner, steps)
        with self._lock:
            self.runs[run_id] = run
        self.tracer.emit(run_id, "run_started", task=task, planner=plan.planner, steps=len(steps))
        return self._advance(run)

    def decide(self, approval_id: str, approve: bool, by: str, reason: str | None = None) -> Run:
        with self._lock:
            ap = self.approvals.get(approval_id)
            if ap is None:
                raise ApprovalError(f"unknown approval {approval_id!r}")
            if ap.status is not ApprovalStatus.PENDING:
                raise ApprovalError(f"approval {approval_id} is already {ap.status.value}")
            if not approve and not reason:
                raise ApprovalError("a reason is required to reject")
            ap.status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
            ap.decided_by, ap.reason, ap.decided_at = by, reason, time.time()
            run = self.runs[ap.run_id]
            step = run.steps[ap.step_id]
            if approve:
                step.approved, step.status = True, StepStatus.PENDING
            else:
                step.status, step.error = StepStatus.REJECTED, f"rejected by {by}: {reason}"
        self.metrics.inc("approvals_approved" if approve else "approvals_rejected")
        self.tracer.emit(run.id, "approval_decided", ap.step_id, approved=approve, by=by, reason=reason)
        return self._advance(run)

    def pending_approvals(self) -> list[Approval]:
        with self._lock:
            return [a for a in self.approvals.values() if a.status is ApprovalStatus.PENDING]

    # ---- execution --------------------------------------------------------------------------------------------
    def _advance(self, run: Run) -> Run:
        run.status = RunStatus.RUNNING
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while True:
                ready = self._ready_steps(run)
                if not ready:
                    break
                list(pool.map(lambda s: self._execute(run, s), ready))
        self._finish(run)
        return run

    def _ready_steps(self, run: Run) -> list[Step]:
        ready: list[Step] = []
        with self._lock:
            done_ok = {s.id for s in run.steps.values() if s.status is StepStatus.SUCCEEDED}
            dead = {StepStatus.FAILED, StepStatus.REJECTED, StepStatus.SKIPPED}
            changed = True
            while changed:  # skip everything downstream of a failed or rejected step
                changed = False
                for s in run.steps.values():
                    if s.status is StepStatus.PENDING and any(run.steps[d].status in dead for d in s.depends_on):
                        s.status, s.error, changed = StepStatus.SKIPPED, "a dependency did not succeed", True
                        self.tracer.emit(run.id, "step_skipped", s.id)
            for s in run.steps.values():
                if s.status is not StepStatus.PENDING or not set(s.depends_on) <= done_ok:
                    continue
                try:
                    s.resolved_args = resolve_args(s.args, {k: run.steps[k].output for k in done_ok})
                    tool = self.registry.get(s.tool)
                    args = tool.validate(s.resolved_args)
                except (ToolError, KeyError, IndexError, TypeError) as e:
                    s.status, s.error = StepStatus.FAILED, f"bad arguments: {e}"
                    self.metrics.inc("steps_failed")
                    self.tracer.emit(run.id, "step_failed", s.id, error=s.error)
                    continue
                if not s.approved and tool.needs_approval(args):
                    s.status = StepStatus.WAITING_APPROVAL
                    ap = Approval(uuid.uuid4().hex[:12], run.id, s.id, s.agent, s.tool, args)
                    self.approvals[ap.id] = ap
                    self.metrics.inc("approvals_requested")
                    self.tracer.emit(run.id, "approval_requested", s.id, approval_id=ap.id, tool=s.tool, args=args)
                    continue
                s.status = StepStatus.RUNNING
                ready.append(s)
        return ready

    def _execute(self, run: Run, step: Step) -> None:
        tool = self.registry.get(step.tool)
        retries = self.default_retries if tool.retries is None else tool.retries
        timeout = self.default_timeout if tool.timeout is None else tool.timeout
        args = tool.validate(step.resolved_args or {})
        for attempt in range(retries + 1):
            step.attempts = attempt + 1
            self.tracer.emit(run.id, "tool_called", step.id, tool=tool.name, agent=step.agent, attempt=attempt + 1)
            t0 = time.perf_counter()
            single = ThreadPoolExecutor(max_workers=1)
            try:
                step.output = single.submit(tool.func, **args).result(timeout=timeout)
                ms = (time.perf_counter() - t0) * 1000
                self.metrics.observe(tool.name, ms)
                step.status, step.error = StepStatus.SUCCEEDED, None
                self.metrics.inc("steps_succeeded")
                self.tracer.emit(run.id, "step_succeeded", step.id, duration_ms=round(ms, 2))
                return
            except FutureTimeout:
                step.error = f"timed out after {timeout}s"
            except Exception as e:  # noqa: BLE001 - tool code is user code; record and retry
                step.error = f"{type(e).__name__}: {e}"
            finally:
                single.shutdown(wait=False, cancel_futures=True)
            self.metrics.inc("tool_errors")
            self.tracer.emit(run.id, "tool_error", step.id, error=step.error, attempt=attempt + 1)
            if attempt < retries:
                time.sleep(self.backoff * (2**attempt))
        step.status = StepStatus.FAILED
        self.metrics.inc("steps_failed")
        self.tracer.emit(run.id, "step_failed", step.id, error=step.error)

    def _finish(self, run: Run) -> None:
        states = {s.status for s in run.steps.values()}
        if StepStatus.WAITING_APPROVAL in states:
            run.status = RunStatus.WAITING_APPROVAL
            self.tracer.emit(run.id, "run_paused")
            return
        run.status = (
            RunStatus.FAILED
            if StepStatus.FAILED in states
            else RunStatus.REJECTED
            if StepStatus.REJECTED in states
            else RunStatus.SUCCEEDED
        )
        run.finished_at = time.time()
        self.metrics.inc(f"runs_{run.status.value}")
        self.tracer.emit(run.id, "run_finished", status=run.status.value)
        if self.memory is not None and run.status is RunStatus.SUCCEEDED:
            self.memory.remember(
                f"run:{run.id}",
                {"task": run.task, "outputs": {s.id: s.output for s in run.steps.values()}},
                tags=["run"],
            )
