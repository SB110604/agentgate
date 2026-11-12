"""Planning: turn a task into a dependency graph of tool calls assigned to agents.

* :class:`RulePlanner` — deterministic playbooks (regex -> steps); no LLM needed.
* :class:`LLMPlanner` — asks an LLM for a JSON plan, validates it against the agents and tools, and retries once
  with the validation error if the plan is invalid.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .llm import LLM
from .memory import ConversationMemory


class PlanError(Exception):
    pass


@dataclass
class StepSpec:
    id: str
    agent: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    task: str
    steps: list[StepSpec]
    planner: str = ""


class AgentInfo(Protocol):
    name: str
    role: str
    tool_names: list[str]


def validate_plan(plan: Plan, agents: dict[str, Any], tool_names: set[str]) -> Plan:
    """Check ids are unique, dependencies exist and are acyclic, and every agent may use its tool."""
    if not plan.steps:
        raise PlanError("plan has no steps")
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate step ids")
    for s in plan.steps:
        if s.agent not in agents:
            raise PlanError(f"step {s.id}: unknown agent {s.agent!r}")
        if s.tool not in tool_names:
            raise PlanError(f"step {s.id}: unknown tool {s.tool!r}")
        if s.tool not in agents[s.agent].tool_names:
            raise PlanError(f"step {s.id}: agent {s.agent!r} is not allowed to use {s.tool!r}")
        for d in s.depends_on:
            if d not in ids:
                raise PlanError(f"step {s.id}: depends on unknown step {d!r}")
    seen: set[str] = set()
    deps = {s.id: set(s.depends_on) for s in plan.steps}
    while deps:
        ready = [k for k, v in deps.items() if v <= seen]
        if not ready:
            raise PlanError(f"dependency cycle between steps {sorted(deps)}")
        for k in ready:
            seen.add(k)
            del deps[k]
    return plan


class Planner(Protocol):
    def plan(self, task: str, agents: dict[str, Any], tools: list[dict[str, Any]]) -> Plan: ...


@dataclass
class Playbook:
    name: str
    pattern: str
    build: Callable[[re.Match[str]], list[StepSpec]]


class RulePlanner:
    def __init__(self, playbooks: list[Playbook]) -> None:
        self.playbooks = playbooks

    def plan(self, task: str, agents: dict[str, Any], tools: list[dict[str, Any]]) -> Plan:
        for pb in self.playbooks:
            m = re.search(pb.pattern, task, re.IGNORECASE)
            if m:
                return validate_plan(Plan(task, pb.build(m), f"rule:{pb.name}"), agents, {t["name"] for t in tools})
        raise PlanError(f"no playbook matches task: {task!r}")


_SYSTEM = """You are the planner of a multi-agent system. Break the user's task into tool calls.
Agents and the tools each may use:
{agents}
Tool schemas:
{tools}
Reply with JSON only, no prose:
{{"steps": [{{"id": "s1", "agent": "<agent>", "tool": "<tool>", "args": {{...}}, "depends_on": []}}]}}
A later step may use an earlier output by writing "${{s1}}" (whole output) or "${{s1.field}}" as an argument value,
and must then list that step in depends_on. Use only the agents and tools listed."""


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise PlanError("planner reply contained no JSON object")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise PlanError(f"planner reply is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError('planner JSON must be an object with a "steps" list')
    return data


class LLMPlanner:
    def __init__(self, llm: LLM, memory: ConversationMemory | None = None, max_attempts: int = 2) -> None:
        self.llm, self.memory, self.max_attempts = llm, memory, max_attempts

    def plan(self, task: str, agents: dict[str, Any], tools: list[dict[str, Any]]) -> Plan:
        agent_txt = "\n".join(f"- {a.name} ({a.role}): {', '.join(a.tool_names)}" for a in agents.values())
        messages = [{"role": "system", "content": _SYSTEM.format(agents=agent_txt, tools=json.dumps(tools))}]
        if self.memory:
            messages += self.memory.messages()
        messages.append({"role": "user", "content": task})
        error: PlanError | None = None
        for _ in range(self.max_attempts):
            reply = self.llm.complete(messages)
            try:
                data = _extract_json(reply)
                steps = [
                    StepSpec(
                        str(s["id"]),
                        str(s["agent"]),
                        str(s["tool"]),
                        dict(s.get("args") or {}),
                        [str(d) for d in s.get("depends_on") or []],
                    )
                    for s in data["steps"]
                ]
                plan = validate_plan(Plan(task, steps, "llm"), agents, {t["name"] for t in tools})
                if self.memory:
                    self.memory.add("user", task)
                    self.memory.add("assistant", reply)
                return plan
            except (PlanError, KeyError, TypeError) as e:
                error = e if isinstance(e, PlanError) else PlanError(f"malformed step: {e}")
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That plan is invalid: {error}. Reply with a corrected JSON plan."},
                ]
        raise error or PlanError("planning failed")
