import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from agentgate import (
    Agent,
    ApprovalError,
    ConversationMemory,
    LLMPlanner,
    LongTermMemory,
    Orchestrator,
    PlanError,
    Playbook,
    RulePlanner,
    RunStatus,
    StepSpec,
    StepStatus,
    ToolError,
    tool,
)
from agentgate.api import create_app
from agentgate.orchestrator import resolve_args


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="Double a number")
def double(x: int) -> int:
    return 2 * x


@tool(description="Transfer money", requires_approval=lambda a: a["amount"] > 100)
def transfer(to: str, amount: float) -> dict:
    return {"to": to, "amount": amount}


class Fixed:
    """Planner that always returns the given steps."""

    def __init__(self, steps):
        self.steps = steps

    def plan(self, task, agents, tools):
        from agentgate.planner import Plan, validate_plan

        return validate_plan(Plan(task, self.steps, "fixed"), agents, {t["name"] for t in tools})


def orch(steps, tools=(add, double, transfer), **kw):
    return Orchestrator([Agent("worker", "does maths", list(tools))], Fixed(steps), backoff=0, **kw)


# ---- tools -----------------------------------------------------------------------------------------------------------
def test_tool_validates_and_coerces_arguments():
    assert add(a="2", b=3) == 5
    with pytest.raises(ToolError):
        add(a="two", b=3)
    assert add.schema()["parameters"]["required"] == ["a", "b"]


def test_approval_rule_is_evaluated_on_arguments():
    assert transfer.needs_approval({"to": "x", "amount": 500}) and not transfer.needs_approval({"to": "x", "amount": 5})


def test_resolve_args_supports_whole_values_fields_and_interpolation():
    out = {"s1": {"name": "rice", "items": [10, 20]}}
    assert resolve_args({"a": "${s1.name}", "b": "${s1.items.1}", "c": "buy ${s1.name}", "d": "${s1}"}, out) == {
        "a": "rice",
        "b": 20,
        "c": "buy rice",
        "d": out["s1"],
    }


# ---- planning --------------------------------------------------------------------------------------------------------
def test_rule_planner_and_plan_validation():
    pb = Playbook("sum", r"add (\d+) and (\d+)", lambda m: [StepSpec("s1", "worker", "add", {"a": m[1], "b": m[2]})])
    o = Orchestrator([Agent("worker", "maths", [add])], RulePlanner([pb]))
    assert o.run("please add 2 and 40").steps["s1"].output == 42
    with pytest.raises(PlanError):
        o.run("unrelated task")


@pytest.mark.parametrize(
    "steps,msg",
    [
        ([StepSpec("a", "worker", "nope")], "unknown tool"),
        ([StepSpec("a", "ghost", "add")], "unknown agent"),
        ([StepSpec("a", "worker", "add", depends_on=["b"]), StepSpec("b", "worker", "add", depends_on=["a"])], "cycle"),
        ([StepSpec("a", "worker", "add", depends_on=["zzz"])], "unknown step"),
    ],
)
def test_invalid_plans_are_rejected(steps, msg):
    with pytest.raises(PlanError, match=msg):
        orch(steps).run("t")


def test_agent_cannot_use_another_agents_tool():
    o = Orchestrator(
        [Agent("a", "x", [add]), Agent("b", "y", [double])], Fixed([StepSpec("s", "a", "double", {"x": 1})])
    )
    with pytest.raises(PlanError, match="not allowed"):
        o.run("t")


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


def test_llm_planner_retries_with_validation_feedback():
    bad = json.dumps({"steps": [{"id": "s1", "agent": "worker", "tool": "multiply", "args": {}}]})
    good = "Here you go: " + json.dumps(
        {
            "steps": [
                {"id": "s1", "agent": "worker", "tool": "add", "args": {"a": 1, "b": 2}},
                {"id": "s2", "agent": "worker", "tool": "double", "args": {"x": "${s1}"}, "depends_on": ["s1"]},
            ]
        }
    )
    llm, mem = FakeLLM([bad, good]), ConversationMemory()
    o = Orchestrator([Agent("worker", "maths", [add, double])], LLMPlanner(llm, mem))
    run = o.run("add 1 and 2 then double it")
    assert run.status is RunStatus.SUCCEEDED and run.steps["s2"].output == 6
    assert "unknown tool" in llm.calls[1][-1]["content"]
    assert mem.messages()[0] == {"role": "user", "content": "add 1 and 2 then double it"}


def test_llm_planner_gives_up_after_max_attempts():
    o = Orchestrator([Agent("worker", "maths", [add])], LLMPlanner(FakeLLM(["no json", "still none"])))
    with pytest.raises(PlanError):
        o.run("t")


# ---- execution -------------------------------------------------------------------------------------------------------
def test_dependencies_pass_outputs_and_independent_steps_run_in_parallel():
    seen = []

    @tool(description="slow")
    def slow(n: int) -> int:
        seen.append(threading.get_ident())
        time.sleep(0.3)
        return n

    steps = [
        StepSpec("a", "worker", "slow", {"n": 1}),
        StepSpec("b", "worker", "slow", {"n": 2}),
        StepSpec("c", "worker", "add", {"a": "${a}", "b": "${b}"}, ["a", "b"]),
    ]
    t0 = time.perf_counter()
    run = orch(steps, tools=(slow, add)).run("t")
    assert run.status is RunStatus.SUCCEEDED and run.steps["c"].output == 3
    assert time.perf_counter() - t0 < 0.55 and len(set(seen)) == 2


def test_retries_then_succeeds():
    calls = {"n": 0}

    @tool(description="flaky", retries=2)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temporary")
        return "ok"

    run = orch([StepSpec("s", "worker", "flaky")], tools=(flaky,)).run("t")
    assert run.steps["s"].output == "ok" and run.steps["s"].attempts == 3


def test_failure_skips_dependents_and_is_traced():
    @tool(description="boom", retries=0)
    def boom() -> None:
        raise RuntimeError("broken")

    o = orch(
        [StepSpec("x", "worker", "boom"), StepSpec("y", "worker", "double", {"x": 1}, ["x"])], tools=(boom, double)
    )
    run = o.run("t")
    assert run.status is RunStatus.FAILED and run.steps["y"].status is StepStatus.SKIPPED
    assert "RuntimeError: broken" in run.steps["x"].error
    assert [e["type"] for e in o.tracer.events(run.id)][-1] == "run_finished"


def test_timeout_is_enforced():
    @tool(description="hangs", retries=0, timeout=0.2)
    def hang() -> None:
        time.sleep(2)

    run = orch([StepSpec("s", "worker", "hang")], tools=(hang,)).run("t")
    assert run.status is RunStatus.FAILED and "timed out" in run.steps["s"].error


def test_bad_reference_fails_step_without_crashing():
    run = orch(
        [StepSpec("a", "worker", "add", {"a": 1, "b": 1}), StepSpec("b", "worker", "double", {"x": "${a.nope}"}, ["a"])]
    ).run("t")
    assert run.steps["b"].status is StepStatus.FAILED and run.steps["b"].error.startswith("bad arguments")


# ---- human approval --------------------------------------------------------------------------------------------------
def test_approval_pauses_run_and_approve_resumes_it():
    o = orch(
        [
            StepSpec("t", "worker", "transfer", {"to": "acme", "amount": 500}),
            StepSpec("d", "worker", "double", {"x": 2}),
        ]
    )
    run = o.run("pay")
    assert run.status is RunStatus.WAITING_APPROVAL and run.steps["d"].status is StepStatus.SUCCEEDED
    [ap] = o.pending_approvals()
    assert ap.args == {"to": "acme", "amount": 500.0}
    run = o.decide(ap.id, True, by="manager")
    assert run.status is RunStatus.SUCCEEDED and run.steps["t"].output["amount"] == 500.0
    with pytest.raises(ApprovalError, match="already"):
        o.decide(ap.id, True, by="manager")


def test_small_amounts_skip_approval_and_rejection_needs_reason():
    o = orch([StepSpec("t", "worker", "transfer", {"to": "acme", "amount": 50})])
    assert o.run("pay").status is RunStatus.SUCCEEDED and not o.pending_approvals()
    o2 = orch(
        [
            StepSpec("t", "worker", "transfer", {"to": "acme", "amount": 900}),
            StepSpec("n", "worker", "double", {"x": 1}, ["t"]),
        ]
    )
    run = o2.run("pay")
    ap = o2.pending_approvals()[0]
    with pytest.raises(ApprovalError, match="reason"):
        o2.decide(ap.id, False, by="manager")
    run = o2.decide(ap.id, False, by="manager", reason="vendor not verified")
    assert run.status is RunStatus.REJECTED and run.steps["n"].status is StepStatus.SKIPPED
    assert o2.metrics.snapshot()["counters"]["approvals_rejected"] == 1


# ---- memory and monitoring -------------------------------------------------------------------------------------------
def test_long_term_memory_persists_and_ranks(tmp_path):
    path = str(tmp_path / "mem.db")
    m = LongTermMemory(path)
    m.remember("pref:supplier", {"sku": "rice", "supplier": "Agarwal"}, tags=["preference"])
    m.remember("note", "oil prices rise in October", tags=["market"])
    m2 = LongTermMemory(path)
    assert len(m2) == 2 and m2.get("pref:supplier")["supplier"] == "Agarwal"
    assert m2.recall("which supplier for rice")[0].key == "pref:supplier"
    assert m2.recall("oil", tag="preference") == [] and m2.forget("note") and len(m2) == 1


def test_conversation_memory_window():
    c = ConversationMemory(max_messages=2)
    for i in range(3):
        c.add("user", str(i))
    assert [m["content"] for m in c.messages()] == ["1", "2"]


def test_successful_runs_are_remembered_and_metrics_recorded():
    mem = LongTermMemory()
    o = orch([StepSpec("s", "worker", "add", {"a": 1, "b": 2})], memory=mem)
    run = o.run("add things")
    assert mem.get(f"run:{run.id}")["outputs"]["s"] == 3
    snap = o.metrics.snapshot()
    assert snap["counters"]["runs_succeeded"] == 1 and snap["tools"]["add"]["calls"] == 1


# ---- HTTP API --------------------------------------------------------------------------------------------------------
def test_api_end_to_end_with_procurement_example():
    from examples.procurement import ORDERS, build

    ORDERS.clear()
    c = TestClient(create_app(build(), api_key="secret"))
    assert c.get("/health").status_code == 200
    assert c.post("/runs", json={"task": "restock SKU-1 x 120"}).status_code == 401
    h = {"X-API-Key": "secret", "X-User": "priya"}
    run = c.post("/runs", json={"task": "restock SKU-1 x 120"}, headers=h).json()
    assert run["status"] == "waiting_approval"
    [ap] = c.get("/approvals", headers=h).json()
    assert ap["args"]["supplier"] == "Pune Wholesale" and ap["args"]["unit_price"] == 398.0
    done = c.post(f"/approvals/{ap['id']}/approve", headers=h).json()
    assert done["status"] == "succeeded" and ORDERS[-1]["total"] == 47760.0
    assert c.post(f"/approvals/{ap['id']}/approve", headers=h).status_code == 409
    assert c.post("/approvals/nope/approve", headers=h).status_code == 404
    assert any(e["type"] == "approval_decided" for e in c.get(f"/runs/{run['id']}/trace", headers=h).json())
    assert c.post("/runs", json={"task": "write a poem"}, headers=h).status_code == 422
    small = c.post("/runs", json={"task": "restock SKU-2 x 10"}, headers=h).json()
    assert small["status"] == "succeeded" and small["steps"][2]["output"]["total"] == 1520.0
    assert c.get("/metrics", headers=h).json()["counters"]["runs_started"] == 3
