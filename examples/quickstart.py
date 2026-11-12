"""Quickstart: two agents, one approval gate. Run with: python examples/quickstart.py"""

from agentgate import Agent, Orchestrator, Playbook, RulePlanner, StepSpec, tool


@tool(description="Look up a customer's outstanding balance")
def get_balance(customer: str) -> float:
    return {"acme": 12_500.0, "globex": 800.0}[customer]


@tool(description="Issue a refund", requires_approval=lambda a: a["amount"] > 1_000)
def refund(customer: str, amount: float) -> str:
    return f"refunded {amount:.2f} to {customer}"


planner = RulePlanner(
    [
        Playbook(
            "refund",
            r"refund (\w+)",
            lambda m: [
                StepSpec("bal", "support", "get_balance", {"customer": m[1]}),
                StepSpec("pay", "finance", "refund", {"customer": m[1], "amount": "${bal}"}, ["bal"]),
            ],
        )
    ]
)
agents = [Agent("support", "support agent", [get_balance]), Agent("finance", "finance agent", [refund])]
orch = Orchestrator(agents, planner)

small = orch.run("refund globex")
print("globex:", small.status.value, "->", small.steps["pay"].output)

big = orch.run("refund acme")
print("acme:  ", big.status.value)
approval = orch.pending_approvals()[0]
print("        approval needed:", approval.tool, approval.args)
big = orch.decide(approval.id, approve=True, by="finance-lead")
print("acme:  ", big.status.value, "->", big.steps["pay"].output)
print("metrics:", orch.metrics.snapshot()["counters"])
