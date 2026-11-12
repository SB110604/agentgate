"""Procurement demo: an analyst checks stock, a buyer compares quotes, and large orders need a human approval.

Run the API:   uvicorn examples.procurement:app --reload
Then:          curl -X POST localhost:8000/runs -H "Content-Type: application/json" -d '{"task": "restock SKU-1 x 120"}'
"""

from __future__ import annotations

import os
import re

from agentgate import (
    Agent,
    LLMPlanner,
    LongTermMemory,
    OpenAICompatibleLLM,
    Orchestrator,
    Playbook,
    RulePlanner,
    StepSpec,
    tool,
)
from agentgate.api import create_app

INVENTORY = {
    "SKU-1": {"name": "Basmati rice 5 kg", "on_hand": 14, "reorder_point": 40},
    "SKU-2": {"name": "Sunflower oil 1 L", "on_hand": 90, "reorder_point": 60},
}
QUOTES = {
    "SKU-1": [
        {"supplier": "Agarwal Traders", "unit_price": 410.0, "lead_days": 3},
        {"supplier": "Pune Wholesale", "unit_price": 398.0, "lead_days": 6},
    ],
    "SKU-2": [{"supplier": "Agarwal Traders", "unit_price": 152.0, "lead_days": 2}],
}
ORDERS: list[dict] = []


@tool(description="Current stock and reorder point for a SKU")
def check_inventory(sku: str) -> dict:
    if sku not in INVENTORY:
        raise ValueError(f"unknown SKU {sku}")
    return {"sku": sku, **INVENTORY[sku]}


@tool(description="Pick the best supplier quote for a SKU (price, then lead time)")
def best_quote(sku: str, max_lead_days: int = 7) -> dict:
    options = [q for q in QUOTES.get(sku, []) if q["lead_days"] <= max_lead_days]
    if not options:
        raise ValueError(f"no quote for {sku} within {max_lead_days} days")
    return min(options, key=lambda q: (q["unit_price"], q["lead_days"]))


@tool(description="Place a purchase order", requires_approval=lambda a: a["quantity"] * a["unit_price"] > 20_000)
def place_order(sku: str, quantity: int, supplier: str, unit_price: float) -> dict:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    order = {
        "po": f"PO-{len(ORDERS) + 1:04d}",
        "sku": sku,
        "quantity": quantity,
        "supplier": supplier,
        "total": round(quantity * unit_price, 2),
    }
    ORDERS.append(order)
    return order


def _restock(m: re.Match[str]) -> list[StepSpec]:
    sku, qty = m.group(1).upper(), int(m.group(2))
    return [
        StepSpec("stock", "analyst", "check_inventory", {"sku": sku}),
        StepSpec("quote", "buyer", "best_quote", {"sku": sku}),
        StepSpec(
            "order",
            "buyer",
            "place_order",
            {"sku": sku, "quantity": qty, "supplier": "${quote.supplier}", "unit_price": "${quote.unit_price}"},
            ["stock", "quote"],
        ),
    ]


def build(memory_path: str = ":memory:") -> Orchestrator:
    agents = [
        Agent("analyst", "inventory analyst", [check_inventory]),
        Agent("buyer", "purchasing agent", [best_quote, place_order]),
    ]
    if os.getenv("AGENTGATE_LLM_MODEL"):  # use an LLM planner when one is configured
        planner = LLMPlanner(OpenAICompatibleLLM())
    else:
        planner = RulePlanner([Playbook("restock", r"restock\s+(sku-\d+)\s*x\s*(\d+)", _restock)])
    return Orchestrator(agents, planner, memory=LongTermMemory(memory_path))


app = create_app(build(), api_key=os.getenv("AGENTGATE_API_KEY") or None, title="agentgate procurement demo")
