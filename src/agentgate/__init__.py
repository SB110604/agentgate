"""agentgate: multi-agent orchestration with planning, typed tools, memory and human approval gates."""

from .llm import LLM, LLMError, OpenAICompatibleLLM
from .memory import ConversationMemory, LongTermMemory
from .monitoring import Metrics, Tracer
from .orchestrator import Agent, Approval, ApprovalError, ApprovalStatus, Orchestrator, Run, RunStatus, StepStatus
from .planner import LLMPlanner, Plan, PlanError, Playbook, RulePlanner, StepSpec
from .tools import Tool, ToolError, ToolRegistry, tool

__version__ = "0.1.0"
__all__ = [
    "LLM",
    "LLMError",
    "OpenAICompatibleLLM",
    "ConversationMemory",
    "LongTermMemory",
    "Metrics",
    "Tracer",
    "Agent",
    "Approval",
    "ApprovalError",
    "ApprovalStatus",
    "Orchestrator",
    "Run",
    "RunStatus",
    "StepStatus",
    "LLMPlanner",
    "Plan",
    "PlanError",
    "Playbook",
    "RulePlanner",
    "StepSpec",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "tool",
    "__version__",
]
