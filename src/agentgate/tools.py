"""Typed tools: a Python function plus a validated argument schema, risk flags and execution limits."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

ApprovalRule = bool | Callable[[dict[str, Any]], bool]


class ToolError(Exception):
    """Raised when a tool cannot be called (unknown tool, invalid arguments, not permitted)."""


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    args_model: type[BaseModel]
    requires_approval: ApprovalRule = False
    retries: int | None = None
    timeout: float | None = None
    tags: list[str] = field(default_factory=list)

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.args_model(**args).model_dump()
        except ValidationError as e:
            raise ToolError(f"invalid arguments for {self.name}: {e.errors(include_url=False)}") from e

    def needs_approval(self, args: dict[str, Any]) -> bool:
        rule = self.requires_approval
        return bool(rule(args)) if callable(rule) else bool(rule)

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.args_model.model_json_schema()}

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**self.validate(kwargs))


def tool(
    name: str | None = None,
    *,
    description: str | None = None,
    requires_approval: ApprovalRule = False,
    retries: int | None = None,
    timeout: float | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Turn a type-hinted function into a :class:`Tool`. Arguments are validated with pydantic before every call.

    >>> @tool(description="Add two numbers")
    ... def add(a: int, b: int) -> int:
    ...     return a + b
    >>> add(a=2, b="3")
    5
    """

    def wrap(func: Callable[..., Any]) -> Tool:
        hints = get_type_hints(func)
        fields: dict[str, Any] = {}
        for p in inspect.signature(func).parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(f"tool {func.__name__}: *args/**kwargs are not supported")
            ann = hints.get(p.name, Any)
            fields[p.name] = (ann, ... if p.default is p.empty else p.default)
        model = create_model(f"{func.__name__}_args", **fields)  # type: ignore[call-overload]
        doc = (inspect.getdoc(func) or "").split("\n\n")[0]
        return Tool(
            name or func.__name__,
            description or doc or func.__name__,
            func,
            model,
            requires_approval,
            retries,
            timeout,
            tags or [],
        )

    return wrap


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> Tool:
        if t.name in self._tools:
            raise ToolError(f"tool {t.name!r} is already registered")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]
