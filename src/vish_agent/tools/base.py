"""Shared toolbox metadata schema.

The Orchestrator matches model output against ToolSpec.name (via fuzzywuzzy).
The Tool Executor calls ToolSpec.extract_arguments to pull argument values
out of conversation text (each tool owns its own extraction rules, defined
alongside it in its own module) and then ToolSpec.func to run the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ArgumentExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: type
    required: bool
    description: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    func: Callable[..., Any]
    # (tool, current_turn_text, history_and_current_text) -> {param_name: value}
    extract_arguments: Callable[["ToolSpec", str, str], dict[str, Any]]

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def required_parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters if p.required)


def cast_param(tool: ToolSpec, param_name: str, raw_value: Any) -> Any:
    """Generic, type-driven casting + sanitization shared by every tool's
    extract_arguments. Tool-specific validation (value ranges, formats)
    belongs in the tool's own extraction function, after casting."""
    param = next(p for p in tool.parameters if p.name == param_name)
    try:
        if param.type is float:
            value: Any = float(raw_value)
        elif param.type is int:
            value = int(raw_value)
        elif param.type is str:
            value = str(raw_value).strip()
        else:
            value = raw_value
    except (TypeError, ValueError) as exc:
        raise ArgumentExtractionError(
            f"Argument '{param_name}' for tool '{tool.name}' could not be parsed as {param.type.__name__}: {raw_value!r}"
        ) from exc

    if param.type is str and not (1 <= len(value) <= 512):
        raise ArgumentExtractionError(f"Argument '{param_name}' for tool '{tool.name}' has invalid length.")

    return value
