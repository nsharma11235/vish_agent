"""Data contracts passed between pipeline layers.

Keeping these in one module (rather than importing layer-to-layer) is what
lets a future REST API layer sit in front of the pipeline without any of the
layers needing to change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ResponseType(str, Enum):
    """What kind of answer the Input Layer expects the pipeline to produce."""

    CONVERSATIONAL = "conversational"
    TOOL_ASSISTED = "tool_assisted"


def new_request_id() -> str:
    return uuid.uuid4().hex


@dataclass
class AggregateInput:
    """Output of the Input Layer; input to the Orchestrator."""

    request_id: str
    raw_user_input: str
    system_prompt: str
    aggregate_text: str
    expected_response_type: ResponseType
    # Prior turns of the conversation, rendered as plain text (empty for a
    # fresh conversation or a single-shot request). Lets later layers resolve
    # references like "there" back to a location established earlier.
    conversation_context: str = ""


@dataclass
class ToolSelection:
    """Output of the Orchestrator; input to the Tool Executor."""

    request_id: str
    aggregate_input: AggregateInput
    raw_model_output: str
    matched_tool_name: Optional[str]
    match_score: int


@dataclass
class ToolResult:
    """Output of the Tool Executor; input to the Response Generator."""

    request_id: str
    tool_name: Optional[str]
    arguments: dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None


@dataclass
class FinalResponse:
    """Output of the Response Generator."""

    request_id: str
    text: str