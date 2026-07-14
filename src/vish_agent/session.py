"""Per-conversation state carried across VishPipeline.run() calls.

A CLI (or, later, a REST layer keying one session per client/conversation
id) creates a ConversationSession and passes it into run() on every turn.
Tool results are rendered as `key=value` pairs so the existing regex-based
argument extraction in the Tool Executor can pick values straight out of
history the same way it reads the current turn's text - e.g. a location
resolved by ip_api_geolocation in turn 1 satisfies nws_weather_radio's
latitude/longitude in turn 2 without the user repeating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from vish_agent.config import MAX_HISTORY_TURNS, MAX_TOOL_RESULT_CONTEXT_CHARS


@dataclass
class Turn:
    user_input: str
    tool_name: Optional[str]
    tool_success: bool
    tool_result: Any
    response_text: str


@dataclass
class ConversationSession:
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = MAX_HISTORY_TURNS

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def context_text(self) -> str:
        """Prior turns as plain text, oldest first, current turn's caller
        appends the live message after this. Extraction methods that care
        about recency should search for the *last* match in the combined
        text, since this renders oldest-to-newest."""
        lines: list[str] = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_input}")
            if turn.tool_name:
                status = "succeeded" if turn.tool_success else "failed"
                lines.append(f"[{turn.tool_name} {status}: {self._format_result(turn.tool_result)}]")
            lines.append(f"Assistant: {turn.response_text}")
        return "\n".join(lines)

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, dict):
            text = ", ".join(f"{key}={value}" for key, value in result.items())
        else:
            text = str(result)

        if len(text) > MAX_TOOL_RESULT_CONTEXT_CHARS:
            omitted = len(text) - MAX_TOOL_RESULT_CONTEXT_CHARS
            text = f"{text[:MAX_TOOL_RESULT_CONTEXT_CHARS]}... [truncated, {omitted} more chars]"
        return text
