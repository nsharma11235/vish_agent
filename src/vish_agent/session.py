"""Per-conversation state carried across VishPipeline.run() calls.

A CLI (or, later, a REST layer keying one session per client/conversation
id) uses ConversationSession.create() to get a session paired with a
VishPipeline wired to the same toolset, then passes the session into that
pipeline's run() on every turn. Building both together means parallel
conversations can each carry their own toolset without the caller having to
hand-match a ConversationSession to the right VishPipeline instance.
Tool results are rendered as `key=value` pairs so the existing regex-based
argument extraction in the Tool Executor can pick values straight out of
history the same way it reads the current turn's text - e.g. a location
resolved by ip_api_geolocation in turn 1 satisfies nws_weather_radio's
latitude/longitude in turn 2 without the user repeating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from vish_agent.config import MAX_HISTORY_TURNS, MAX_TOOL_RESULT_CONTEXT_CHARS, RESPONSE_TEMPERATURE
from vish_agent.llm.client import LLMClient
from vish_agent.logging_utils import TransactionLogger, default_logger
from vish_agent.prompts import CONVERSATION_START_PROMPT
from vish_agent.tools.base import ToolSpec
from vish_agent.tools.registry import TOOLBOX

if TYPE_CHECKING:
    # Deferred: pipeline.py imports ConversationSession, so importing
    # VishPipeline at module level here would be circular.
    from vish_agent.pipeline import VishPipeline


@dataclass
class Turn:
    user_input: str
    tool_name: Optional[str]
    tool_success: bool
    tool_result: Any
    response_text: str


@dataclass
class VishConversationSession:
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = MAX_HISTORY_TURNS

    @classmethod
    def create(
        cls,
        tools: dict[str, ToolSpec] | None = None,
        llm_client: LLMClient | None = None,
        logger: TransactionLogger = default_logger,
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> tuple["VishConversationSession", "VishPipeline"]:
        """Build a session and a VishPipeline wired to the same toolset, so
        parallel conversations can each run their own tools without the
        caller having to hand-match a session to the right pipeline."""
        from vish_agent.pipeline import VishPipeline

        session = cls(max_turns=max_turns)
        pipeline = VishPipeline(llm_client=llm_client, logger=logger, tools=tools)
        return session, pipeline

    def start_session(
        self,
        llm_client: LLMClient | None = None,
        tools: dict[str, ToolSpec] | None = None,
    ) -> str:
        """Greet the user and introduce the tools available this session."""
        llm_client = llm_client or LLMClient.get_shared()
        tools = tools if tools is not None else TOOLBOX
        toolbox_description = ", ".join(tool.name for tool in tools.values())

        prompt = CONVERSATION_START_PROMPT.format(toolbox=toolbox_description)
        greeting = llm_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=RESPONSE_TEMPERATURE,
            max_new_tokens=256,
        )
        return greeting

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    # Add to future versions
    # def context_text(self) -> str:
    #     """Prior turns as plain text, oldest first, current turn's caller
    #     appends the live message after this. Extraction methods that care
    #     about recency should search for the *last* match in the combined
    #     text, since this renders oldest-to-newest."""
    #     lines: list[str] = []
    #     for turn in self.turns:
    #         lines.append(f"User: {turn.user_input}")
    #         if turn.tool_name:
    #             status = "succeeded" if turn.tool_success else "failed"
    #             lines.append(f"[{turn.tool_name} {status}: {self._format_result(turn.tool_result)}]")
    #         lines.append(f"Assistant: {turn.response_text}")
    #     return "\n".join(lines)
    #
    # @staticmethod
    # def _format_result(result: Any) -> str:
    #     if isinstance(result, dict):
    #         text = ", ".join(f"{key}={value}" for key, value in result.items())
    #     else:
    #         text = str(result)
    #
    #     if len(text) > MAX_TOOL_RESULT_CONTEXT_CHARS:
    #         omitted = len(text) - MAX_TOOL_RESULT_CONTEXT_CHARS
    #         text = f"{text[:MAX_TOOL_RESULT_CONTEXT_CHARS]}... [truncated, {omitted} more chars]"
    #     return text
