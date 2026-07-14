"""Orchestrator: AggregateInput -> ToolSelection for the Tool Executor.

Runs inference at low temperature to produce a candidate tool name, then uses
fuzzywuzzy to match that free-text output to the closest valid tool name in
the toolbox. The toolbox is injectable (defaults to the full registry) so
experiments can vary the number of available tools and measure the effect on
selection accuracy.
"""

from __future__ import annotations

from fuzzywuzzy import process as fuzzy_process

from vish_agent.config import FUZZY_MATCH_THRESHOLD, ORCHESTRATOR_TEMPERATURE
from vish_agent.llm.client import LLMClient
from vish_agent.logging_utils import TransactionLogger, default_logger
from vish_agent.prompts import TOOL_SELECTION_PROMPT
from vish_agent.schemas import AggregateInput, ToolSelection
from vish_agent.tools.base import ToolSpec
from vish_agent.tools.registry import TOOLBOX


class Orchestrator:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        logger: TransactionLogger = default_logger,
        tools: dict[str, ToolSpec] | None = None,
        fuzzy_match_threshold: int = FUZZY_MATCH_THRESHOLD,
    ):
        self.llm_client = llm_client or LLMClient.get_shared()
        self.logger = logger
        self.tools = tools if tools is not None else TOOLBOX
        self.fuzzy_match_threshold = fuzzy_match_threshold

    def select_tool(self, aggregate_input: AggregateInput) -> ToolSelection:
        tool_names = list(self.tools.keys())
        if aggregate_input.conversation_context:
            user_input = f"{aggregate_input.conversation_context}\nUser: {aggregate_input.raw_user_input}"
        else:
            user_input = aggregate_input.raw_user_input
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=self._format_tool_descriptions(),
            user_input=user_input,
        )
        print(f"    ORCHESTRATOR: Considering {len(tool_names)} tools: {', '.join(tool_names)}")
        raw_model_output = self.llm_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=ORCHESTRATOR_TEMPERATURE,
            max_new_tokens=16,
        )
        print(f"    ORCHESTRATOR: Raw tool match output: {raw_model_output}")
        matched_tool_name, match_score = self._match_tool_name(raw_model_output, tool_names)
        if matched_tool_name is None:
            print(
                f"    ORCHESTRATOR: No tool selected (best fuzzy score {match_score} "
                f"below threshold {self.fuzzy_match_threshold})"
            )
        else:
            print(
                f"    ORCHESTRATOR: Matched '{raw_model_output.strip()}' -> '{matched_tool_name}' "
                f"(fuzzy score {match_score}, threshold {self.fuzzy_match_threshold})"
            )

        selection = ToolSelection(
            request_id=aggregate_input.request_id,
            aggregate_input=aggregate_input,
            raw_model_output=raw_model_output,
            matched_tool_name=matched_tool_name,
            match_score=match_score,
        )

        self.logger.log(
            request_id=aggregate_input.request_id,
            layer="orchestrator",
            event="tool_selected",
            data={
                "available_tools": tool_names,
                "num_available_tools": len(tool_names),
                "raw_model_output": raw_model_output,
                "matched_tool_name": matched_tool_name,
                "match_score": match_score,
                "fuzzy_match_threshold": self.fuzzy_match_threshold,
            },
        )
        return selection

    def _format_tool_descriptions(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools.values())

    def _match_tool_name(self, raw_output: str, tool_names: list[str]) -> tuple[str | None, int]:
        normalized = raw_output.strip().strip(".\"'").upper()
        if not tool_names or normalized == "NONE" or normalized.startswith("NONE"):
            return None, 0

        best_match, score = fuzzy_process.extractOne(raw_output, tool_names)
        if score >= self.fuzzy_match_threshold:
            return best_match, score
        return None, score
