"""Response Generator: ToolSelection + ToolResult -> FinalResponse.

Combines the original user prompt, the Input Layer's expected response type,
and the Tool Executor's result into a single prompt, runs inference, and
returns a natural language response. Also writes a single consolidated
"full_transaction" log record spanning all four layers, keyed by
request_id, for experiment evaluation.
"""

from __future__ import annotations

from vish_agent.config import RESPONSE_TEMPERATURE
from vish_agent.llm.qwen_client import QwenClient
from vish_agent.logging_utils import TransactionLogger, default_logger
from vish_agent.prompts import RESPONSE_GENERATION_PROMPT
from vish_agent.schemas import FinalResponse, ToolResult, ToolSelection


class ResponseGenerator:
    def __init__(self, llm_client: QwenClient | None = None, logger: TransactionLogger = default_logger):
        self.llm_client = llm_client or QwenClient.get_shared()
        self.logger = logger

    def generate(self, tool_selection: ToolSelection, tool_result: ToolResult) -> FinalResponse:
        aggregate_input = tool_selection.aggregate_input

        conversation_context = (
            f"Conversation so far:\n{aggregate_input.conversation_context}\n\n"
            if aggregate_input.conversation_context
            else ""
        )
        prompt = RESPONSE_GENERATION_PROMPT.format(
            system_prompt=aggregate_input.system_prompt,
            conversation_context=conversation_context,
            user_input=aggregate_input.raw_user_input,
            expected_response_type=aggregate_input.expected_response_type.value,
            tool_name=tool_result.tool_name or "none",
            tool_result=self._format_tool_result(tool_result),
        )
        response_text = self.llm_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=RESPONSE_TEMPERATURE,
            max_new_tokens=256,
        )

        final_response = FinalResponse(request_id=aggregate_input.request_id, text=response_text)

        self.logger.log(
            request_id=aggregate_input.request_id,
            layer="response_generator",
            event="full_transaction",
            data={
                "raw_user_input": aggregate_input.raw_user_input,
                "system_prompt": aggregate_input.system_prompt,
                "conversation_context": aggregate_input.conversation_context,
                "expected_response_type": aggregate_input.expected_response_type.value,
                "orchestrator_raw_output": tool_selection.raw_model_output,
                "matched_tool_name": tool_selection.matched_tool_name,
                "match_score": tool_selection.match_score,
                "tool_arguments": tool_result.arguments,
                "tool_success": tool_result.success,
                "tool_result": tool_result.result if tool_result.success else None,
                "tool_error": tool_result.error,
                "final_response_text": response_text,
            },
        )
        return final_response

    @staticmethod
    def _format_tool_result(tool_result: ToolResult) -> str:
        if tool_result.tool_name is None:
            return "N/A (no tool was used)"
        if not tool_result.success:
            return f"Tool '{tool_result.tool_name}' failed: {tool_result.error}"
        return str(tool_result.result)
