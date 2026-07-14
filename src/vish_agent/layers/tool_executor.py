"""Tool Executor: ToolSelection -> ToolResult for the Response Generator.

Looks up the selected tool, delegates to its own ToolSpec.extract_arguments
to pull argument values out of conversation text (each tool owns its own
extraction rules, defined alongside it in its own module), and executes the
tool. Argument extraction and execution failures are captured on ToolResult
rather than raised, so the Response Generator can still produce an answer
(and the failure is preserved in the logs).
"""

from __future__ import annotations

from vish_agent.logging_utils import TransactionLogger, default_logger
from vish_agent.schemas import ToolResult, ToolSelection
from vish_agent.tools.base import ArgumentExtractionError, ToolSpec
from vish_agent.tools.registry import TOOLBOX


class ToolExecutor:
    def __init__(self, logger: TransactionLogger = default_logger, tools: dict[str, ToolSpec] | None = None):
        self.logger = logger
        self.tools = tools if tools is not None else TOOLBOX

    def execute(self, selection: ToolSelection) -> ToolResult:
        request_id = selection.request_id
        tool_name = selection.matched_tool_name

        if tool_name is None:
            print("    TOOL EXECUTOR: No tool selected, skipping execution.")
            return self._finish(ToolResult(
                request_id=request_id, tool_name=None, arguments={},
                success=False, result=None, error="No tool was selected.",
            ))

        tool = self.tools.get(tool_name)
        if tool is None:
            print(f"    TOOL EXECUTOR: '{tool_name}' matched but is not in the toolbox.")
            return self._finish(ToolResult(
                request_id=request_id, tool_name=tool_name, arguments={},
                success=False, result=None, error=f"Unknown tool '{tool_name}'.",
            ))

        aggregate_input = selection.aggregate_input
        current_text = aggregate_input.raw_user_input
        combined_text = (
            f"{aggregate_input.conversation_context}\n{current_text}"
            if aggregate_input.conversation_context
            else current_text
        )
        try:
            arguments = tool.extract_arguments(tool, current_text, combined_text)
            print(f"    TOOL EXECUTOR: Tool arguments: {arguments}")
        except ArgumentExtractionError as exc:
            print(f"    TOOL EXECUTOR: Argument extraction failed for '{tool_name}': {exc}")
            return self._finish(ToolResult(
                request_id=request_id, tool_name=tool_name, arguments={},
                success=False, result=None, error=str(exc),
            ))

        try:
            output = tool.func(**arguments)
        except Exception as exc:  # noqa: BLE001 - execution failures are reported, not raised
            return self._finish(ToolResult(
                request_id=request_id, tool_name=tool_name, arguments=arguments,
                success=False, result=None, error=str(exc),
            ))

        return self._finish(ToolResult(
            request_id=request_id, tool_name=tool_name, arguments=arguments,
            success=True, result=output,
        ))

    def _finish(self, result: ToolResult) -> ToolResult:
        self.logger.log(
            request_id=result.request_id,
            layer="tool_executor",
            event="tool_executed",
            data={
                "tool_name": result.tool_name,
                "arguments": result.arguments,
                "success": result.success,
                "error": result.error,
            },
        )
        return result
