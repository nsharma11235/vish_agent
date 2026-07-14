"""End-to-end pipeline wiring the four layers together.

A future REST API layer should depend only on VishPipeline.run(), not on the
individual layer classes, so the pipeline can be swapped out or extended
without restructuring the layers themselves. The toolbox is injectable so
experiments can vary the number of available tools.
"""

from __future__ import annotations

from vish_agent.layers.input_layer import InputLayer
from vish_agent.layers.orchestrator import Orchestrator
from vish_agent.layers.response_generator import ResponseGenerator
from vish_agent.layers.tool_executor import ToolExecutor
from vish_agent.llm.client import LLMClient
from vish_agent.logging_utils import TransactionLogger, default_logger
from vish_agent.schemas import FinalResponse
from vish_agent.session import VishConversationSession, Turn
from vish_agent.tools.base import ToolSpec
from vish_agent.tools.registry import TOOLBOX


class VishPipeline:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        logger: TransactionLogger = default_logger,
        tools: dict[str, ToolSpec] | None = None,
    ):
        llm_client = llm_client or LLMClient.get_shared()
        tools = tools if tools is not None else TOOLBOX

        self.input_layer = InputLayer(llm_client=llm_client, logger=logger)
        self.orchestrator = Orchestrator(llm_client=llm_client, logger=logger, tools=tools)
        self.tool_executor = ToolExecutor(logger=logger, tools=tools)
        self.response_generator = ResponseGenerator(llm_client=llm_client, logger=logger)

    def run(self, raw_user_input: str, session: VishConversationSession | None = None) -> FinalResponse:
        # Add to future versions: conversation_context = session.context_text() if session else ""
        conversation_context = ""
        aggregate_input = self.input_layer.process(raw_user_input, conversation_context=conversation_context)
        print(f"    Request type: {aggregate_input.expected_response_type}")
        tool_selection = self.orchestrator.select_tool(aggregate_input)
        tool_result = self.tool_executor.execute(tool_selection)
        print(f"    Tool selected: {tool_result.tool_name}")
        final_response = self.response_generator.generate(tool_selection, tool_result)

        if session is not None:
            session.add_turn(
                Turn(
                    user_input=raw_user_input,
                    tool_name=tool_result.tool_name,
                    tool_success=tool_result.success,
                    tool_result=tool_result.result if tool_result.success else tool_result.error,
                    response_text=final_response.text,
                )
            )

        return final_response
