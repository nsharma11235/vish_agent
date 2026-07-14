"""Input Layer: raw user text -> AggregateInput for the Orchestrator.

Combines the raw text with the pre-written system prompt, then runs a
lightweight classification pass through Qwen2.5-1.5B-Instruct to decide
whether the request is conversational or needs a tool.
"""

from __future__ import annotations

from vish_agent.config import CLASSIFIER_TEMPERATURE
from vish_agent.llm.qwen_client import QwenClient
from vish_agent.logging_utils import TransactionLogger, default_logger
from vish_agent.prompts import RESPONSE_TYPE_CLASSIFIER_PROMPT, SYSTEM_PROMPT
from vish_agent.schemas import AggregateInput, ResponseType, new_request_id


class InputLayer:
    def __init__(
        self,
        llm_client: QwenClient | None = None,
        logger: TransactionLogger = default_logger,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.llm_client = llm_client or QwenClient.get_shared()
        self.logger = logger
        self.system_prompt = system_prompt

    def process(self, raw_user_input: str, conversation_context: str = "") -> AggregateInput:
        request_id = new_request_id()
        if conversation_context:
            aggregate_text = f"{self.system_prompt}\n\n{conversation_context}\nUser: {raw_user_input}"
        else:
            aggregate_text = f"{self.system_prompt}\n\nUser: {raw_user_input}"

        classification_prompt = RESPONSE_TYPE_CLASSIFIER_PROMPT.format(user_input=raw_user_input)
        raw_classification = self.llm_client.chat(
            [{"role": "user", "content": classification_prompt}],
            temperature=CLASSIFIER_TEMPERATURE,
            max_new_tokens=8,
        )
        expected_response_type = self._resolve_response_type(raw_classification)
        print(f"    INPUT LAYER: Expected response type: {raw_classification}")

        aggregate_input = AggregateInput(
            request_id=request_id,
            raw_user_input=raw_user_input,
            system_prompt=self.system_prompt,
            aggregate_text=aggregate_text,
            expected_response_type=expected_response_type,
            conversation_context=conversation_context,
        )

        self.logger.log(
            request_id=request_id,
            layer="input_layer",
            event="processed",
            data={
                "raw_user_input": raw_user_input,
                "system_prompt": self.system_prompt,
                "aggregate_text": aggregate_text,
                "conversation_context": conversation_context,
                "raw_classification_output": raw_classification,
                "expected_response_type": expected_response_type.value,
            },
        )
        return aggregate_input

    @staticmethod
    def _resolve_response_type(raw_output: str) -> ResponseType:
        normalized = raw_output.strip().lower()
        if ResponseType.TOOL_ASSISTED.value in normalized:
            return ResponseType.TOOL_ASSISTED
        if ResponseType.CONVERSATIONAL.value in normalized:
            return ResponseType.CONVERSATIONAL
        # Ambiguous/unparseable model output: default to tool_assisted so the
        # pipeline still attempts to ground its answer rather than silently
        # skipping tool use.
        return ResponseType.TOOL_ASSISTED
