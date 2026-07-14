import json

import pytest

from vish_agent.logging_utils import TransactionLogger
from vish_agent.prompts import SYSTEM_PROMPT
from vish_agent.layers.response_generator import ResponseGenerator
from vish_agent.schemas import (
    AggregateInput,
    FinalResponse,
    ResponseType,
    ToolResult,
    ToolSelection,
    new_request_id,
)


def make_aggregate_input(raw_user_input: str, expected_response_type: ResponseType) -> AggregateInput:
    return AggregateInput(
        request_id=new_request_id(),
        raw_user_input=raw_user_input,
        system_prompt=SYSTEM_PROMPT,
        aggregate_text=f"{SYSTEM_PROMPT}\n\nUser: {raw_user_input}",
        expected_response_type=expected_response_type,
    )


def make_tool_selection(aggregate_input: AggregateInput, matched_tool_name, raw_model_output="", match_score=0) -> ToolSelection:
    return ToolSelection(
        request_id=aggregate_input.request_id,
        aggregate_input=aggregate_input,
        raw_model_output=raw_model_output or (matched_tool_name or "NONE"),
        matched_tool_name=matched_tool_name,
        match_score=match_score,
    )


def make_tool_result(request_id, tool_name, arguments=None, success=True, result=None, error=None) -> ToolResult:
    return ToolResult(
        request_id=request_id,
        tool_name=tool_name,
        arguments=arguments or {},
        success=success,
        result=result,
        error=error,
    )


@pytest.fixture
def response_generator(qwen_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    return ResponseGenerator(llm_client=qwen_client, logger=logger), logger.log_path


def test_generate_returns_final_response_with_text(response_generator):
    generator, _ = response_generator
    aggregate_input = make_aggregate_input("Tell me a fun fact about octopuses.", ResponseType.CONVERSATIONAL)
    selection = make_tool_selection(aggregate_input, matched_tool_name=None)
    tool_result = make_tool_result(aggregate_input.request_id, tool_name=None, success=False, error="No tool was selected.")

    result = generator.generate(selection, tool_result)

    assert isinstance(result, FinalResponse)
    assert result.request_id == aggregate_input.request_id
    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0


def test_generate_incorporates_successful_tool_result(response_generator):
    generator, _ = response_generator
    aggregate_input = make_aggregate_input(
        "What's my IP address's location?", ResponseType.TOOL_ASSISTED
    )
    selection = make_tool_selection(aggregate_input, matched_tool_name="ip_api_geolocation", match_score=100)
    tool_result = make_tool_result(
        aggregate_input.request_id,
        tool_name="ip_api_geolocation",
        arguments={"query": "8.8.8.8"},
        success=True,
        result={"status": "success", "city": "Ashburn", "regionName": "Virginia", "country": "United States"},
    )

    result = generator.generate(selection, tool_result)
    assert len(result.text.strip()) > 0
    assert "ashburn" in result.text.lower() or "virginia" in result.text.lower()


def test_generate_handles_tool_failure_gracefully(response_generator):
    generator, _ = response_generator
    aggregate_input = make_aggregate_input(
        "What's the weather like right now?", ResponseType.TOOL_ASSISTED
    )
    selection = make_tool_selection(aggregate_input, matched_tool_name="nws_weather_radio", match_score=90)
    tool_result = make_tool_result(
        aggregate_input.request_id,
        tool_name="nws_weather_radio",
        success=False,
        error="Could not extract required latitude/longitude for tool 'nws_weather_radio' from user input.",
    )

    result = generator.generate(selection, tool_result)
    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0


def test_generate_writes_full_transaction_log_entry(response_generator):
    generator, log_path = response_generator
    aggregate_input = make_aggregate_input("Search the web for octopus facts.", ResponseType.TOOL_ASSISTED)
    selection = make_tool_selection(aggregate_input, matched_tool_name="tavily_web_search", match_score=100)
    tool_result = make_tool_result(
        aggregate_input.request_id,
        tool_name="tavily_web_search",
        arguments={"query": "octopus facts"},
        success=True,
        result={"results": [{"title": "Octopus Facts", "content": "Octopuses have three hearts."}]},
    )

    result = generator.generate(selection, tool_result)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["layer"] == "response_generator"
    assert record["event"] == "full_transaction"
    assert record["request_id"] == aggregate_input.request_id
    assert record["data"]["matched_tool_name"] == "tavily_web_search"
    assert record["data"]["tool_success"] is True
    assert record["data"]["final_response_text"] == result.text
