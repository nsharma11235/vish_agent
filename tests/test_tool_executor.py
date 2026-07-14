"""ToolExecutor tests using the real per-tool extraction logic with fake
(offline, no-network) execution functions.

Each tool's ToolSpec.extract_arguments is imported straight from its module
(ip_api.py, weather.py, tavily_search.py) so these tests exercise the actual
production extraction rules; only ToolSpec.func (the part that would hit a
real API) is swapped out.
"""

import json

import pytest

from vish_agent.layers.tool_executor import ToolExecutor
from vish_agent.logging_utils import TransactionLogger
from vish_agent.schemas import AggregateInput, ResponseType, ToolSelection, new_request_id
from vish_agent.tools import ip_api, tavily_search, weather
from vish_agent.tools.base import ToolParameter, ToolSpec


def make_tool(name, parameters, func, extract_arguments):
    return ToolSpec(name=name, description=f"test {name}", parameters=parameters, func=func, extract_arguments=extract_arguments)


IP_TOOL = make_tool(
    "ip_api_geolocation",
    (ToolParameter(name="query", type=str, required=False),),
    lambda **kwargs: {"query": kwargs.get("query", ""), "lat": 1.0, "lon": 2.0},
    ip_api.extract_arguments,
)
WEATHER_TOOL = make_tool(
    "nws_weather_radio",
    (
        ToolParameter(name="latitude", type=float, required=True),
        ToolParameter(name="longitude", type=float, required=True),
    ),
    lambda **kwargs: f"forecast for {kwargs['latitude']},{kwargs['longitude']}",
    weather.extract_arguments,
)
SEARCH_TOOL = make_tool(
    "tavily_web_search",
    (
        ToolParameter(name="query", type=str, required=True),
        ToolParameter(name="max_results", type=int, required=False),
    ),
    lambda **kwargs: {"query": kwargs["query"], "max_results": kwargs.get("max_results")},
    tavily_search.extract_arguments,
)
TEST_TOOLBOX = {t.name: t for t in (IP_TOOL, WEATHER_TOOL, SEARCH_TOOL)}


def make_selection(raw_user_input: str, matched_tool_name, conversation_context: str = ""):
    aggregate_input = AggregateInput(
        request_id=new_request_id(),
        raw_user_input=raw_user_input,
        system_prompt="You are Vish.",
        aggregate_text=f"You are Vish.\n\nUser: {raw_user_input}",
        expected_response_type=ResponseType.TOOL_ASSISTED,
        conversation_context=conversation_context,
    )
    return ToolSelection(
        request_id=aggregate_input.request_id,
        aggregate_input=aggregate_input,
        raw_model_output=matched_tool_name or "NONE",
        matched_tool_name=matched_tool_name,
        match_score=100 if matched_tool_name else 0,
    )


@pytest.fixture
def executor(tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    return ToolExecutor(logger=logger, tools=TEST_TOOLBOX), logger.log_path


def test_execute_returns_failure_when_no_tool_matched(executor):
    ex, _ = executor
    result = ex.execute(make_selection("hello", None))
    assert result.success is False
    assert result.tool_name is None
    assert "No tool was selected" in result.error


def test_execute_returns_failure_for_unknown_tool(executor):
    ex, _ = executor
    result = ex.execute(make_selection("hello", "not_a_real_tool"))
    assert result.success is False
    assert "Unknown tool" in result.error


def test_ip_api_extraction_with_ip_present(executor):
    ex, _ = executor
    result = ex.execute(make_selection("What's the location of 8.8.8.8?", "ip_api_geolocation"))
    assert result.success is True
    assert result.arguments == {"query": "8.8.8.8"}


def test_ip_api_extraction_without_ip_omits_optional_query(executor):
    ex, _ = executor
    result = ex.execute(make_selection("What's my location?", "ip_api_geolocation"))
    assert result.success is True
    assert result.arguments == {}


def test_weather_extraction_with_keyword_form(executor):
    ex, _ = executor
    result = ex.execute(make_selection("weather at latitude 40.7128 longitude -74.0060", "nws_weather_radio"))
    assert result.success is True
    assert result.arguments == {"latitude": 40.7128, "longitude": -74.006}


def test_weather_extraction_with_pair_form(executor):
    ex, _ = executor
    result = ex.execute(make_selection("weather near 40.7128, -74.0060", "nws_weather_radio"))
    assert result.success is True
    assert result.arguments == {"latitude": 40.7128, "longitude": -74.006}


def test_weather_extraction_missing_coordinates_fails(executor):
    ex, _ = executor
    result = ex.execute(make_selection("what's the weather like", "nws_weather_radio"))
    assert result.success is False
    assert "latitude/longitude" in result.error


def test_weather_extraction_rejects_out_of_range_latitude(executor):
    ex, _ = executor
    result = ex.execute(make_selection("weather at latitude 999 longitude -74.0", "nws_weather_radio"))
    assert result.success is False
    assert "out of range" in result.error


def test_tavily_extraction_uses_full_message_as_query(executor):
    ex, _ = executor
    result = ex.execute(make_selection("search the web for octopus facts", "tavily_web_search"))
    assert result.success is True
    assert result.arguments["query"] == "search the web for octopus facts"
    assert "max_results" not in result.arguments


def test_tavily_extraction_parses_max_results(executor):
    ex, _ = executor
    result = ex.execute(make_selection("search the web for octopus facts, give me 5 results", "tavily_web_search"))
    assert result.success is True
    assert result.arguments["max_results"] == 5


def test_execute_captures_tool_func_exceptions(tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    failing_tool = make_tool(
        "ip_api_geolocation",
        (ToolParameter(name="query", type=str, required=False),),
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        ip_api.extract_arguments,
    )
    ex = ToolExecutor(logger=logger, tools={"ip_api_geolocation": failing_tool})
    result = ex.execute(make_selection("look up 8.8.8.8", "ip_api_geolocation"))
    assert result.success is False
    assert "boom" in result.error


def test_weather_extraction_falls_back_to_conversation_context(executor):
    ex, _ = executor
    context = "User: What's my location?\n[ip_api_geolocation succeeded: lat=39.03, lon=-77.5]\nAssistant: You're near Ashburn, Virginia."
    selection = make_selection("What's the weather broadcast there?", "nws_weather_radio", conversation_context=context)

    result = ex.execute(selection)
    assert result.success is True
    assert result.arguments == {"latitude": 39.03, "longitude": -77.5}


def test_weather_extraction_prefers_most_recent_location(executor):
    ex, _ = executor
    context = (
        "User: What's the weather at 40.7128, -74.0060?\n"
        "[nws_weather_radio succeeded: forecast for 40.7128,-74.006]\n"
        "Assistant: Here's the New York forecast.\n"
        "User: What about 34.0522, -118.2437?\n"
        "[nws_weather_radio succeeded: forecast for 34.0522,-118.2437]\n"
        "Assistant: Here's the Los Angeles forecast."
    )
    selection = make_selection("And again?", "nws_weather_radio", conversation_context=context)

    result = ex.execute(selection)
    assert result.success is True
    assert result.arguments == {"latitude": 34.0522, "longitude": -118.2437}


def test_ip_api_extraction_falls_back_to_conversation_context(executor):
    ex, _ = executor
    context = "User: Look up 8.8.8.8\n[ip_api_geolocation succeeded: query=8.8.8.8, lat=39.03, lon=-77.5]\nAssistant: That's in Ashburn, Virginia."
    selection = make_selection("Look it up again", "ip_api_geolocation", conversation_context=context)

    result = ex.execute(selection)
    assert result.success is True
    assert result.arguments == {"query": "8.8.8.8"}


def test_tavily_extraction_ignores_conversation_context(executor):
    ex, _ = executor
    context = "User: Search for octopus facts\n[tavily_web_search succeeded: query=octopus facts]\nAssistant: Octopuses have three hearts."
    selection = make_selection("Now search for penguin facts", "tavily_web_search", conversation_context=context)

    result = ex.execute(selection)
    assert result.success is True
    assert result.arguments["query"] == "Now search for penguin facts"


def test_execute_writes_structured_log_entry(executor):
    ex, log_path = executor
    result = ex.execute(make_selection("What's the location of 8.8.8.8?", "ip_api_geolocation"))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["layer"] == "tool_executor"
    assert record["event"] == "tool_executed"
    assert record["request_id"] == result.request_id
    assert record["data"]["tool_name"] == "ip_api_geolocation"
    assert record["data"]["success"] is True
