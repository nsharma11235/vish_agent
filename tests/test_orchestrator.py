import json

import pytest

from vish_agent.logging_utils import TransactionLogger
from vish_agent.layers.orchestrator import Orchestrator
from vish_agent.schemas import AggregateInput, ResponseType, ToolSelection, new_request_id
from vish_agent.tools.base import ToolParameter, ToolSpec

_NO_EXTRACTION = lambda tool, current_text, combined_text: {}  # noqa: E731 - Orchestrator never calls this

TOOL_A = ToolSpec(
    name="tavily_web_search",
    description="Searches the web for a plain text query.",
    parameters=(ToolParameter(name="query", type=str, required=True),),
    func=lambda **kwargs: None,
    extract_arguments=_NO_EXTRACTION,
)
TOOL_B = ToolSpec(
    name="ip_api_geolocation",
    description="Looks up latitude/longitude for an IP address.",
    parameters=(ToolParameter(name="query", type=str, required=False),),
    func=lambda **kwargs: None,
    extract_arguments=_NO_EXTRACTION,
)
TOOL_C = ToolSpec(
    name="nws_weather_radio",
    description="Fetches a NOAA weather radio broadcast script.",
    parameters=(
        ToolParameter(name="latitude", type=float, required=True),
        ToolParameter(name="longitude", type=float, required=True),
    ),
    func=lambda **kwargs: None,
    extract_arguments=_NO_EXTRACTION,
)
SMALL_TOOLBOX = {t.name: t for t in (TOOL_A, TOOL_B, TOOL_C)}


def make_aggregate_input(raw_user_input: str) -> AggregateInput:
    return AggregateInput(
        request_id=new_request_id(),
        raw_user_input=raw_user_input,
        system_prompt="You are Vish.",
        aggregate_text=f"You are Vish.\n\nUser: {raw_user_input}",
        expected_response_type=ResponseType.TOOL_ASSISTED,
    )


@pytest.fixture
def orchestrator(qwen_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    return Orchestrator(llm_client=qwen_client, logger=logger, tools=SMALL_TOOLBOX), logger.log_path


# --- Pure fuzzy-matching unit tests (no model calls) ---------------------


def test_match_tool_name_exact_match(orchestrator):
    orch, _ = orchestrator
    matched, score = orch._match_tool_name("tavily_web_search", list(SMALL_TOOLBOX))
    assert matched == "tavily_web_search"
    assert score == 100


def test_match_tool_name_handles_typo(orchestrator):
    orch, _ = orchestrator
    matched, score = orch._match_tool_name("tavly_web_serach", list(SMALL_TOOLBOX))
    assert matched == "tavily_web_search"
    assert score >= orch.fuzzy_match_threshold


def test_match_tool_name_none_output(orchestrator):
    orch, _ = orchestrator
    matched, score = orch._match_tool_name("NONE", list(SMALL_TOOLBOX))
    assert matched is None
    assert score == 0


def test_match_tool_name_unrelated_output_below_threshold(orchestrator):
    orch, _ = orchestrator
    matched, score = orch._match_tool_name("banana", list(SMALL_TOOLBOX))
    assert matched is None
    assert score < orch.fuzzy_match_threshold


def test_match_tool_name_empty_toolbox(orchestrator):
    orch, _ = orchestrator
    matched, score = orch._match_tool_name("tavily_web_search", [])
    assert matched is None
    assert score == 0


# --- Integration tests through the real model -----------------------------


def test_select_tool_returns_tool_selection(orchestrator):
    orch, _ = orchestrator
    aggregate_input = make_aggregate_input("Search the web for the latest news on Mars rovers.")
    selection = orch.select_tool(aggregate_input)

    assert isinstance(selection, ToolSelection)
    assert selection.request_id == aggregate_input.request_id
    assert selection.aggregate_input is aggregate_input
    assert selection.matched_tool_name is None or selection.matched_tool_name in SMALL_TOOLBOX


def test_select_tool_prefers_web_search_for_search_prompt(orchestrator):
    orch, _ = orchestrator
    aggregate_input = make_aggregate_input(
        "Please search the web for today's top technology news headlines."
    )
    selection = orch.select_tool(aggregate_input)
    assert selection.matched_tool_name == "tavily_web_search"


def test_select_tool_writes_log_entry(orchestrator):
    orch, log_path = orchestrator
    aggregate_input = make_aggregate_input("What's my IP address's location?")
    selection = orch.select_tool(aggregate_input)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["layer"] == "orchestrator"
    assert record["event"] == "tool_selected"
    assert record["request_id"] == selection.request_id
    assert record["data"]["num_available_tools"] == len(SMALL_TOOLBOX)
    assert set(record["data"]["available_tools"]) == set(SMALL_TOOLBOX)


def test_orchestrator_respects_injected_toolbox_size(qwen_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    single_tool_orch = Orchestrator(llm_client=qwen_client, logger=logger, tools={"tavily_web_search": TOOL_A})

    assert single_tool_orch._format_tool_descriptions() == f"- tavily_web_search: {TOOL_A.description}"

    aggregate_input = make_aggregate_input("Search for the latest AI research papers.")
    selection = single_tool_orch.select_tool(aggregate_input)
    assert selection.matched_tool_name in (None, "tavily_web_search")
