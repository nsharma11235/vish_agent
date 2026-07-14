import json

import pytest

from vish_agent.logging_utils import TransactionLogger
from vish_agent.pipeline import VishPipeline
from vish_agent.schemas import FinalResponse
from vish_agent.session import ConversationSession
from vish_agent.tools.registry import TOOLBOX


@pytest.fixture
def pipeline(qwen_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    return VishPipeline(llm_client=qwen_client, logger=logger), logger.log_path


def test_pipeline_run_returns_final_response(pipeline):
    vish, _ = pipeline
    result = vish.run("Tell me a fun fact about octopuses.")

    assert isinstance(result, FinalResponse)
    assert result.request_id
    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0


def test_pipeline_run_writes_one_log_entry_per_layer(pipeline):
    vish, log_path = pipeline
    result = vish.run("What's my IP address's location?")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    layers_seen = {r["layer"] for r in records}
    assert layers_seen == {"input_layer", "orchestrator", "tool_executor", "response_generator"}
    assert all(r["request_id"] == result.request_id for r in records)


def test_pipeline_accepts_custom_toolbox_size(qwen_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    single_tool = {"ip_api_geolocation": TOOLBOX["ip_api_geolocation"]}
    vish = VishPipeline(llm_client=qwen_client, logger=logger, tools=single_tool)

    vish.run("What's my IP address's location?")

    lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    orchestrator_record = next(r for r in records if r["layer"] == "orchestrator")
    assert orchestrator_record["data"]["num_available_tools"] == 1
    assert orchestrator_record["data"]["available_tools"] == ["ip_api_geolocation"]


def test_pipeline_without_session_has_no_cross_turn_memory(pipeline):
    vish, _ = pipeline
    vish.run("Can you look up the geolocation for IP address 8.8.8.8?")
    # No session passed, so the second call has no history to draw a location from.
    result = vish.run("What's the weather broadcast there?")
    assert isinstance(result, FinalResponse)


def test_pipeline_carries_location_across_turns_in_a_session(pipeline):
    vish, _ = pipeline
    session = ConversationSession()

    first = vish.run("Can you look up the geolocation for IP address 8.8.8.8?", session=session)
    assert isinstance(first, FinalResponse)
    assert len(session.turns) == 1
    assert session.turns[0].tool_name == "ip_api_geolocation"
    assert session.turns[0].tool_success is True

    second = vish.run("What's the weather broadcast there?", session=session)
    assert isinstance(second, FinalResponse)
    assert len(session.turns) == 2
    second_turn = session.turns[1]
    assert second_turn.tool_name == "nws_weather_radio"
    assert second_turn.tool_success is True, second_turn.tool_result
