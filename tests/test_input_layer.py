import json

import pytest

from vish_agent.layers.input_layer import InputLayer
from vish_agent.logging_utils import TransactionLogger
from vish_agent.prompts import SYSTEM_PROMPT
from vish_agent.schemas import AggregateInput, ResponseType


@pytest.fixture
def logged_input_layer(llm_client, tmp_path):
    log_path = tmp_path / "transactions.jsonl"
    logger = TransactionLogger(log_path)
    layer = InputLayer(llm_client=llm_client, logger=logger)
    return layer, log_path


def test_process_returns_well_formed_aggregate_input(logged_input_layer):
    layer, _ = logged_input_layer
    result = layer.process("Hello, how are you today?")

    assert isinstance(result, AggregateInput)
    assert result.request_id
    assert result.raw_user_input == "Hello, how are you today?"
    assert result.system_prompt == SYSTEM_PROMPT
    assert SYSTEM_PROMPT in result.aggregate_text
    assert "Hello, how are you today?" in result.aggregate_text
    assert isinstance(result.expected_response_type, ResponseType)


def test_process_classifies_conversational_prompt(logged_input_layer):
    layer, _ = logged_input_layer
    result = layer.process("Tell me a fun fact about octopuses.")
    assert result.expected_response_type == ResponseType.CONVERSATIONAL


def test_process_classifies_tool_assisted_prompt(logged_input_layer):
    layer, _ = logged_input_layer
    result = layer.process("What is the weather like right now at my current location?")
    assert result.expected_response_type == ResponseType.TOOL_ASSISTED


def test_process_writes_structured_log_entry(logged_input_layer):
    layer, log_path = logged_input_layer
    result = layer.process("What is my IP address's location?")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["layer"] == "input_layer"
    assert record["event"] == "processed"
    assert record["request_id"] == result.request_id
    assert record["data"]["raw_user_input"] == "What is my IP address's location?"
    assert record["data"]["expected_response_type"] == result.expected_response_type.value


def test_each_call_gets_a_unique_request_id(logged_input_layer):
    layer, _ = logged_input_layer
    first = layer.process("Hi there.")
    second = layer.process("Hi again.")
    assert first.request_id != second.request_id


def test_custom_system_prompt_is_used(llm_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    layer = InputLayer(llm_client=llm_client, logger=logger, system_prompt="You are a pirate.")

    result = layer.process("Ahoy!")
    assert result.system_prompt == "You are a pirate."
    assert "You are a pirate." in result.aggregate_text


def test_process_includes_conversation_context_in_aggregate_text(logged_input_layer):
    layer, _ = logged_input_layer
    context = "User: What's my location?\n[ip_api_geolocation succeeded: lat=39.03, lon=-77.5]\nAssistant: You're near Ashburn, Virginia."

    result = layer.process("What's the weather there?", conversation_context=context)

    assert result.conversation_context == context
    assert context in result.aggregate_text
    assert "What's the weather there?" in result.aggregate_text


def test_process_defaults_to_empty_conversation_context(logged_input_layer):
    layer, _ = logged_input_layer
    result = layer.process("Hello")
    assert result.conversation_context == ""
