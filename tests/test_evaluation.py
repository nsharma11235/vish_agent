import json

import pytest

from vish_agent.evaluation.dataset import LabeledPrompt, load_labeled_prompts
from vish_agent.evaluation.harness import EvaluationHarness
from vish_agent.logging_utils import TransactionLogger
from vish_agent.pipeline import VishPipeline
from vish_agent.tools.registry import TOOLBOX


def test_load_labeled_prompts_parses_starter_dataset():
    prompts = load_labeled_prompts()

    assert len(prompts) >= 10
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids)), "prompt ids must be unique"
    for prompt in prompts:
        assert prompt.text.strip()
        assert isinstance(prompt.expected_tools, tuple)


# --- Pure grading logic (no model, no network) -----------------------------


def test_grade_tool_conversational_correct_when_no_tool_selected():
    prompt = LabeledPrompt(id="x", text="hi", expected_tools=(), expected_arguments=None)
    assert EvaluationHarness._grade_tool(prompt, None) is True


def test_grade_tool_conversational_incorrect_when_tool_selected():
    prompt = LabeledPrompt(id="x", text="hi", expected_tools=(), expected_arguments=None)
    assert EvaluationHarness._grade_tool(prompt, "tavily_web_search") is False


def test_grade_tool_matches_single_expected_tool():
    prompt = LabeledPrompt(id="x", text="hi", expected_tools=("tavily_web_search",), expected_arguments=None)
    assert EvaluationHarness._grade_tool(prompt, "tavily_web_search") is True
    assert EvaluationHarness._grade_tool(prompt, "ip_api_geolocation") is False
    assert EvaluationHarness._grade_tool(prompt, None) is False


def test_grade_tool_accepts_any_of_multiple_expected_tools():
    prompt = LabeledPrompt(
        id="x", text="hi", expected_tools=("ip_api_geolocation", "nws_weather_radio"), expected_arguments=None
    )
    assert EvaluationHarness._grade_tool(prompt, "ip_api_geolocation") is True
    assert EvaluationHarness._grade_tool(prompt, "nws_weather_radio") is True
    assert EvaluationHarness._grade_tool(prompt, "tavily_web_search") is False


def test_grade_arguments_none_when_not_checkable():
    prompt = LabeledPrompt(id="x", text="hi", expected_tools=(), expected_arguments=None)
    assert EvaluationHarness._grade_arguments(prompt, {"query": "anything"}) is None


def test_grade_arguments_exact_match():
    prompt = LabeledPrompt(
        id="x", text="hi", expected_tools=("ip_api_geolocation",), expected_arguments={"query": "8.8.8.8"}
    )
    assert EvaluationHarness._grade_arguments(prompt, {"query": "8.8.8.8"}) is True


def test_grade_arguments_mismatched_value():
    prompt = LabeledPrompt(
        id="x", text="hi", expected_tools=("ip_api_geolocation",), expected_arguments={"query": "8.8.8.8"}
    )
    assert EvaluationHarness._grade_arguments(prompt, {"query": "1.1.1.1"}) is False


def test_grade_arguments_mismatched_keys():
    prompt = LabeledPrompt(
        id="x",
        text="hi",
        expected_tools=("nws_weather_radio",),
        expected_arguments={"latitude": 40.7128, "longitude": -74.006},
    )
    assert EvaluationHarness._grade_arguments(prompt, {"latitude": 40.7128}) is False


def test_grade_arguments_float_tolerance():
    prompt = LabeledPrompt(
        id="x",
        text="hi",
        expected_tools=("nws_weather_radio",),
        expected_arguments={"latitude": 40.7128, "longitude": -74.006},
    )
    assert EvaluationHarness._grade_arguments(prompt, {"latitude": 40.71280001, "longitude": -74.006}) is True


# --- End-to-end harness run (real model, small prompt subset) --------------


@pytest.fixture
def harness_and_results_path(llm_client, tmp_path):
    logger = TransactionLogger(tmp_path / "transactions.jsonl")
    results_path = tmp_path / "evaluation_results.jsonl"
    pipeline = VishPipeline(llm_client=llm_client, logger=logger, tools=TOOLBOX)
    return EvaluationHarness(pipeline, results_path=results_path), results_path


def test_harness_run_grades_and_logs_results(harness_and_results_path):
    harness, results_path = harness_and_results_path
    prompts = [
        LabeledPrompt(
            id="search_test",
            text="Search the web for the latest developments in fusion energy.",
            expected_tools=("tavily_web_search",),
            expected_arguments={"query": "Search the web for the latest developments in fusion energy."},
        ),
        LabeledPrompt(
            id="conversational_test",
            text="Tell me a joke.",
            expected_tools=(),
            expected_arguments=None,
        ),
    ]

    results = harness.run(prompts)

    assert len(results) == 2
    assert results[0].prompt_id == "search_test"
    assert results[0].num_available_tools == len(TOOLBOX)
    assert results[0].tool_correct in (True, False)
    assert results[1].prompt_id == "conversational_test"
    assert results[1].arguments_correct is None

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["prompt_id"] for r in records} == {"search_test", "conversational_test"}
