"""Runs the standardized labeled prompt set (eval_prompts.json) through the
pipeline at several toolbox sizes and prints per-configuration accuracy.
Full per-prompt results are logged to logs/evaluation_results.jsonl.
"""

from __future__ import annotations

from vish_agent.evaluation.dataset import load_labeled_prompts
from vish_agent.evaluation.harness import EvaluationHarness
from vish_agent.llm.client import LLMClient
from vish_agent.pipeline import VishPipeline
from vish_agent.tools.registry import TOOLBOX

TOOLBOX_CONFIGURATIONS = [
    {"ip_api_geolocation": TOOLBOX["ip_api_geolocation"]},
    {name: TOOLBOX[name] for name in ("ip_api_geolocation", "nws_weather_radio")},
    dict(TOOLBOX),
]


def main() -> None:
    prompts = load_labeled_prompts()
    llm_client = LLMClient.get_shared()

    for tools in TOOLBOX_CONFIGURATIONS:
        pipeline = VishPipeline(llm_client=llm_client, tools=tools)
        harness = EvaluationHarness(pipeline)
        results = harness.run(prompts)

        graded_tool = [r for r in results if r.tool_correct is not None]
        graded_args = [r for r in results if r.arguments_correct is not None]
        tool_accuracy = sum(r.tool_correct for r in graded_tool) / len(graded_tool)
        arg_accuracy = (
            sum(r.arguments_correct for r in graded_args) / len(graded_args) if graded_args else float("nan")
        )

        print(f"Toolbox size {len(tools)} ({sorted(tools)}):")
        print(f"  tool selection accuracy: {tool_accuracy:.1%} ({len(graded_tool)} prompts)")
        print(f"  argument accuracy:       {arg_accuracy:.1%} ({len(graded_args)} prompts)")
        print()


if __name__ == "__main__":
    main()
