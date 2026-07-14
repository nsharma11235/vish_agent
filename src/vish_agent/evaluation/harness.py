"""Evaluation harness: runs the labeled prompt set through a pipeline and
grades tool selection / argument correctness by exact match against labels
(reqs 10-13).

Final response correctness isn't exact-matchable against free text, so it's
left as None (for manual review) unless a judge_fn is supplied.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from vish_agent.config import LOG_DIR
from vish_agent.evaluation.dataset import LabeledPrompt
from vish_agent.pipeline import VishPipeline

FLOAT_TOLERANCE = 1e-6

JudgeFn = Callable[[LabeledPrompt, str], Optional[bool]]


@dataclass
class EvaluationResult:
    prompt_id: str
    raw_user_input: str
    num_available_tools: int
    available_tools: list[str]
    expected_tools: list[str]
    actual_tool: Optional[str]
    tool_correct: bool
    expected_arguments: Optional[dict[str, Any]]
    actual_arguments: dict[str, Any]
    arguments_correct: Optional[bool]
    tool_execution_success: bool
    tool_execution_error: Optional[str]
    final_response_text: str
    response_correct: Optional[bool]


class EvaluationHarness:
    def __init__(
        self,
        pipeline: VishPipeline,
        results_path: Path | str | None = None,
        judge_fn: JudgeFn | None = None,
    ):
        self.pipeline = pipeline
        self.results_path = Path(results_path) if results_path else LOG_DIR / "evaluation_results.jsonl"
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.judge_fn = judge_fn

    def run(self, prompts: list[LabeledPrompt]) -> list[EvaluationResult]:
        available_tools = sorted(self.pipeline.orchestrator.tools.keys())
        results = []
        for prompt in prompts:
            aggregate_input = self.pipeline.input_layer.process(prompt.text)
            tool_selection = self.pipeline.orchestrator.select_tool(aggregate_input)
            tool_result = self.pipeline.tool_executor.execute(tool_selection)
            final_response = self.pipeline.response_generator.generate(tool_selection, tool_result)

            response_correct = self.judge_fn(prompt, final_response.text) if self.judge_fn else None

            result = EvaluationResult(
                prompt_id=prompt.id,
                raw_user_input=prompt.text,
                num_available_tools=len(available_tools),
                available_tools=available_tools,
                expected_tools=list(prompt.expected_tools),
                actual_tool=tool_selection.matched_tool_name,
                tool_correct=self._grade_tool(prompt, tool_selection.matched_tool_name),
                expected_arguments=prompt.expected_arguments,
                actual_arguments=tool_result.arguments,
                arguments_correct=self._grade_arguments(prompt, tool_result.arguments),
                tool_execution_success=tool_result.success,
                tool_execution_error=tool_result.error,
                final_response_text=final_response.text,
                response_correct=response_correct,
            )
            self._write(result)
            results.append(result)
        return results

    @staticmethod
    def _grade_tool(prompt: LabeledPrompt, actual_tool: Optional[str]) -> bool:
        if not prompt.expected_tools:
            return actual_tool is None
        return actual_tool in prompt.expected_tools

    @staticmethod
    def _grade_arguments(prompt: LabeledPrompt, actual_arguments: dict[str, Any]) -> Optional[bool]:
        if prompt.expected_arguments is None:
            return None
        if set(prompt.expected_arguments) != set(actual_arguments):
            return False
        for key, expected_value in prompt.expected_arguments.items():
            actual_value = actual_arguments.get(key)
            if isinstance(expected_value, float) and isinstance(actual_value, (int, float)):
                if not math.isclose(expected_value, actual_value, abs_tol=FLOAT_TOLERANCE):
                    return False
            elif actual_value != expected_value:
                return False
        return True

    def _write(self, result: EvaluationResult) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **asdict(result)}
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
