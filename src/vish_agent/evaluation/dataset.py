"""The standardized labeled prompt set (reqs 10-13).

Each entry pairs a user prompt with the tool(s) a correct Orchestrator run
should select and, where checkable, the arguments a correct Tool Executor
run should extract. The same set is meant to be rerun against different
toolbox configurations so tool-selection accuracy can be tracked as the
number of available tools changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from vish_agent.config import EVAL_PROMPTS_PATH


@dataclass(frozen=True)
class LabeledPrompt:
    id: str
    text: str
    # Empty tuple means no tool should be selected (conversational). More than
    # one entry means multiple tools are considered acceptable (ambiguous cases).
    expected_tools: tuple[str, ...]
    # None means "don't grade arguments" for this prompt (e.g. extraction is
    # expected to fail, or no tool is expected).
    expected_arguments: Optional[dict[str, Any]]
    notes: str = ""


def load_labeled_prompts(path: Path | str | None = None) -> list[LabeledPrompt]:
    path = Path(path) if path else EVAL_PROMPTS_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        LabeledPrompt(
            id=item["id"],
            text=item["text"],
            expected_tools=tuple(item.get("expected_tools", [])),
            expected_arguments=item.get("expected_arguments"),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]
