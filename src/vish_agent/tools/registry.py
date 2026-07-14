"""The toolbox: every tool the Orchestrator can select from."""

from __future__ import annotations

from vish_agent.tools.base import ToolSpec
from vish_agent.tools.ip_api import TOOL_SPEC as IP_API_TOOL
from vish_agent.tools.tavily_search import TOOL_SPEC as TAVILY_SEARCH_TOOL
from vish_agent.tools.weather import TOOL_SPEC as WEATHER_TOOL

TOOLBOX: dict[str, ToolSpec] = {
    tool.name: tool for tool in (IP_API_TOOL, WEATHER_TOOL, TAVILY_SEARCH_TOOL)
}


def get_tool(name: str) -> ToolSpec | None:
    return TOOLBOX.get(name)


def get_tool_names() -> list[str]:
    return list(TOOLBOX.keys())
