"""Tavily web search tool.

Uses the tavily-python client library. Requires TAVILY_API_KEY (see config).
"""

from __future__ import annotations

import re
from typing import Any

from vish_agent.config import TAVILY_API_KEY
from vish_agent.tools.base import ArgumentExtractionError, ToolParameter, ToolSpec, cast_param

_MAX_RESULTS_RE = re.compile(r"\b(\d+)\s*results?\b", re.IGNORECASE)

_client = None


def _get_client():
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set; add it to your .env file.")
        from tavily import TavilyClient

        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def tavily_web_search(query: str, max_results: int | None = None) -> dict:
    client = _get_client()
    kwargs = {}
    if max_results is not None:
        kwargs["max_results"] = max_results
    return client.search(query, **kwargs)


def extract_arguments(tool: ToolSpec, current_text: str, combined_text: str) -> dict[str, Any]:
    """Only looks at current_text - a search query shouldn't absorb history
    the way a location can, or an old search would bleed into a new one."""
    query = current_text.strip()
    if not query:
        raise ArgumentExtractionError(f"Could not extract required 'query' for tool '{tool.name}' from user input.")
    arguments: dict[str, Any] = {"query": cast_param(tool, "query", query)}

    max_results_match = _MAX_RESULTS_RE.search(current_text)
    if max_results_match:
        arguments["max_results"] = cast_param(tool, "max_results", max_results_match.group(1))
    return arguments


TOOL_SPEC = ToolSpec(
    name="tavily_web_search",
    description="Searches the web for a plain text query and returns matching results. "
     "Use this for informational queries including geolocation questions.",
    parameters=(
        ToolParameter(name="query", type=str, required=True, description="The search query."),
        ToolParameter(name="max_results", type=int, required=False, description="Maximum number of results to return."),
    ),
    func=tavily_web_search,
    extract_arguments=extract_arguments,
)
