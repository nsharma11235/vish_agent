"""IP-API geolocation tool.

GET http://ip-api.com/json/{query} -> JSON with latitude, longitude, and
other location data. `query` is an optional IP address; omitted, the API
geolocates the caller.
"""

from __future__ import annotations

import re
from typing import Any

import requests

from vish_agent.config import HTTP_TIMEOUT_SECONDS
from vish_agent.tools.base import ArgumentExtractionError, ToolParameter, ToolSpec, cast_param

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def ip_geolocate(query: str = "") -> dict:
    url = f"http://ip-api.com/json/{query}"
    response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def extract_arguments(tool: ToolSpec, current_text: str, combined_text: str) -> dict[str, Any]:
    """Searches combined_text (history + current turn), preferring the most
    recent IP address, so a location established earlier in the conversation
    carries forward (e.g. "look it up again")."""
    matches = list(_IPV4_RE.finditer(combined_text))
    if not matches:
        return {}  # query is optional; the API geolocates the caller when omitted

    value = cast_param(tool, "query", matches[-1].group(0))
    if not _IPV4_RE.fullmatch(value):
        raise ArgumentExtractionError(f"Argument 'query' is not a valid IPv4 address: {value!r}")
    return {"query": value}


TOOL_SPEC = ToolSpec(
    name="ip_api_geolocation",
    description=(
        "Looks up latitude, longitude, and other location data for an IP "
        "address using the IP-API geolocation service. Omit the query to "
        "look up the caller's own public IP. This does not find the coordinates "
        "for a given location. This only finds the coordinates for a given IP address."
        "Only use this tool if the user asks for the location of a specific IP address"
        "or the user asks for their current location. Do not use this for other location queries."
    ),
    parameters=(
        ToolParameter(
            name="query",
            type=str,
            required=False,
            description="An IP address to geolocate. Defaults to the caller's IP.",
        ),
    ),
    func=ip_geolocate,
    extract_arguments=extract_arguments,
)
