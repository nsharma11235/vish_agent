"""National Weather Service weather-radio tool.

GET http://api.weather.gov/points/{latitude},{longitude}/radio -> an XML
NOAA weather radio broadcast script, which the tool transforms to plaintext.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import requests

from vish_agent.config import HTTP_TIMEOUT_SECONDS, NWS_USER_AGENT
from vish_agent.tools.base import ArgumentExtractionError, ToolParameter, ToolSpec, cast_param

_LATITUDE_KEYWORD_RE = re.compile(r"lat(?:itude)?\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_LONGITUDE_KEYWORD_RE = re.compile(r"lon(?:g|gitude)?\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_LAT_LON_PAIR_RE = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)")


def _xml_to_plaintext(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    lines = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
    return "\n".join(lines)


def nws_weather_radio(latitude: float, longitude: float) -> str:
    url = f"http://api.weather.gov/points/{latitude},{longitude}/radio"
    response = requests.get(
        url,
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/xml"},
    )
    response.raise_for_status()
    return _xml_to_plaintext(response.text)


def extract_arguments(tool: ToolSpec, current_text: str, combined_text: str) -> dict[str, Any]:
    """Searches combined_text (history + current turn), preferring the most
    recent coordinates, so a location resolved earlier in the conversation
    (e.g. by ip_api_geolocation) satisfies this tool without repeating it."""
    lat_matches = list(_LATITUDE_KEYWORD_RE.finditer(combined_text))
    lon_matches = list(_LONGITUDE_KEYWORD_RE.finditer(combined_text))
    if lat_matches and lon_matches:
        raw_lat, raw_lon = lat_matches[-1].group(1), lon_matches[-1].group(1)
    else:
        pair_matches = list(_LAT_LON_PAIR_RE.finditer(combined_text))
        if not pair_matches:
            raise ArgumentExtractionError(
                f"Could not extract required latitude/longitude for tool '{tool.name}' from user input."
            )
        raw_lat, raw_lon = pair_matches[-1].group(1), pair_matches[-1].group(2)

    latitude = cast_param(tool, "latitude", raw_lat)
    longitude = cast_param(tool, "longitude", raw_lon)
    if not -90 <= latitude <= 90:
        raise ArgumentExtractionError(f"Latitude {latitude} is out of range [-90, 90].")
    if not -180 <= longitude <= 180:
        raise ArgumentExtractionError(f"Longitude {longitude} is out of range [-180, 180].")
    return {"latitude": latitude, "longitude": longitude}


TOOL_SPEC = ToolSpec(
    name="nws_weather_radio",
    description=(
        "Fetches the current NOAA weather radio broadcast script for a "
        "location and returns it as plain text. Requires latitude and "
        "longitude."
    ),
    parameters=(
        ToolParameter(name="latitude", type=float, required=True, description="Latitude of the location."),
        ToolParameter(name="longitude", type=float, required=True, description="Longitude of the location."),
    ),
    func=nws_weather_radio,
    extract_arguments=extract_arguments,
)
