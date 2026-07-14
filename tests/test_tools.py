"""Real-network tests for the tool implementations themselves.

IP-API and NWS require no API key and are hit directly. Tavily is skipped
unless TAVILY_API_KEY is configured.
"""

import pytest

from vish_agent.config import TAVILY_API_KEY
from vish_agent.tools.ip_api import ip_geolocate
from vish_agent.tools.tavily_search import tavily_web_search
from vish_agent.tools.weather import nws_weather_radio


def test_ip_geolocate_self_lookup_returns_expected_fields():
    result = ip_geolocate()
    assert result["status"] == "success"
    assert "lat" in result
    assert "lon" in result


def test_ip_geolocate_specific_ip_returns_matching_query():
    result = ip_geolocate("8.8.8.8")
    assert result["status"] == "success"
    assert result["query"] == "8.8.8.8"
    assert "lat" in result and "lon" in result


def test_nws_weather_radio_returns_plaintext_for_nyc():
    text = nws_weather_radio(40.7128, -74.0060)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    assert "<" not in text
    assert ">" not in text


@pytest.mark.skipif(not TAVILY_API_KEY, reason="TAVILY_API_KEY not configured")
def test_tavily_web_search_returns_results():
    result = tavily_web_search("what is the capital of France", max_results=3)
    assert "results" in result
    assert len(result["results"]) > 0
