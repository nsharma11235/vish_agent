from vish_agent.session import VishConversationSession, Turn


def test_context_text_empty_for_fresh_session():
    session = VishConversationSession()
    assert session.context_text() == ""


def test_context_text_formats_dict_tool_result_as_key_value_pairs():
    session = VishConversationSession()
    session.add_turn(
        Turn(
            user_input="Can you look up the geolocation for IP address 8.8.8.8?",
            tool_name="ip_api_geolocation",
            tool_success=True,
            tool_result={"status": "success", "lat": 39.03, "lon": -77.5, "city": "Ashburn"},
            response_text="You're near Ashburn, Virginia.",
        )
    )

    text = session.context_text()
    assert "User: Can you look up the geolocation for IP address 8.8.8.8?" in text
    assert "ip_api_geolocation succeeded" in text
    assert "lat=39.03" in text
    assert "lon=-77.5" in text
    assert "Assistant: You're near Ashburn, Virginia." in text


def test_context_text_formats_conversational_turn_without_tool_bracket():
    session = VishConversationSession()
    session.add_turn(
        Turn(
            user_input="Tell me a joke.",
            tool_name=None,
            tool_success=False,
            tool_result=None,
            response_text="Why did the chicken cross the road?",
        )
    )

    text = session.context_text()
    assert "User: Tell me a joke." in text
    assert "Assistant: Why did the chicken cross the road?" in text
    assert "[" not in text


def test_context_text_marks_failed_tool_calls():
    session = VishConversationSession()
    session.add_turn(
        Turn(
            user_input="What's the weather like?",
            tool_name="nws_weather_radio",
            tool_success=False,
            tool_result="Could not extract required latitude/longitude.",
            response_text="I couldn't find your coordinates.",
        )
    )

    text = session.context_text()
    assert "nws_weather_radio failed" in text


def test_context_text_truncates_long_tool_results():
    session = VishConversationSession()
    long_broadcast = "Saturday: a chance of showers. " * 200  # >> MAX_TOOL_RESULT_CONTEXT_CHARS
    session.add_turn(
        Turn(
            user_input="What's the weather broadcast here?",
            tool_name="nws_weather_radio",
            tool_success=True,
            tool_result=long_broadcast,
            response_text="Here's the forecast.",
        )
    )

    text = session.context_text()
    assert long_broadcast not in text
    assert "[truncated," in text
    # A second, unrelated turn's context shouldn't be dominated by the first
    # turn's giant result once truncated.
    assert len(text) < len(long_broadcast)


def test_context_text_does_not_truncate_short_dict_results():
    session = VishConversationSession()
    session.add_turn(
        Turn(
            user_input="Look up 8.8.8.8",
            tool_name="ip_api_geolocation",
            tool_success=True,
            tool_result={
                "status": "success",
                "country": "United States",
                "countryCode": "US",
                "region": "VA",
                "regionName": "Virginia",
                "city": "Ashburn",
                "zip": "20149",
                "lat": 39.03,
                "lon": -77.5,
                "timezone": "America/New_York",
                "isp": "Google LLC",
                "org": "Google Public DNS",
                "as": "AS15169 Google LLC",
                "query": "8.8.8.8",
            },
            response_text="That's Ashburn, Virginia.",
        )
    )

    text = session.context_text()
    assert "lat=39.03" in text
    assert "lon=-77.5" in text
    assert "[truncated," not in text


def test_add_turn_trims_to_max_turns():
    session = VishConversationSession(max_turns=2)
    for i in range(4):
        session.add_turn(
            Turn(user_input=f"turn {i}", tool_name=None, tool_success=False, tool_result=None, response_text=f"reply {i}")
        )

    assert len(session.turns) == 2
    assert [t.user_input for t in session.turns] == ["turn 2", "turn 3"]
