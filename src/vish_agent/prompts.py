"""Centralized prompt text for every layer that talks to the model."""

SYSTEM_PROMPT = (
    "You are Vish, a concise local AI assistant. You can answer directly from "
    "general knowledge or use tools to look up live information such as IP "
    "geolocation, weather, and web search. Prioritize tool use over general "
    "knowledge when possible."
)

CONVERSATION_START_PROMPT = """Introduce yourself. Mention your name, and that you're able to select tools from your toolbox.\
Your available tools are {toolbox}.
"""


RESPONSE_TYPE_CLASSIFIER_PROMPT = """Classify the user's message into exactly one category. \
Reply with only the category word, nothing else.

Categories:
- conversational: answerable from general knowledge; no live/external data needed.
- tool_assisted: answering well requires live/external data (e.g. current location, current weather, current web/news information).

Examples:
User message: What's the capital of France?
Category: tool_assisted

User message: What is the weather like right now at my current location?
Category: tool_assisted

User message: Can you tell me a joke?
Category: conversational

User message: Search the web for today's top news headlines.
Category: tool_assisted

User message: {user_input}

Category:"""

# The examples matter more than the tool descriptions here: measured on a
# 15-prompt selection suite (2 sampled passes each), the example-free prompt
# scored 24/30 because memory-answerable knowledge questions ("Who wrote War
# and Peace?") fell out to NONE, and no description wording recovered them —
# stronger "prefer search over memory" phrasing scored *worse* (16-20/30).
# Knowledge->search examples spanning question forms (when/how tall/who/what
# is) scored 30/30. If a tool is added or renamed, update its example too.
TOOL_SELECTION_PROMPT = """You must choose exactly one tool to help answer the user's message, \
or reply NONE if no tool applies.

Available tools:
{tool_descriptions}

Examples:
User message: When did the Roman Empire fall?
Tool name: tavily_web_search

User message: When did Canberra become the capital of Australia?
Tool name: tavily_web_search

User message: How tall is the Eiffel Tower?
Tool name: tavily_web_search

User message: Who painted the Mona Lisa?
Tool name: tavily_web_search

User message: What is quantum entanglement?
Tool name: tavily_web_search

User message: Where is the IP address 9.9.9.9 located?
Tool name: ip_api_geolocation

User message: What is the weather at 35.0, -90.0?
Tool name: nws_weather_radio

User message: Tell me something funny.
Tool name: NONE

User message: {user_input}

Reply with only the tool name (or NONE), nothing else.

Tool name:"""

RESPONSE_GENERATION_PROMPT = """{system_prompt}

{conversation_context}User asked: {user_input}

Expected response type: {expected_response_type}
Tool used: {tool_name}
Tool result: {tool_result}

Write a concise, natural language response to the user that incorporates the tool \
result if present. Do not mention internal implementation details."""
