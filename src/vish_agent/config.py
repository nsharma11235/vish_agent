import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = Path(os.getenv("VISH_LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("VISH_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_DEVICE = os.getenv("VISH_MODEL_DEVICE", "auto")

# LLMClient always checks the local Hugging Face cache first and only hits
# the network if nothing is cached yet. Set this to make that a hard
# requirement: raise instead of silently downloading when the cache is empty.
MODEL_LOCAL_FILES_ONLY = os.getenv("VISH_MODEL_LOCAL_FILES_ONLY", "false").lower() == "true"

CLASSIFIER_TEMPERATURE = float(os.getenv("VISH_CLASSIFIER_TEMPERATURE", "0.0"))
ORCHESTRATOR_TEMPERATURE = float(os.getenv("VISH_ORCHESTRATOR_TEMPERATURE", "0.1"))
RESPONSE_TEMPERATURE = float(os.getenv("VISH_RESPONSE_TEMPERATURE", "0.7"))

FUZZY_MATCH_THRESHOLD = int(os.getenv("VISH_FUZZY_MATCH_THRESHOLD", "60"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# The NWS API asks callers to identify themselves with a descriptive User-Agent.
# https://www.weather.gov/documentation/services-web-api
NWS_USER_AGENT = os.getenv(
    "VISH_NWS_USER_AGENT", "vish-agent-research-project (contact: set VISH_NWS_USER_AGENT)"
)

HTTP_TIMEOUT_SECONDS = float(os.getenv("VISH_HTTP_TIMEOUT_SECONDS", "10"))

# Standardized labeled prompt set for the evaluation harness (reqs 10-13).
EVAL_PROMPTS_PATH = Path(os.getenv("VISH_EVAL_PROMPTS_PATH", str(PROJECT_ROOT / "eval_prompts.json")))

# How many prior turns a ConversationSession keeps for cross-turn context
# (e.g. reusing a location resolved in an earlier turn for a later tool call).
MAX_HISTORY_TURNS = int(os.getenv("VISH_MAX_HISTORY_TURNS", "5"))

# Tool results embedded in conversation history are truncated to this many
# characters. Some tools (e.g. the weather broadcast) return thousands of
# characters of free text; left untruncated, a few turns of that flood the
# model's context and it starts echoing whatever tool name dominates the
# token stream instead of reasoning about the current request. Structured
# results (dicts like ip_api_geolocation's) stay well under this cap already,
# so the values later turns actually reuse (e.g. lat=/lon=) survive.
MAX_TOOL_RESULT_CONTEXT_CHARS = int(os.getenv("VISH_MAX_TOOL_RESULT_CONTEXT_CHARS", "300"))