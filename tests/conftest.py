import pytest

from vish_agent.llm.client import LLMClient


@pytest.fixture(scope="session")
def llm_client() -> LLMClient:
    """Load the configured chat model once and share it across the whole test session."""
    return LLMClient.get_shared()
