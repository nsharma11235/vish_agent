import pytest

from vish_agent.llm.qwen_client import QwenClient


@pytest.fixture(scope="session")
def qwen_client() -> QwenClient:
    """Load Qwen2.5-1.5B-Instruct once and share it across the whole test session."""
    return QwenClient.get_shared()
