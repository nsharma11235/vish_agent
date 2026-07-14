"""Pre-downloads Qwen2.5-1.5B-Instruct into the local Hugging Face cache.

Run this once while online. After that, QwenClient loads straight from
cache with no network access required (see VISH_MODEL_LOCAL_FILES_ONLY
in .env.example to make that a hard requirement).
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from vish_agent.config import MODEL_NAME


def main() -> None:
    print(f"Downloading {MODEL_NAME}...")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Done. The model is now cached locally.")


if __name__ == "__main__":
    main()
