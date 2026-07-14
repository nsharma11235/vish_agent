"""Thin, model-agnostic wrapper around a locally-loaded Hugging Face chat model.

The model is chosen through environment variables (VISH_MODEL_NAME,
VISH_MODEL_DEVICE — see config.py); any causal LM whose tokenizer provides a
chat template works. Every layer that needs inference goes through this class
rather than touching Transformers directly, so the model only has to load once
per process and so layers stay unit-testable via dependency injection.
"""

from __future__ import annotations

import threading
from typing import Optional

from vish_agent.config import MODEL_DEVICE, MODEL_LOCAL_FILES_ONLY, MODEL_NAME


class LLMClient:
    _instance: Optional["LLMClient"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = MODEL_DEVICE,
        strict_offline: bool = MODEL_LOCAL_FILES_ONLY,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = self._load_local_first(AutoTokenizer, model_name, strict_offline=strict_offline)
        self.model = self._load_local_first(
            AutoModelForCausalLM,
            model_name,
            strict_offline=strict_offline,
            device_map=device,
            torch_dtype="auto",
        )

    @staticmethod
    def _load_local_first(loader, model_name: str, *, strict_offline: bool, **kwargs):
        """Load from the local Hugging Face cache without touching the
        network when a cached copy already exists. Falls back to downloading
        only if nothing is cached yet, unless strict_offline is set."""
        try:
            return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
        except OSError:
            if strict_offline:
                raise OSError(
                    f"'{model_name}' is not cached locally and VISH_MODEL_LOCAL_FILES_ONLY is set. "
                    "Run `python scripts/download_model.py` once while online, then retry."
                ) from None
            return loader.from_pretrained(model_name, local_files_only=False, **kwargs)

    @classmethod
    def get_shared(cls) -> "LLMClient":
        """Return a process-wide singleton so the model is loaded at most once."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7, max_new_tokens: int = 256) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
