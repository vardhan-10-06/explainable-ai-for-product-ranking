"""
llm/llm_client.py
===================
Thin LLM client abstraction. Two providers are supported:

  * "template" (default) - a fully offline, deterministic generator that
    turns the evidence into a natural-language explanation using string
    templates instead of a model call. This keeps the whole pipeline
    runnable with zero external dependencies/API keys, and is also what
    the hallucination detector's unit tests run against for determinism.

  * "openai_compatible" - calls any OpenAI-compatible chat completions
    endpoint. This covers hosted OpenAI models as well as self-hosted
    Llama / Qwen deployments that expose an OpenAI-compatible API (e.g.
    vLLM, text-generation-inference, Ollama's OpenAI shim) -- the request
    format is identical, only `base_url` and `model` change in config.

Trade-off: a real production system would also want streaming, retries,
and token-budget-aware truncation; those are noted as extension points in
the docstrings rather than implemented in full, to keep this reference
client readable.
"""
from __future__ import annotations

import os
from typing import List, Optional

import requests

from common.config import get_config, get_logger
from llm.prompt_templates import SYSTEM_PROMPT

logger = get_logger(__name__)


class LLMClient:
    """Provider-agnostic client for generating text from either a template engine or a real LLM API."""

    def __init__(self):
        self.cfg = get_config()["llm"]
        self.provider = self.cfg["provider"]

    def generate(self, user_prompt: str) -> str:
        """
        Generate a natural-language response for the given user prompt.

        Args:
            user_prompt: The fully-constructed prompt (see prompt_templates.py).

        Returns:
            The generated explanation text.
        """
        if self.provider == "template":
            return self._generate_template(user_prompt)
        if self.provider == "openai_compatible":
            return self._generate_openai_compatible(user_prompt)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ------------------------------------------------------------------ #
    def _generate_template(self, user_prompt: str) -> str:
        """
        Deterministic, dependency-free explanation generator: extracts the
        evidence lines from the prompt and composes a plain-language
        sentence directly from them (no fabrication possible, since it only
        ever repeats supplied evidence).
        """
        lines = [l.strip("- ").strip() for l in user_prompt.splitlines() if l.strip().startswith("-")]
        significant = [l for l in lines if "No significant" not in l and "No meaningfully" not in l][:3]

        if not significant:
            return (
                "This item was ranked based on the model's overall scoring, but there is "
                "not strong, specific evidence available to explain the decision in detail."
            )

        readable = []
        for line in significant:
            feature_part = line.split("(")[0].strip()
            readable.append(feature_part.replace("_", " "))

        explanation = (
            "This product was ranked highly mainly because of: " + ", ".join(readable) + ". "
            "These factors had the strongest measured influence on the ranking score for this user."
        )
        return explanation

    def _generate_openai_compatible(self, user_prompt: str) -> str:
        api_key = os.environ.get(self.cfg["openai_compatible"]["api_key_env"], "")
        if not api_key:
            logger.warning(
                "No API key found in env var '%s'; falling back to template mode.",
                self.cfg["openai_compatible"]["api_key_env"],
            )
            return self._generate_template(user_prompt)

        url = f"{self.cfg['openai_compatible']['base_url']}/chat/completions"
        payload = {
            "model": self.cfg["openai_compatible"]["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg["temperature"],
            "max_tokens": self.cfg["max_tokens"],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("LLM API call failed (%s); falling back to template mode.", exc)
            return self._generate_template(user_prompt)
