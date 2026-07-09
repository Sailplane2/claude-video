"""Gemini API adapter with retry logic, cost computation, and error handling.

Reads GEMINI_API_KEY from:
  1. GEMINI_API_KEY environment variable
  2. ~/.config/watch/.env (line starting with GEMINI_API_KEY=)
"""

import os
import time
import re
import warnings
from pathlib import Path
from typing import Optional, List, Dict

from .base import ProviderAdapter, ProviderResponse
from ..errors import (
    NoAPIKeyError, ProviderAuthError, ProviderRateLimitError,
    ProviderOverloadError, ProviderContextLengthError, ProviderError,
)


MODEL_COSTS = {
    "gemini-2.5-flash":   {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro":     {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash":   {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro":     {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash":   {"input": 0.075, "output": 0.30},
}

DEFAULT_COST = {"input": 0.15, "output": 0.60}


def _read_gemini_api_key() -> str:
    """Read GEMINI_API_KEY from environment or ~/.config/watch/.env."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Try ~/.config/watch/.env
    env_path = Path.home() / ".config" / "watch" / ".env"
    if env_path.exists():
        try:
            content = env_path.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        except (IOError, OSError):
            pass
    return ""


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini API."""

    name: str = "gemini"
    default_model: str = "gemini-2.5-flash"
    supports_vision: bool = True

    def __init__(self):
        api_key = _read_gemini_api_key()
        if not api_key:
            raise NoAPIKeyError(provider="gemini")
        # Lazy import to suppress deprecation warnings until adapter is actually used
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as _genai
        self._genai = _genai
        self._genai.configure(api_key=api_key)
        self.cost_per_million_input = 0.15
        self.cost_per_million_output = 0.60

    def _get_model_cost(self, model: str) -> dict:
        """Get cost rates for a given model, with fallback."""
        if model in MODEL_COSTS:
            return MODEL_COSTS[model]
        for key, costs in MODEL_COSTS.items():
            if model.startswith(key):
                return costs
        return DEFAULT_COST

    def _convert_messages_to_gemini(self, messages: List[Dict]) -> tuple:
        """Convert OpenAI-format messages to Gemini content format.

        Returns (system_instruction, contents).

        Merges consecutive same-role messages — Gemini requires strictly
        alternating user/model roles, but merge_context can produce
        consecutive user messages from brief + transcript injection.
        """
        system_text = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini uses system_instruction parameter (only first one)
                if system_text is None:
                    system_text = content
                else:
                    system_text += "\n\n" + content
            elif role == "user":
                # Merge consecutive user messages to prevent Gemini error
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append(content)
                else:
                    contents.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                # Merge consecutive assistant messages
                if contents and contents[-1]["role"] == "model":
                    contents[-1]["parts"].append(content)
                else:
                    contents.append({"role": "model", "parts": [content]})

        return system_text, contents

    def call(self, messages: List[Dict], model: Optional[str] = None,
             max_tokens: int = 2048, temperature: float = 0.7,
             timeout: int = 60) -> ProviderResponse:
        """Send a chat completion request to Gemini."""
        model = model or self.default_model
        start_time = time.time()

        system_instruction, contents = self._convert_messages_to_gemini(messages)

        # Gemini safety - need at least one user message and one model message
        # Ensure alternate user/model roles
        if len(contents) < 1:
            raise ProviderError("Gemini requires at least one user message.")

        last_error = None
        retry_delays_429 = [5, 15, 45]
        retry_delays_5xx = [1, 3, 9]
        attempt_429 = 0
        attempt_5xx = 0

        while True:
            try:
                gen_model = self._genai.GenerativeModel(
                    model_name=model,
                    system_instruction=system_instruction or None,
                    generation_config=self._genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )

                chat = gen_model.start_chat(history=contents[:-1] if contents else [])
                user_msg = contents[-1]["parts"][0] if contents else ""

                response = chat.send_message(
                    user_msg,
                    request_options={"timeout": timeout * 1000},
                )

                elapsed_ms = int((time.time() - start_time) * 1000)

                response_text = response.text

                # Gemini doesn't provide token counts in the same way
                # Try to get from usage_metadata if available
                try:
                    usage = response.usage_metadata
                    if usage:
                        tokens_in = usage.prompt_token_count or 0
                        tokens_out = usage.candidates_token_count or 0
                    else:
                        tokens_in = 0
                        tokens_out = 0
                except (AttributeError, Exception):
                    tokens_in = 0
                    tokens_out = 0

                costs = self._get_model_cost(model)
                cost = (tokens_in * costs["input"] / 1_000_000) + \
                       (tokens_out * costs["output"] / 1_000_000)

                finish_reason = "stop"
                try:
                    candidate = response.candidates[0]
                    if candidate.finish_reason:
                        finish_reason = str(candidate.finish_reason.name).lower()
                except (IndexError, AttributeError):
                    pass

                response_id = f"gemini-{int(start_time)}"

                return ProviderResponse(
                    content=response_text,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    response_id=response_id,
                    finish_reason=finish_reason,
                    cost=cost,
                    elapsed_ms=elapsed_ms,
                )

            except Exception as e:
                error_str = str(e).lower()

                # Auth errors
                if any(term in error_str for term in ["api_key", "api key", "unauthorized", "permission"]):
                    raise ProviderAuthError(
                        f"Gemini API key is invalid or unauthorized. "
                        f"Check your GEMINI_API_KEY. Detail: {e}"
                    )

                # Context length
                if any(term in error_str for term in ["context_length", "maximum context", "too long", "token"]):
                    raise ProviderContextLengthError(
                        context_tokens=0, provider="gemini"
                    )

                # Rate limit
                if any(term in error_str for term in ["rate_limit", "rate limit", "quota", "resource exhausted", "429"]):
                    attempt_429 += 1
                    if attempt_429 <= len(retry_delays_429):
                        delay = retry_delays_429[attempt_429 - 1]
                        time.sleep(delay)
                        last_error = e
                        continue
                    raise ProviderRateLimitError(
                        f"Rate limited by Gemini after {attempt_429} retries. "
                        f"Reduce send frequency. Detail: {e}"
                    )

                # Server errors
                if any(term in error_str for term in ["500", "503", "service unavailable", "internal"]):
                    attempt_5xx += 1
                    if attempt_5xx <= len(retry_delays_5xx):
                        delay = retry_delays_5xx[attempt_5xx - 1]
                        time.sleep(delay)
                        last_error = e
                        continue
                    raise ProviderOverloadError(
                        f"Gemini API temporarily unavailable. Try again later. Detail: {e}"
                    )

                # Timeout / connection
                if any(term in error_str for term in ["timeout", "deadline", "connection", "unavailable"]):
                    raise ProviderOverloadError(
                        f"Gemini API timeout or connection error. "
                        f"Check your internet and try again. Detail: {e}"
                    )

                # Generic
                raise ProviderError(f"Gemini API error: {e}")
