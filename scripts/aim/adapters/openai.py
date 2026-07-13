"""OpenAI API adapter with retry logic, cost computation, and error handling.

Reads OPENAI_API_KEY from:
  1. OPENAI_API_KEY environment variable
  2. ~/.config/watch/.env (line starting with OPENAI_API_KEY=)
"""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict

from openai import OpenAI
from openai import APIError, AuthenticationError, RateLimitError, APITimeoutError
from openai import BadRequestError, APIConnectionError

from .base import ProviderAdapter, ProviderResponse
from ..errors import (
    NoAPIKeyError, ProviderAuthError, ProviderRateLimitError,
    ProviderOverloadError, ProviderContextLengthError, ProviderError,
)


def _read_openai_api_key() -> str:
    """Read OPENAI_API_KEY from environment or ~/.config/watch/.env."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = Path.home() / ".config" / "watch" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return ""


MODEL_COSTS = {
    "gpt-5.6-sol":       {"input": 5.00, "output": 30.00},
    "gpt-5.6-terra":     {"input": 2.50, "output": 15.00},
    "gpt-5.6-luna":      {"input": 1.00, "output": 6.00},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
    "gpt-4-turbo":       {"input": 10.00, "output": 30.00},
    "o1-preview":        {"input": 15.00, "output": 60.00},
    "o1-mini":           {"input": 3.00, "output": 12.00},
}

DEFAULT_COST = {"input": 2.50, "output": 10.00}


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI chat completions API."""

    name: str = "openai"
    default_model: str = "gpt-4o"
    supports_vision: bool = True

    def __init__(self):
        api_key = _read_openai_api_key()
        if not api_key:
            raise NoAPIKeyError(provider="openai")
        self.client = OpenAI(api_key=api_key)
        self.cost_per_million_input = 2.50  # gpt-4o default
        self.cost_per_million_output = 10.00

    def _get_model_cost(self, model: str) -> dict:
        """Get cost rates for a given model, with fallback."""
        # Check exact match
        if model in MODEL_COSTS:
            return MODEL_COSTS[model]
        # Check prefix match (e.g. gpt-4o-2024-08-06)
        for key, costs in MODEL_COSTS.items():
            if model.startswith(key):
                return costs
        return DEFAULT_COST

    # Newer model families reject `max_tokens` and require `max_completion_tokens`
    # instead (first seen with o1-*, confirmed also true for gpt-5.6-*).
    _NEW_PARAM_PREFIXES = ("o1-", "gpt-5.6-")

    def call(self, messages: List[Dict], model: Optional[str] = None,
             max_tokens: Optional[int] = None, temperature: float = 0.7,
             timeout: int = 60) -> ProviderResponse:
        """Send a chat completion request to OpenAI."""
        model = model or self.default_model
        start_time = time.time()
        uses_max_completion_tokens = model.startswith(self._NEW_PARAM_PREFIXES)
        if max_tokens is None:
            # Reasoning models (o1-*, gpt-5.6-*) spend part of this budget on
            # hidden reasoning tokens before any visible output — 2048 can be
            # entirely consumed by reasoning with nothing visible left over.
            max_tokens = 16000 if uses_max_completion_tokens else 2048

        last_error = None
        # Retry loop for 429 and 5xx
        retry_delays_429 = [5, 15, 45]
        retry_delays_5xx = [1, 3, 9]
        attempt_429 = 0
        attempt_5xx = 0

        while True:
            try:
                token_param = (
                    {"max_completion_tokens": max_tokens}
                    if uses_max_completion_tokens
                    else {"max_tokens": max_tokens}
                )
                # o1-*/gpt-5.6-* reasoning models only support the default
                # temperature (1) and reject any explicit override.
                temp_param = {} if uses_max_completion_tokens else {"temperature": temperature}
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=timeout,
                    **token_param,
                    **temp_param,
                )
                elapsed_ms = int((time.time() - start_time) * 1000)

                choice = response.choices[0]
                content = choice.message.content or ""

                tokens_in = response.usage.prompt_tokens if response.usage else 0
                tokens_out = response.usage.completion_tokens if response.usage else 0

                costs = self._get_model_cost(model)
                cost = (tokens_in * costs["input"] / 1_000_000) + \
                       (tokens_out * costs["output"] / 1_000_000)

                return ProviderResponse(
                    content=content,
                    model=response.model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    response_id=response.id,
                    finish_reason=str(choice.finish_reason or "stop"),
                    cost=cost,
                    elapsed_ms=elapsed_ms,
                )

            except AuthenticationError as e:
                raise ProviderAuthError(
                    f"OpenAI API key is invalid. Check your OPENAI_API_KEY environment variable. "
                    f"Detail: {e}"
                )
            except BadRequestError as e:
                error_body = str(e)
                if "context_length_exceeded" in error_body:
                    raise ProviderContextLengthError(
                        context_tokens=0, provider="openai"
                    )
                raise ProviderError(f"OpenAI bad request: {e}")
            except RateLimitError as e:
                attempt_429 += 1
                if attempt_429 <= len(retry_delays_429):
                    delay = retry_delays_429[attempt_429 - 1]
                    time.sleep(delay)
                    last_error = e
                    continue
                raise ProviderRateLimitError(
                    f"Rate limited by OpenAI after {attempt_429} retries. "
                    f"Reduce send frequency. Detail: {e}"
                )
            except APITimeoutError as e:
                # Retry once after 3s for timeouts
                if attempt_5xx < 1:
                    time.sleep(3)
                    attempt_5xx += 1
                    last_error = e
                    continue
                raise ProviderOverloadError(
                    f"OpenAI API timeout after retry. Check network connectivity. Detail: {e}"
                )
            except APIConnectionError as e:
                raise ProviderOverloadError(
                    f"No network connection to api.openai.com. "
                    f"Check your internet and try again. Detail: {e}"
                )
            except APIError as e:
                status = getattr(e, 'status_code', 0)
                if 500 <= status < 600:
                    attempt_5xx += 1
                    if attempt_5xx <= len(retry_delays_5xx):
                        delay = retry_delays_5xx[attempt_5xx - 1]
                        time.sleep(delay)
                        last_error = e
                        continue
                    raise ProviderOverloadError(
                        f"OpenAI API temporarily unavailable ({status}). "
                        f"Try again later. Detail: {e}"
                    )
                raise ProviderError(f"OpenAI API error ({status}): {e}")
            except Exception as e:
                raise ProviderError(f"Unexpected OpenAI API error: {e}")
