"""Abstract adapter interface for AI provider adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ProviderResponse:
    """Standardized response from any AI provider."""
    content: str                        # Response text
    model: str                          # Model that actually served it
    tokens_in: int                      # Input tokens consumed
    tokens_out: int                     # Output tokens generated
    response_id: str                    # Provider's unique response ID
    finish_reason: str                  # "stop", "length", "content_filter", "error"
    cost: float                         # Computed cost in USD
    elapsed_ms: int = 0                 # Wall-clock time in milliseconds


class ProviderAdapter(ABC):
    """Base class for AI provider adapters."""

    name: str = ""
    default_model: str = ""
    supports_vision: bool = False
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0

    @abstractmethod
    def call(self, messages: List[Dict], model: Optional[str] = None,
             max_tokens: int = 2048, temperature: float = 0.7,
             timeout: int = 60) -> ProviderResponse:
        """Send a chat completion request and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model identifier. Defaults to self.default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0-2.0).
            timeout: Request timeout in seconds.

        Returns:
            ProviderResponse with the response content and metadata.

        Raises:
            ProviderAuthError: API key missing or invalid.
            ProviderRateLimitError: Rate limited by provider.
            ProviderOverloadError: Provider server unavailable.
            ProviderContextLengthError: Context exceeds provider limit.
            ProviderError: Other API errors.
        """
        pass

    def compute_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Compute cost for a call using provider's per-token rates."""
        return (tokens_in * self.cost_per_million_input / 1_000_000) + \
               (tokens_out * self.cost_per_million_output / 1_000_000)
