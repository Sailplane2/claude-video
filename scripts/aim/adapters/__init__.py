"""Adapter registry for AI provider adapters."""

from .base import ProviderAdapter, ProviderResponse
from .openai import OpenAIAdapter
from .gemini import GeminiAdapter


_registry = {
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
}


def get_adapter(provider: str = "openai") -> ProviderAdapter:
    """Get an adapter instance for the given provider name.

    Args:
        provider: Provider name ('openai' or 'gemini').

    Returns:
        An instance of the provider's adapter class.

    Raises:
        ValueError: If the provider is not registered.
    """
    if provider not in _registry:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {', '.join(_registry.keys())}"
        )
    return _registry[provider]()


def register_adapter(name: str, adapter_class: type) -> None:
    """Register a new adapter class."""
    _registry[name] = adapter_class


def list_available_providers() -> list:
    """List all registered provider names."""
    return list(_registry.keys())


__all__ = [
    "ProviderAdapter", "ProviderResponse",
    "OpenAIAdapter", "GeminiAdapter",
    "get_adapter", "register_adapter", "list_available_providers",
]
