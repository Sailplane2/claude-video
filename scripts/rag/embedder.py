"""OpenAI embedder: convert text chunks to embedding vectors via text-embedding-3-small."""

import os
import time
from pathlib import Path

from .utils import EMBED_BATCH_SIZE, MAX_RETRIES, BACKOFF_BASE


def _load_openai_key() -> str:
    """Load OpenAI API key from environment or config file."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key

    # Try the watch config file
    env_file = Path.home() / ".config" / "watch" / ".env"
    if env_file.exists():
        for line in env_file.read_text().split("\n"):
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return ""


def _load_openai_base_url() -> str:
    """Load OpenAI base URL if using a proxy."""
    return os.environ.get("OPENAI_BASE_URL", "")


def embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a list of text strings into 1536-dimension vectors.

    Batches requests and retries on transient failures.
    Raises RuntimeError if no API key is found.
    Raises ImportError if openai package is not installed.
    """
    from openai import OpenAI, RateLimitError, APIError

    api_key = _load_openai_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or add it to ~/.config/watch/.env"
        )

    base_url = _load_openai_base_url()
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(model=model, input=batch)
                # Sort by index to preserve input order
                ordered = sorted(response.data, key=lambda d: d.index)
                all_embeddings.extend([d.embedding for d in ordered])
                break
            except RateLimitError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
            except APIError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(BACKOFF_BASE * (2 ** attempt))

    return all_embeddings


def embed_single(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Embed a single text string. Convenience wrapper."""
    return embed([text], model=model)[0]
