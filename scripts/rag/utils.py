"""Shared utilities: path helpers, hashing, token counting, config loading."""

import hashlib
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

def vault_root() -> Path:
    """Return the Obsidian vault root path."""
    return Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/michael-brain"


def videos_dir() -> Path:
    """Return the save-watched-videos directory."""
    return vault_root() / "videos" / "save-watched-videos"


def index_path() -> Path:
    """Return path to index.json."""
    return videos_dir() / "index.json"


def chroma_dir() -> Path:
    """Return path to the Chroma persistent storage directory."""
    return Path.home() / ".claude-video" / "chroma"


def documents_dir() -> Path:
    """Return the documents directory (extracted PDF markdown)."""
    return vault_root() / "documents"


def documents_index_path() -> Path:
    """Return path to documents/index.json manifest."""
    return documents_dir() / "index.json"


def docs_source_dir() -> Path:
    """Return the drop folder for source PDFs (kept out of the vault)."""
    return Path.home() / "claude-video" / "docs" / "source"


def eval_set_path() -> Path:
    """Return path to the eval set JSON file."""
    return Path.home() / ".claude-video" / "eval_set.json"


# ── Hashing ────────────────────────────────────────────────────────────────

def hash_text(text: str) -> str:
    """SHA256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """SHA256 hash of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Token counting ─────────────────────────────────────────────────────────

TOKEN_CHAR_RATIO = 4  # ~4 characters per token for English text


def count_tokens(text: str) -> int:
    """Approximate token count for OpenAI embedding models."""
    return max(1, len(text) // TOKEN_CHAR_RATIO)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit, preserving sentence boundaries."""
    max_chars = max_tokens * TOKEN_CHAR_RATIO
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at last sentence boundary
    last_period = truncated.rfind(". ")
    last_newline = truncated.rfind("\n")
    cut = max(
        last_period + 1 if last_period >= 0 else 0,
        last_newline if last_newline >= 0 else 0,
    )
    if cut > max_chars // 2:
        return text[:cut]
    return truncated


# ── Date helpers ───────────────────────────────────────────────────────────

def date_to_epoch(date_str: str) -> int:
    """Convert YYYY-MM-DD string to Unix epoch. Returns 0 on failure."""
    import time
    from datetime import datetime

    if not date_str:
        return 0
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return int(dt.timestamp())
    except (ValueError, AttributeError):
        return 0


# ── Config ─────────────────────────────────────────────────────────────────

CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 1000
FULL_SUMMARY_MAX_TOKENS = 2000
EMBED_BATCH_SIZE = 20
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds
DEFAULT_SEARCH_K = 25


# ── Backend selection ──────────────────────────────────────────────────────

def get_store(backend: str = "neo4j"):
    """Return the storage module for the requested backend.

    Both chroma_store and neo4j_store expose the same video-facing surface so
    rebuild/sync/search/status/validate/prune work unchanged behind a
    `--backend {chroma,neo4j}` flag. neo4j_store additionally exposes document
    functions. Neo4j is the default; Chroma remains available as the frozen
    fallback via `--backend chroma`.
    """
    if backend == "neo4j":
        from . import neo4j_store
        return neo4j_store
    from . import chroma_store
    return chroma_store
