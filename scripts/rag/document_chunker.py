"""Document chunker: page-aware chunking of extracted PDF markdown.

Mirrors chunker.py for documents. Two tiers:
  - "full" chunk (capped at ~2000 tokens)
  - page-grouped section chunks (capped at ~1000 tokens, split on form-feed)
"""

from .chunker import Chunk
from .utils import (
    count_tokens, truncate_to_tokens,
    CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS, FULL_SUMMARY_MAX_TOKENS,
)


def make_contextual_header(metadata: dict) -> str:
    """Contextual header prepended to every document chunk for vocabulary injection."""
    title = metadata.get("title", "Untitled")
    author = metadata.get("author", "")
    tags = metadata.get("tags", [])

    if isinstance(tags, list):
        tags_str = ", ".join(str(t) for t in tags)
    elif isinstance(tags, str):
        tags_str = tags
    else:
        tags_str = ""

    header = f"Document: {title}"
    if author:
        header += f" | Author: {author}"
    header += f" | Tags: {tags_str}"
    return header


def _split_pages(body: str) -> list[str]:
    """Split body on form-feed page markers (emitted by document_ingest)."""
    pages = [p.strip() for p in body.split("\f") if p.strip()]
    return pages if pages else [body.strip()]


def _group_pages(pages: list[str]) -> list[tuple[str, int, int]]:
    """Group consecutive pages into chunks ≤ CHUNK_MAX_TOKENS.

    Returns list of (text, start_page_idx, end_page_idx) with 0-based page idx.
    """
    groups: list[tuple[str, int, int]] = []
    buf = ""
    start = 0
    for i, page in enumerate(pages):
        combined = f"{buf}\n\n{page}".strip() if buf else page
        if count_tokens(combined) <= CHUNK_MAX_TOKENS:
            buf = combined
        else:
            if buf:
                groups.append((buf, start, i - 1))
            buf = page
            start = i
    if buf:
        groups.append((buf, start, len(pages) - 1))
    return groups


def chunk_document(body: str, metadata: dict) -> list[Chunk]:
    """Split an extracted document body into embeddable chunks.

    Produces one "full" chunk plus page-grouped "sec_N" chunks.
    """
    chunks: list[Chunk] = []
    header = make_contextual_header(metadata)

    if not body or not body.strip():
        text = f"{header}\n\n(No content)"
        chunks.append(Chunk(text, "full", "full", count_tokens(text), len(text.encode("utf-8"))))
        return chunks

    # ── Tier 1: full document chunk ──
    full_body = truncate_to_tokens(body.strip(), FULL_SUMMARY_MAX_TOKENS)
    full_text = f"{header}\n\n{full_body}"
    chunks.append(Chunk(
        text=full_text,
        section_title="full",
        chunk_index="full",
        n_tokens=count_tokens(full_text),
        n_bytes=len(full_text.encode("utf-8")),
    ))

    # ── Tier 2: page-aware section chunks ──
    pages = _split_pages(body)
    groups = _group_pages(pages)

    for i, (group_text, start_idx, end_idx) in enumerate(groups):
        section_text = f"{header}\n\n{truncate_to_tokens(group_text, CHUNK_MAX_TOKENS)}"
        n_tok = count_tokens(section_text)
        if n_tok < CHUNK_MIN_TOKENS:
            continue
        page_label = f"pages {start_idx + 1}-{end_idx + 1}" if start_idx != end_idx else f"page {start_idx + 1}"
        chunks.append(Chunk(
            text=section_text,
            section_title=page_label,
            chunk_index=f"sec_{i}",
            n_tokens=n_tok,
            n_bytes=len(section_text.encode("utf-8")),
        ))

    return chunks
