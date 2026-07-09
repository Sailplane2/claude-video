"""Section chunker: split summary body into embeddable chunks with contextual headers."""

import re
from dataclasses import dataclass

from .utils import count_tokens, truncate_to_tokens, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS, FULL_SUMMARY_MAX_TOKENS


@dataclass
class Chunk:
    text: str
    section_title: str
    chunk_index: str
    n_tokens: int
    n_bytes: int


def make_contextual_header(metadata: dict) -> str:
    """Build a contextual header that prepends every chunk for vocabulary injection."""
    title = metadata.get("title", "Untitled")
    channel = metadata.get("channel", "Unknown")
    tags = metadata.get("tags", [])

    if isinstance(tags, str):
        tags_str = tags
    elif isinstance(tags, list):
        tags_str = ", ".join(tags)
    else:
        tags_str = ""

    return f"Video: {title} | Channel: {channel} | Tags: {tags_str}"


def _split_on_h2(body: str) -> list[tuple[str, str]]:
    """Split markdown body on H2 headings. Returns [(heading, content), ...]."""
    # Match H2 headings (## Title) — not H3 or deeper
    pattern = r"^##\s+(.+)$"
    sections = []
    current_title = "Introduction"
    current_content: list[str] = []
    started = False

    for line in body.split("\n"):
        m = re.match(pattern, line.strip())
        if m:
            if started or current_content:
                sections.append((current_title, "\n".join(current_content).strip()))
            current_title = m.group(1).strip()
            current_content = []
            started = True
        else:
            current_content.append(line)

    # Last section
    if started or current_content:
        sections.append((current_title, "\n".join(current_content).strip()))

    return sections


def _split_on_paragraphs(body: str) -> list[str]:
    """Split body on double-newline paragraph boundaries."""
    parts = re.split(r"\n\n+", body.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_summary(body: str, metadata: dict) -> list[Chunk]:
    """Split a video summary body into embeddable chunks with contextual headers.

    Produces:
    - 1 "full" chunk (capped at ~2000 tokens)
    - N section chunks (one per H2 heading, capped at ~1000 tokens)
    """
    chunks: list[Chunk] = []
    header = make_contextual_header(metadata)
    if not body or not body.strip():
        text = f"{header}\n\n## Full Summary\n\n(No content)"
        chunks.append(Chunk(text, "full", "full", count_tokens(text), len(text)))
        return chunks

    # ── Full summary chunk ──
    full_body = truncate_to_tokens(body.strip(), FULL_SUMMARY_MAX_TOKENS)
    full_text = f"{header}\n\n## Full Summary\n\n{full_body}"
    chunks.append(Chunk(
        text=full_text,
        section_title="full",
        chunk_index="full",
        n_tokens=count_tokens(full_text),
        n_bytes=len(full_text.encode("utf-8")),
    ))

    # ── Section chunks ──
    sections = _split_on_h2(body)

    if len(sections) > 1:
        for i, (title, content) in enumerate(sections):
            if not content.strip():
                continue
            section_text = f"{header}\n\n## {title}\n\n{content.strip()}"
            section_text = truncate_to_tokens(section_text, CHUNK_MAX_TOKENS)
            n_tok = count_tokens(section_text)
            if n_tok < CHUNK_MIN_TOKENS:
                continue
            chunks.append(Chunk(
                text=section_text,
                section_title=title,
                chunk_index=f"sec_{i}",
                n_tokens=n_tok,
                n_bytes=len(section_text.encode("utf-8")),
            ))
    else:
        # Fallback: paragraph-based chunks for flat summaries
        paragraphs = _split_on_paragraphs(body)
        # Merge short paragraphs
        merged: list[str] = []
        buf = ""
        for p in paragraphs:
            combined = f"{buf}\n\n{p}".strip() if buf else p
            if count_tokens(combined) < CHUNK_MIN_TOKENS and merged:
                buf = combined
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        for i, para in enumerate(merged):
            section_text = f"{header}\n\n## Summary (Part {i + 1})\n\n{para}"
            section_text = truncate_to_tokens(section_text, CHUNK_MAX_TOKENS)
            n_tok = count_tokens(section_text)
            chunks.append(Chunk(
                text=section_text,
                section_title="summary",
                chunk_index=f"sec_{i}",
                n_tokens=n_tok,
                n_bytes=len(section_text.encode("utf-8")),
            ))

    return chunks
