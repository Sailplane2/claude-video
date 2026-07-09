"""Rebuild orchestrator: scan → chunk → embed → store. Full index rebuild from scratch."""

import sys
import time

from .scanner import scan
from .chunker import chunk_summary
from .chroma_store import rebuild_collection, upsert_chunks, get_chunk_count


EMBED_MODEL = "text-embedding-3-small"
CHUNKER_VERSION = "v1.0"


def run(force: bool = False) -> int:
    """Run a full rebuild of the RAG index.

    Args:
        force: Skip confirmation prompt.

    Returns:
        0 on success, 1 on error.
    """
    if not force:
        print("This will drop and rebuild the entire RAG index.")
        print("All videos will be re-chunked and re-embedded (OpenAI API costs apply).")
        try:
            resp = input("Continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 0

    print("Scanning vault for video notes...")
    result = scan()

    if not result.videos:
        print("No videos found. Creating empty collection.", file=sys.stderr)
        rebuild_collection()
        print("Empty collection created. Index 0 videos, 0 chunks.")
        return 0

    print(f"Found {len(result.videos)} videos ({len(result.issues)} issues).")

    # Show issues but don't block
    for issue in result.issues:
        if issue.severity == "error":
            print(f"  ERROR [{issue.category}] {issue.description}", file=sys.stderr)
        elif issue.severity == "warning":
            print(f"  WARN  [{issue.category}] {issue.description}", file=sys.stderr)

    # Drop and recreate
    print("Rebuilding Chroma collection...")
    rebuild_collection()

    # Process each video
    total_chunks = 0
    errors = 0
    start_time = time.time()

    for i, video in enumerate(result.videos):
        status = video.status
        if status in ("archived", "deleted"):
            print(f"  [{i + 1}/{len(result.videos)}] SKIP {video.video_id} (status={status})")
            continue

        try:
            chunks = chunk_summary(video.body, video.frontmatter)
            upsert_chunks(chunks, video, EMBED_MODEL, CHUNKER_VERSION)
            total_chunks += len(chunks)
            print(f"  [{i + 1}/{len(result.videos)}] {video.video_id} → {len(chunks)} chunks "
                  f"({video.frontmatter.get('title', 'untitled')[:60]})")
        except Exception as e:
            print(f"  [{i + 1}/{len(result.videos)}] ERROR {video.video_id}: {e}", file=sys.stderr)
            errors += 1

    elapsed = time.time() - start_time
    print()
    print(f"Rebuild complete: {len(result.videos)} videos → {total_chunks} chunks in {elapsed:.1f}s")
    if errors:
        print(f"  {errors} error(s) — see above", file=sys.stderr)
        return 1
    return 0
