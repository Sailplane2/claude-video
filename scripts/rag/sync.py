"""Sync orchestrator: incremental update — new, stale, deleted."""

import sys
import time

from .scanner import scan
from .chunker import chunk_summary
from .chroma_store import (
    upsert_chunks, delete_video, get_video_ids, get_chunk_count, rebuild_collection,
)
from .rebuild import EMBED_MODEL, CHUNKER_VERSION


def run() -> int:
    """Run incremental sync: add new, update stale, prune deleted/archived.

    Returns 0 on success, 1 on error.
    """
    result = scan()
    indexed_ids = get_video_ids()
    chunk_count_before = get_chunk_count()

    new_count = 0
    updated_count = 0
    pruned_count = 0
    errors = 0

    # Process each video on disk
    for video in result.videos:
        status = video.status

        # Remove deleted/archived
        if status in ("deleted", "archived"):
            if video.video_id in indexed_ids:
                try:
                    delete_video(video.video_id)
                    pruned_count += 1
                    print(f"  PRUNE {video.video_id} (status={status})")
                except Exception as e:
                    print(f"  PRUNE ERROR {video.video_id}: {e}", file=sys.stderr)
                    errors += 1
            continue

        # Skip non-indexable
        if status not in ("active", "indexed"):
            continue

        # New video (not in Chroma)
        if video.video_id not in indexed_ids:
            try:
                chunks = chunk_summary(video.body, video.frontmatter)
                upsert_chunks(chunks, video, EMBED_MODEL, CHUNKER_VERSION)
                new_count += 1
                print(f"  NEW  {video.video_id} → {len(chunks)} chunks "
                      f"({video.frontmatter.get('title', 'untitled')[:60]})")
            except Exception as e:
                print(f"  NEW  ERROR {video.video_id}: {e}", file=sys.stderr)
                errors += 1
            continue

        # Existing — check hash (simplified: always re-ingest if status=active)
        if status == "active":
            try:
                delete_video(video.video_id)
                chunks = chunk_summary(video.body, video.frontmatter)
                upsert_chunks(chunks, video, EMBED_MODEL, CHUNKER_VERSION)
                updated_count += 1
            except Exception as e:
                print(f"  UPDATE ERROR {video.video_id}: {e}", file=sys.stderr)
                errors += 1

    chunk_count_after = get_chunk_count()
    print()
    print(f"Sync complete: {new_count} new, {updated_count} updated, "
          f"{pruned_count} pruned ({chunk_count_before} → {chunk_count_after} chunks)")

    return 1 if errors else 0
