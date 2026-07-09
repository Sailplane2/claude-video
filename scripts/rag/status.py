"""Status reporter: print coverage report of current RAG state."""

import sys
import time
from datetime import datetime

from .scanner import scan
from .chroma_store import get_chunk_count, get_video_ids, get_collection_info


def run():
    """Print a comprehensive status report of the RAG index."""
    result = scan()
    indexed_ids = get_video_ids()
    chunk_count = get_chunk_count()

    # Categorize videos
    indexed = [v for v in result.videos if v.video_id in indexed_ids]
    active = [v for v in result.videos if v.status == "active" and v.video_id not in indexed_ids]
    archived = [v for v in result.videos if v.status == "archived"]
    deleted = [v for v in result.videos if v.status == "deleted"]

    # Detect stale: hash mismatch between indexed and on-disk
    # (Requires metadata from Chroma — simplified: count indexed that differ)
    # For now, report known counts

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  Video Knowledge Base — Status Report    ║")
    print("╚══════════════════════════════════════════╝")
    print()

    info = get_collection_info()
    print(f"Index:            ~/.claude-video/chroma/")
    print(f"Chunk count:      {chunk_count} chunks")
    try:
        import chromadb
        print(f"Chroma version:   chromadb {chromadb.__version__}")
    except Exception:
        pass
    print(f"Embed model:      text-embedding-3-small")
    print(f"Chunker version:  v1.0")
    print()

    print("Videos:")
    print(f"  Total on disk:         {len(result.videos)}")
    print(f"  Indexed (in Chroma):   {len(indexed)}")
    print(f"  Active (pending):      {len(active)}")
    print(f"  Archived:              {len(archived)}")
    print(f"  Deleted:               {len(deleted)}")
    print()

    # Consistency check
    only_in_chroma = indexed_ids - {v.video_id for v in result.videos}
    only_on_disk = {v.video_id for v in result.videos} - indexed_ids

    if only_in_chroma or only_on_disk:
        print("Filesystem / Chroma consistency: MISMATCH")
        for vid in only_in_chroma:
            print(f"  Orphan in Chroma (no file on disk): {vid}")
        for vid in only_on_disk:
            v = next((x for x in result.videos if x.video_id == vid), None)
            if v:
                print(f"  Not indexed (on disk): {vid} — {v.frontmatter.get('title', 'untitled')[:60]}")
    else:
        print("Filesystem / Chroma consistency: OK")

    print()

    # Issues
    if result.issues:
        print(f"Issues ({len(result.issues)}):")
        for issue in result.issues:
            prefix = {"error": "  ❌", "warning": "  ⚠️", "info": "  ℹ️"}.get(issue.severity, "  •")
            print(f"{prefix} [{issue.category}] {issue.description}")
    else:
        print("Issues: (none)")
    print()
