"""Validator: cross-check filesystem vs store for consistency."""

import sys

from .scanner import scan
from .utils import get_store


def run(backend: str = "neo4j") -> int:
    """Cross-validate filesystem vs store. Report inconsistencies.

    Returns 0 if consistent, 1 if issues found.
    """
    store = get_store(backend)
    result = scan()
    indexed_ids = store.get_video_ids()
    on_disk_ids = {v.video_id for v in result.videos}
    label = "Chroma" if backend == "chroma" else "Neo4j"

    issues = 0

    # Orphans in store
    for vid in sorted(indexed_ids - on_disk_ids):
        print(f"ORPHAN in {label}: {vid} (indexed but no file on disk)")
        issues += 1

    # Missing from store
    for v in result.videos:
        if v.status in ("active", "indexed") and v.video_id not in indexed_ids:
            print(f"MISSING from {label}: {v.video_id} — "
                  f"{v.frontmatter.get('title', 'untitled')[:60]} "
                  f"(on disk, status={v.status}, not indexed)")
            issues += 1

    # Wrongly indexed
    for v in result.videos:
        if v.status in ("deleted", "archived") and v.video_id in indexed_ids:
            print(f"STALE in {label}: {v.video_id} — "
                  f"{v.frontmatter.get('title', 'untitled')[:60]} "
                  f"(status={v.status} but still indexed)")
            issues += 1

    if issues == 0:
        print(f"Validate: OK — filesystem and {label} are consistent.")
        return 0
    else:
        print(f"\nValidate: {issues} issue(s) found. Run `rag prune` to clean up.")
        return 1
