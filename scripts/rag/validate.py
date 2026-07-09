"""Validator: cross-check filesystem vs Chroma for consistency."""

import sys

from .scanner import scan
from .chroma_store import get_video_ids


def run() -> int:
    """Cross-validate filesystem vs Chroma. Report inconsistencies.

    Returns 0 if consistent, 1 if issues found.
    """
    result = scan()
    indexed_ids = get_video_ids()
    on_disk_ids = {v.video_id for v in result.videos}

    issues = 0

    # Orphans in Chroma
    for vid in sorted(indexed_ids - on_disk_ids):
        print(f"ORPHAN in Chroma: {vid} (indexed but no file on disk)")
        issues += 1

    # Missing from Chroma
    for v in result.videos:
        if v.status in ("active", "indexed") and v.video_id not in indexed_ids:
            print(f"MISSING from Chroma: {v.video_id} — "
                  f"{v.frontmatter.get('title', 'untitled')[:60]} "
                  f"(on disk, status={v.status}, not indexed)")
            issues += 1

    # Wrongly indexed
    for v in result.videos:
        if v.status in ("deleted", "archived") and v.video_id in indexed_ids:
            print(f"STALE in Chroma: {v.video_id} — "
                  f"{v.frontmatter.get('title', 'untitled')[:60]} "
                  f"(status={v.status} but still indexed)")
            issues += 1

    if issues == 0:
        print("Validate: OK — filesystem and Chroma are consistent.")
        return 0
    else:
        print(f"\nValidate: {issues} issue(s) found. Run `rag prune` to clean up.")
        return 1
