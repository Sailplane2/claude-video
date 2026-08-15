"""Pruner: remove deleted/archived video chunks from the store."""

import sys

from .scanner import scan
from .utils import get_store


def run(backend: str = "neo4j") -> int:
    """Remove all store entries for videos with status=deleted or status=archived.

    Returns 0 on success.
    """
    store = get_store(backend)
    result = scan()
    indexed_ids = store.get_video_ids()

    pruned = 0
    for video in result.videos:
        if video.status in ("deleted", "archived") and video.video_id in indexed_ids:
            try:
                store.delete_video(video.video_id)
                pruned += 1
                print(f"  Pruned: {video.video_id} ({video.status})")
            except Exception as e:
                print(f"  Error pruning {video.video_id}: {e}", file=sys.stderr)

    # Also prune orphan entries in store (no file on disk)
    on_disk_ids = {v.video_id for v in result.videos}
    for vid in indexed_ids - on_disk_ids:
        try:
            store.delete_video(vid)
            pruned += 1
            print(f"  Pruned orphan: {vid} (no file on disk)")
        except Exception as e:
            print(f"  Error pruning orphan {vid}: {e}", file=sys.stderr)

    print(f"\nPrune complete: {pruned} video(s) removed from index.")
    return 0
