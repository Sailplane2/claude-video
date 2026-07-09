"""Pruner: remove deleted/archived video chunks from Chroma."""

import sys

from .scanner import scan
from .chroma_store import delete_video, get_video_ids


def run() -> int:
    """Remove all Chroma entries for videos with status=deleted or status=archived.

    Returns 0 on success.
    """
    result = scan()
    indexed_ids = get_video_ids()

    pruned = 0
    for video in result.videos:
        if video.status in ("deleted", "archived") and video.video_id in indexed_ids:
            try:
                delete_video(video.video_id)
                pruned += 1
                print(f"  Pruned: {video.video_id} ({video.status})")
            except Exception as e:
                print(f"  Error pruning {video.video_id}: {e}", file=sys.stderr)

    # Also prune orphan entries in Chroma (no file on disk)
    on_disk_ids = {v.video_id for v in result.videos}
    for vid in indexed_ids - on_disk_ids:
        try:
            delete_video(vid)
            pruned += 1
            print(f"  Pruned orphan: {vid} (no file on disk)")
        except Exception as e:
            print(f"  Error pruning orphan {vid}: {e}", file=sys.stderr)

    print(f"\nPrune complete: {pruned} video(s) removed from index.")
    return 0
