"""Sync orchestrator: incremental update — new, stale, deleted."""

import sys

from .scanner import scan
from .chunker import chunk_summary
from .utils import get_store
from .rebuild import EMBED_MODEL, CHUNKER_VERSION


def _sync_videos(store, videos, indexed_ids) -> tuple[int, int, int, int]:
    new_count = updated_count = pruned_count = errors = 0
    for video in videos:
        status = video.status
        if status in ("deleted", "archived"):
            if video.video_id in indexed_ids:
                try:
                    store.delete_video(video.video_id)
                    pruned_count += 1
                    print(f"  PRUNE {video.video_id} (status={status})")
                except Exception as e:
                    print(f"  PRUNE ERROR {video.video_id}: {e}", file=sys.stderr)
                    errors += 1
            continue
        if status not in ("active", "indexed"):
            continue
        if video.video_id not in indexed_ids:
            try:
                chunks = chunk_summary(video.body, video.frontmatter)
                store.upsert_chunks(chunks, video, EMBED_MODEL, CHUNKER_VERSION)
                new_count += 1
                print(f"  NEW  {video.video_id} → {len(chunks)} chunks "
                      f"({video.frontmatter.get('title', 'untitled')[:60]})")
            except Exception as e:
                print(f"  NEW  ERROR {video.video_id}: {e}", file=sys.stderr)
                errors += 1
            continue
        if status == "active":
            try:
                store.delete_video(video.video_id)
                chunks = chunk_summary(video.body, video.frontmatter)
                store.upsert_chunks(chunks, video, EMBED_MODEL, CHUNKER_VERSION)
                updated_count += 1
            except Exception as e:
                print(f"  UPDATE ERROR {video.video_id}: {e}", file=sys.stderr)
                errors += 1
    return new_count, updated_count, pruned_count, errors


def _sync_documents(store, documents, indexed_ids) -> tuple[int, int, int, int]:
    from .document_chunker import chunk_document

    new_count = updated_count = pruned_count = errors = 0
    for doc in documents:
        status = doc.status
        if status in ("deleted", "archived"):
            if doc.document_id in indexed_ids:
                try:
                    store.delete_document(doc.document_id)
                    pruned_count += 1
                    print(f"  PRUNE {doc.document_id} (status={status})")
                except Exception as e:
                    print(f"  PRUNE ERROR {doc.document_id}: {e}", file=sys.stderr)
                    errors += 1
            continue
        if status not in ("active", "indexed"):
            continue
        if doc.document_id not in indexed_ids:
            try:
                chunks = chunk_document(doc.body, doc.frontmatter)
                store.upsert_document_chunks(chunks, doc, EMBED_MODEL, CHUNKER_VERSION)
                new_count += 1
                print(f"  NEW  {doc.document_id} → {len(chunks)} chunks "
                      f"({doc.frontmatter.get('title', 'untitled')[:60]})")
            except Exception as e:
                print(f"  NEW  ERROR {doc.document_id}: {e}", file=sys.stderr)
                errors += 1
            continue
        if status == "active":
            try:
                store.delete_document(doc.document_id)
                chunks = chunk_document(doc.body, doc.frontmatter)
                store.upsert_document_chunks(chunks, doc, EMBED_MODEL, CHUNKER_VERSION)
                updated_count += 1
            except Exception as e:
                print(f"  UPDATE ERROR {doc.document_id}: {e}", file=sys.stderr)
                errors += 1
    return new_count, updated_count, pruned_count, errors


def run(backend: str = "neo4j") -> int:
    store = get_store(backend)

    result = scan()
    indexed_ids = store.get_video_ids()
    chunk_count_before = store.get_chunk_count()

    new_count = updated_count = pruned_count = errors = 0

    if result.videos:
        nc, uc, pc, er = _sync_videos(store, result.videos, indexed_ids)
        new_count += nc
        updated_count += uc
        pruned_count += pc
        errors += er

    doc_chunk_before = 0
    if backend == "neo4j":
        from .document_scanner import scan as scan_documents
        dresult = scan_documents()
        doc_ids = store.get_document_ids()
        doc_chunk_before = store.get_document_chunk_count()
        if dresult.documents:
            nc, uc, pc, er = _sync_documents(store, dresult.documents, doc_ids)
            new_count += nc
            updated_count += uc
            pruned_count += pc
            errors += er

    chunk_count_after = store.get_chunk_count()
    print()
    if backend == "neo4j":
        doc_chunk_after = store.get_document_chunk_count()
        print(f"Sync complete: {new_count} new, {updated_count} updated, "
              f"{pruned_count} pruned (video chunks {chunk_count_before} → {chunk_count_after}, "
              f"doc chunks {doc_chunk_before} → {doc_chunk_after})")
    else:
        print(f"Sync complete: {new_count} new, {updated_count} updated, "
              f"{pruned_count} pruned ({chunk_count_before} → {chunk_count_after} chunks)")

    return 1 if errors else 0
