"""Rebuild orchestrator: scan → chunk → embed → store. Full index rebuild from scratch."""

import sys
import time

from .scanner import scan
from .chunker import chunk_summary
from .utils import get_store


EMBED_MODEL = "text-embedding-3-small"
CHUNKER_VERSION = "v1.0"


def _print_issue(issue) -> None:
    if issue.severity == "error":
        print(f"  ERROR [{issue.category}] {issue.description}", file=sys.stderr)
    elif issue.severity == "warning":
        print(f"  WARN  [{issue.category}] {issue.description}", file=sys.stderr)


def _rebuild_videos(store, videos) -> tuple[int, int]:
    """Index videos through the LangGraph indexing graph (orchestrator.py).

    The graph fans out one process_video node per video; we pass the
    already-scanned list in so it doesn't re-scan. `store` is unused here —
    each graph node resolves its own store from the backend name.
    """
    from .orchestrator import build_indexing_graph
    backend = "neo4j" if store.__name__.endswith("neo4j_store") else "chroma"
    out = build_indexing_graph().invoke({"backend": backend, "videos": videos})
    by_id = {s["video_id"]: s for s in out.get("summaries", [])}
    total_chunks = errors = 0
    for i, video in enumerate(videos):
        s = by_id.get(video.video_id, {})
        tag = f"  [video {i + 1}/{len(videos)}]"
        if s.get("skipped"):
            print(f"{tag} SKIP {video.video_id} (status={video.status})")
        elif "error" in s:
            print(f"{tag} ERROR {video.video_id}: {s['error']}", file=sys.stderr)
            errors += 1
        else:
            total_chunks += s.get("chunks", 0)
            print(f"{tag} {video.video_id} → {s.get('chunks', 0)} chunks "
                  f"({video.frontmatter.get('title', 'untitled')[:60]})")
    return total_chunks, errors


def _rebuild_documents(store, documents) -> tuple[int, int]:
    from .document_chunker import chunk_document

    total_chunks = 0
    errors = 0
    for i, doc in enumerate(documents):
        if doc.status in ("archived", "deleted"):
            print(f"  [doc {i + 1}/{len(documents)}] SKIP {doc.document_id} (status={doc.status})")
            continue
        try:
            chunks = chunk_document(doc.body, doc.frontmatter)
            store.upsert_document_chunks(chunks, doc, EMBED_MODEL, CHUNKER_VERSION)
            total_chunks += len(chunks)
            print(f"  [doc {i + 1}/{len(documents)}] {doc.document_id} → {len(chunks)} chunks "
                  f"({doc.frontmatter.get('title', 'untitled')[:60]})")
        except Exception as e:
            print(f"  [doc {i + 1}/{len(documents)}] ERROR {doc.document_id}: {e}", file=sys.stderr)
            errors += 1
    return total_chunks, errors


def run(force: bool = False, backend: str = "neo4j") -> int:
    store = get_store(backend)

    if not force:
        scope = "videos + documents" if backend == "neo4j" else "videos"
        print(f"This will drop and rebuild the entire RAG index ({scope}).")
        print("All content will be re-chunked and re-embedded (OpenAI API costs apply).")
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

    documents = []
    doc_issues = []
    if backend == "neo4j":
        from .document_scanner import scan as scan_documents
        dresult = scan_documents()
        documents = dresult.documents
        doc_issues = dresult.issues
        print(f"Found {len(result.videos)} videos and {len(documents)} documents "
              f"({len(result.issues) + len(doc_issues)} issues).")
    else:
        print(f"Found {len(result.videos)} videos ({len(result.issues)} issues).")

    for issue in result.issues + doc_issues:
        _print_issue(issue)

    # Drop and recreate
    if backend == "neo4j":
        print("Initializing Neo4j schema and clearing video/document subgraphs...")
        store.init_schema()
        store.rebuild_collection()
        store.rebuild_documents()
    else:
        print("Rebuilding Chroma collection...")
        store.rebuild_collection()

    if not result.videos and not documents:
        print("No content found. Index left empty.")
        return 0

    start_time = time.time()
    total_chunks = 0
    doc_chunks = 0
    errors = 0

    if result.videos:
        vc, ve = _rebuild_videos(store, result.videos)
        total_chunks += vc
        errors += ve

    if backend == "neo4j" and documents:
        dc, de = _rebuild_documents(store, documents)
        doc_chunks += dc
        errors += de

    elapsed = time.time() - start_time
    print()
    if backend == "neo4j":
        print(f"Rebuild complete: {len(result.videos)} videos → {total_chunks} chunks; "
              f"{len(documents)} documents → {doc_chunks} chunks in {elapsed:.1f}s")
    else:
        print(f"Rebuild complete: {len(result.videos)} videos → {total_chunks} chunks in {elapsed:.1f}s")
    if errors:
        print(f"  {errors} error(s) — see above", file=sys.stderr)
        return 1
    return 0
