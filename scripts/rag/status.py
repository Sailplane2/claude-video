"""Status reporter: print coverage report of current RAG state."""

from .scanner import scan
from .utils import get_store


def _render_issues(issues) -> None:
    if issues:
        print(f"Issues ({len(issues)}):")
        for issue in issues:
            prefix = {"error": "  ❌", "warning": "  ⚠️", "info": "  ℹ️"}.get(issue.severity, "  •")
            print(f"{prefix} [{issue.category}] {issue.description}")
    else:
        print("Issues: (none)")
    print()


def _run_chroma(store, result) -> None:
    indexed_ids = store.get_video_ids()
    chunk_count = store.get_chunk_count()
    indexed = [v for v in result.videos if v.video_id in indexed_ids]
    active = [v for v in result.videos if v.status == "active" and v.video_id not in indexed_ids]
    archived = [v for v in result.videos if v.status == "archived"]
    deleted = [v for v in result.videos if v.status == "deleted"]

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  Video Knowledge Base — Status Report    ║")
    print("╚══════════════════════════════════════════╝")
    print()
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
    _render_issues(result.issues)


def _run_neo4j(store, result) -> None:
    from .document_scanner import scan as scan_documents

    if not store.connection_ok():
        print()
        print("Neo4j: bolt://localhost:7687 — UNREACHABLE")
        print("  → is the Docker container running?")
        print("    docker compose -f ~/ai-dev/neo4j/docker-compose.yml up -d")
        print("  → falling back to grep (degraded mode)")
        print()
        return

    video_ids = store.get_video_ids()
    video_chunks = store.get_chunk_count()
    doc_count = store.get_document_count()
    doc_chunks = store.get_document_chunk_count()

    indexed = [v for v in result.videos if v.video_id in video_ids]
    active = [v for v in result.videos if v.status == "active" and v.video_id not in video_ids]
    archived = [v for v in result.videos if v.status == "archived"]
    deleted = [v for v in result.videos if v.status == "deleted"]

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  Video Knowledge Base — Status Report    ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("Index:            Neo4j bolt://localhost:7687")
    print(f"Video chunks:     {video_chunks}")
    print(f"Documents:        {doc_count} ({doc_chunks} chunks)")
    print("Embed model:      text-embedding-3-small")
    print("Chunker version:  v1.0")
    print()
    print("Videos:")
    print(f"  Total on disk:         {len(result.videos)}")
    print(f"  Indexed (in Neo4j):    {len(indexed)}")
    print(f"  Active (pending):      {len(active)}")
    print(f"  Archived:              {len(archived)}")
    print(f"  Deleted:               {len(deleted)}")
    print()

    only_in_db = video_ids - {v.video_id for v in result.videos}
    only_on_disk = {v.video_id for v in result.videos} - video_ids
    if only_in_db or only_on_disk:
        print("Filesystem / Neo4j consistency: MISMATCH")
        for vid in only_in_db:
            print(f"  Orphan in Neo4j (no file on disk): {vid}")
        for vid in only_on_disk:
            v = next((x for x in result.videos if x.video_id == vid), None)
            if v:
                print(f"  Not indexed (on disk): {vid} — {v.frontmatter.get('title', 'untitled')[:60]}")
    else:
        print("Filesystem / Neo4j consistency: OK")

    print()

    dresult = scan_documents()
    print("Documents:")
    print(f"  Total on disk:         {dresult.stats.get('total', 0)}")
    print(f"  Indexed (in Neo4j):    {doc_count}")
    print()

    _render_issues(result.issues + dresult.issues)


def run(backend: str = "neo4j"):
    store = get_store(backend)
    result = scan()
    if backend == "neo4j":
        _run_neo4j(store, result)
    else:
        _run_chroma(store, result)
