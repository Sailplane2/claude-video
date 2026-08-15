"""Neo4j vector store: mirrors chroma_store's video API and adds document support.

Neo4j is the long-term home for Michael's Brain knowledge graph. This module
replaces Chroma for storage while keeping the exact same video-facing surface
so rebuild/sync/search/status/validate/prune work unchanged behind a
`--backend neo4j` flag.

Denormalization note: like Chroma, the search-relevant metadata (title,
channel/author, tags, dates, etc.) is stored directly on each :Chunk node so
the vector query returns a self-contained result in the same shape Chroma
returns. The typed :Video/:Document nodes hold the canonical copy, and the
HAS_CHUNK relationship is the long-term path for entity-link / graph queries.
"""

import time
from pathlib import Path

from .utils import date_to_epoch

# ── Module-level driver singleton ───────────────────────────────────────────

_driver = None

_URI = "bolt://localhost:7687"


def _load_neo4j_auth() -> tuple[str, str]:
    """Read NEO4J_AUTH from ~/.config/michaels-brain/neo4j.env."""
    env_file = Path.home() / ".config" / "michaels-brain" / "neo4j.env"
    user, password = "neo4j", ""
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("NEO4J_AUTH="):
                auth = line.split("=", 1)[1].strip().strip('"').strip("'")
                if "/" in auth:
                    user, _, password = auth.partition("/")
                break
    return user, password


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        user, password = _load_neo4j_auth()
        _driver = GraphDatabase.driver(_URI, auth=(user, password))
    return _driver


def connection_ok() -> bool:
    """Return True if the Neo4j server is reachable and authenticated."""
    try:
        _get_driver().verify_connectivity()
        return True
    except Exception:
        return False


def init_schema() -> None:
    """Create constraints + vector index (idempotent via IF NOT EXISTS)."""
    statements = [
        "CREATE CONSTRAINT video_id_unique IF NOT EXISTS "
        "FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE VECTOR INDEX chunk_embeddings_3small IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 1536, "
        "`vector.similarity_function`: 'cosine'}}",
    ]
    driver = _get_driver()
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)


# ── Helpers ────────────────────────────────────────────────────────────────

def _as_list(tags) -> list[str]:
    """Normalize tags (list, sentinel string, or comma string) to a list."""
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    if isinstance(tags, str):
        s = tags.strip()
        if s.startswith("|"):
            return [t.strip() for t in s.strip("|").split("|") if t.strip()]
        # Handle the scanner's bracket-wrapped form: "[a, b, c]" or "['a', 'b']"
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [t.strip().strip("'\"").strip() for t in s.split(",") if t.strip().strip("'\"")]
    return []


def _tags_to_sentinel(tags) -> str:
    """Convert tags (list) to the sentinel-delimited string search.py expects."""
    return "|" + "|".join(_as_list(tags)) + "|"


def _safe_int(value, default=-1):
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _empty_result() -> dict:
    return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}


# ── Video: public API (mirrors chroma_store exactly) ───────────────────────

_VIDEO_UPSERT = """
UNWIND $rows AS row
MERGE (v:Video {video_id: row.video_id})
SET v.title = row.title, v.channel = row.channel, v.source = row.source,
    v.platform = row.platform, v.type = row.video_type, v.tags = row.tags,
    v.date_iso = row.date_iso, v.date_epoch = row.date_epoch,
    v.rating = row.rating, v.status = row.status,
    v.summary_path = row.summary_path, v.content_hash = row.content_hash,
    v.domain = 'video-kb'
WITH v, row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.section_title = row.section_title, c.chunk_index = row.chunk_index,
    c.text = row.text, c.embedding = row.embedding,
    c.n_tokens = row.n_tokens, c.n_bytes = row.n_bytes,
    c.embed_model = row.embed_model, c.chunker_version = row.chunker_version,
    c.indexed_at_epoch = row.indexed_at_epoch,
    c.source_type = 'video', c.video_id = row.video_id,
    c.title = row.title, c.channel = row.channel, c.source = row.source,
    c.tags = row.tags, c.date_iso = row.date_iso, c.date_epoch = row.date_epoch,
    c.rating = row.rating, c.summary_path = row.summary_path,
    c.content_hash = row.content_hash, c.platform = row.platform,
    c.video_type = row.video_type
MERGE (v)-[:HAS_CHUNK]->(c)
"""


def upsert_chunks(chunks: list, video_record, embed_model: str, chunker_version: str) -> None:
    """Embed and upsert all chunks for a single video."""
    from .embedder import embed as embed_texts

    fm = video_record.frontmatter
    tags = _as_list(fm.get("tags", []))
    title = fm.get("title", video_record.video_id)
    now_epoch = int(time.time())
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "video_id": video_record.video_id,
            "title": title,
            "channel": fm.get("channel", ""),
            "source": fm.get("source", ""),
            "platform": fm.get("platform", ""),
            "video_type": fm.get("type", ""),
            "tags": tags,
            "date_iso": fm.get("date", ""),
            "date_epoch": date_to_epoch(fm.get("date", "")),
            "rating": _safe_int(fm.get("rating"), -1),
            "status": video_record.status,
            "summary_path": str(video_record.summary_path),
            "content_hash": video_record.content_hash,
            "chunk_id": f"{video_record.video_id}_{chunk.chunk_index}",
            "chunk_index": chunk.chunk_index,
            "section_title": chunk.section_title,
            "text": chunk.text,
            "embedding": embedding,
            "n_tokens": chunk.n_tokens,
            "n_bytes": chunk.n_bytes,
            "embed_model": embed_model,
            "chunker_version": chunker_version,
            "indexed_at_epoch": now_epoch,
        })

    if rows:
        with _get_driver().session() as session:
            session.run(_VIDEO_UPSERT, rows=rows)


def delete_video(video_id: str) -> None:
    """Delete a video's node and all its chunks (scoped)."""
    try:
        with _get_driver().session() as session:
            session.run(
                "MATCH (c:Chunk) WHERE c.source_type = 'video' AND c.video_id = $id DETACH DELETE c",
                id=video_id,
            )
            session.run("MATCH (v:Video {video_id: $id}) DETACH DELETE v", id=video_id)
    except Exception:
        pass


def rebuild_collection() -> None:
    """Drop all Video + video-Chunk nodes (scoped — never touches other labels)."""
    try:
        with _get_driver().session() as session:
            session.run("MATCH (c:Chunk) WHERE c.source_type = 'video' DETACH DELETE c")
            session.run("MATCH (v:Video) DETACH DELETE v")
    except Exception:
        pass


def search_chunks(query_embedding: list[float], k: int = 25, where: dict | None = None) -> dict:
    """Vector search across video + document chunks. Returns chroma's raw shape.

    `where` is accepted for API parity but ignored — search.py applies its own
    in-memory filters afterward. Neo4j returns cosine `score` (higher = closer);
    we convert to `distance = 1 - score` so search.py's existing
    distance→similarity conversion is preserved.
    """
    driver = _get_driver()
    ids, metadatas, documents, distances = [], [], [], []
    with driver.session() as session:
        result = session.run(
            "CALL db.index.vector.queryNodes('chunk_embeddings_3small', $k, $embedding) "
            "YIELD node AS c, score "
            "RETURN c.chunk_id AS id, c.text AS document, properties(c) AS metadata, score AS score",
            k=k,
            embedding=query_embedding,
        )
        for rec in result:
            ids.append(rec["id"])
            documents.append(rec["document"])
            meta = dict(rec["metadata"])
            meta["tags"] = _tags_to_sentinel(meta.get("tags"))
            metadatas.append(meta)
            distances.append(1.0 - float(rec["score"]))

    if not ids:
        return _empty_result()
    return {"ids": [ids], "distances": [distances], "metadatas": [metadatas], "documents": [documents]}


def get_video_ids() -> set[str]:
    try:
        with _get_driver().session() as session:
            result = session.run("MATCH (v:Video) RETURN v.video_id AS id")
            return {rec["id"] for rec in result}
    except Exception:
        return set()


def get_chunk_count() -> int:
    """Number of video chunks (mirrors chroma_store's meaning)."""
    try:
        with _get_driver().session() as session:
            result = session.run(
                "MATCH (c:Chunk) WHERE c.source_type = 'video' RETURN count(c) AS n"
            )
            return result.single()["n"]
    except Exception:
        return 0


def get_collection_info() -> dict:
    try:
        return {
            "count": get_chunk_count(),
            "video_ids": get_video_ids(),
            "name": "neo4j",
        }
    except Exception:
        return {"count": 0, "video_ids": set(), "name": "neo4j"}


# ── Document: public API (additional) ───────────────────────────────────────

_DOCUMENT_UPSERT = """
UNWIND $rows AS row
MERGE (d:Document {document_id: row.document_id})
SET d.title = row.title, d.author = row.author, d.source_path = row.source_path,
    d.summary_path = row.summary_path, d.page_count = row.page_count,
    d.tags = row.tags, d.date_iso = row.date_iso, d.date_epoch = row.date_epoch,
    d.status = row.status, d.content_hash = row.content_hash, d.domain = 'documents'
WITH d, row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.section_title = row.section_title, c.chunk_index = row.chunk_index,
    c.text = row.text, c.embedding = row.embedding,
    c.n_tokens = row.n_tokens, c.n_bytes = row.n_bytes,
    c.embed_model = row.embed_model, c.chunker_version = row.chunker_version,
    c.indexed_at_epoch = row.indexed_at_epoch,
    c.source_type = 'document', c.document_id = row.document_id,
    c.title = row.title, c.author = row.author, c.source = row.source_path,
    c.tags = row.tags, c.date_iso = row.date_iso, c.date_epoch = row.date_epoch,
    c.rating = row.rating, c.summary_path = row.summary_path,
    c.content_hash = row.content_hash, c.page_count = row.page_count
MERGE (d)-[:HAS_CHUNK]->(c)
"""


def upsert_document_chunks(chunks: list, document_record, embed_model: str, chunker_version: str) -> None:
    """Embed and upsert all chunks for a single document."""
    from .embedder import embed as embed_texts

    fm = document_record.frontmatter
    tags = _as_list(fm.get("tags", []))
    title = fm.get("title", document_record.document_id)
    now_epoch = int(time.time())
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "document_id": document_record.document_id,
            "title": title,
            "author": fm.get("author", ""),
            "source_path": fm.get("source", ""),
            "summary_path": str(document_record.summary_path),
            "page_count": _safe_int(fm.get("page_count"), 0),
            "tags": tags,
            "date_iso": fm.get("date", ""),
            "date_epoch": date_to_epoch(fm.get("date", "")),
            "rating": _safe_int(fm.get("rating"), -1),
            "status": document_record.status,
            "content_hash": document_record.content_hash,
            "chunk_id": f"{document_record.document_id}_{chunk.chunk_index}",
            "chunk_index": chunk.chunk_index,
            "section_title": chunk.section_title,
            "text": chunk.text,
            "embedding": embedding,
            "n_tokens": chunk.n_tokens,
            "n_bytes": chunk.n_bytes,
            "embed_model": embed_model,
            "chunker_version": chunker_version,
            "indexed_at_epoch": now_epoch,
        })

    if rows:
        with _get_driver().session() as session:
            session.run(_DOCUMENT_UPSERT, rows=rows)


def delete_document(document_id: str) -> None:
    """Delete a document's node and all its chunks (scoped)."""
    try:
        with _get_driver().session() as session:
            session.run(
                "MATCH (c:Chunk) WHERE c.source_type = 'document' AND c.document_id = $id DETACH DELETE c",
                id=document_id,
            )
            session.run("MATCH (d:Document {document_id: $id}) DETACH DELETE d", id=document_id)
    except Exception:
        pass


def rebuild_documents() -> None:
    """Drop all Document + document-Chunk nodes (scoped)."""
    try:
        with _get_driver().session() as session:
            session.run("MATCH (c:Chunk) WHERE c.source_type = 'document' DETACH DELETE c")
            session.run("MATCH (d:Document) DETACH DELETE d")
    except Exception:
        pass


def get_document_ids() -> set[str]:
    try:
        with _get_driver().session() as session:
            result = session.run("MATCH (d:Document) RETURN d.document_id AS id")
            return {rec["id"] for rec in result}
    except Exception:
        return set()


def get_document_count() -> int:
    try:
        with _get_driver().session() as session:
            result = session.run("MATCH (d:Document) RETURN count(d) AS n")
            return result.single()["n"]
    except Exception:
        return 0


def get_document_chunk_count() -> int:
    try:
        with _get_driver().session() as session:
            result = session.run(
                "MATCH (c:Chunk) WHERE c.source_type = 'document' RETURN count(c) AS n"
            )
            return result.single()["n"]
    except Exception:
        return 0
