"""Chroma vector store: collection management, upsert, delete, search, rebuild."""

import time
from pathlib import Path

from .utils import chroma_dir, date_to_epoch

# ── Module-level client singleton ──────────────────────────────────────────

_persistent_client = None


def _get_client():
    """Get or create the persistent Chroma client."""
    global _persistent_client
    if _persistent_client is None:
        import chromadb
        path = chroma_dir()
        path.mkdir(parents=True, exist_ok=True)
        _persistent_client = chromadb.PersistentClient(path=str(path))
    return _persistent_client


def _get_collection():
    """Get or create the 'videos' collection with cosine distance."""
    client = _get_client()
    try:
        return client.get_collection("videos")
    except Exception:
        return client.create_collection(
            name="videos",
            metadata={"hnsw:space": "cosine"},
        )


# ── Public API ─────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list, video_record, embed_model: str, chunker_version: str) -> None:
    """Embed and upsert all chunks for a single video.

    Args:
        chunks: list[Chunk] from chunker.py
        video_record: VideoRecord from scanner.py
        embed_model: embedding model identifier string
        chunker_version: chunker version string
    """
    from .embedder import embed as embed_texts

    collection = _get_collection()
    now_epoch = int(time.time())
    fm = video_record.frontmatter

    # Embed all chunks
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    # Build Chroma metadata
    tags_str = _build_tag_string(fm.get("tags", []))

    ids = []
    metadatas = []
    documents = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = f"{video_record.video_id}_{chunk.chunk_index}"
        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append({
            "doc_id": chunk_id,
            "video_id": video_record.video_id,
            "chunk_index": chunk.chunk_index,
            "content_hash": video_record.content_hash,
            "embed_model": embed_model,
            "chunker_version": chunker_version,
            "title": fm.get("title", video_record.video_id),
            "source": fm.get("source", ""),
            "channel": fm.get("channel", ""),
            "platform": fm.get("platform", ""),
            "video_type": fm.get("type", ""),
            "tags": tags_str,
            "status": "indexed",
            "rating": _safe_int(fm.get("rating"), -1),
            "date_epoch": date_to_epoch(fm.get("date", "")),
            "date_iso": fm.get("date", ""),
            "summary_path": str(video_record.summary_path),
            "section_title": chunk.section_title,
            "n_tokens": chunk.n_tokens,
            "n_bytes": chunk.n_bytes,
            "indexed_at_epoch": now_epoch,
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def delete_video(video_id: str) -> None:
    """Delete all chunks for a given video_id from Chroma."""
    collection = _get_collection()
    try:
        collection.delete(where={"video_id": video_id})
    except Exception:
        pass  # Collection might not exist or video might not be indexed


def rebuild_collection() -> None:
    """Drop and recreate the Chroma collection. Used by --rebuild."""
    client = _get_client()
    try:
        client.delete_collection("videos")
    except Exception:
        pass
    _get_collection()  # Recreates


def search_chunks(
    query_embedding: list[float],
    k: int = 25,
    where: dict | None = None,
) -> dict:
    """Search Chroma for chunks similar to query_embedding.

    Returns Chroma's raw query result dict with keys: ids, distances, metadatas, documents.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        where=where,
        include=["metadatas", "documents", "distances"],
    )


def get_video_ids() -> set[str]:
    """Get all distinct video_ids currently indexed in Chroma."""
    collection = _get_collection()
    if collection.count() == 0:
        return set()
    result = collection.get(include=["metadatas"])
    return set(m["video_id"] for m in result["metadatas"])


def get_chunk_count() -> int:
    """Get total number of chunks in the collection."""
    collection = _get_collection()
    return collection.count()


def get_collection_info() -> dict:
    """Get collection metadata for status reporting."""
    try:
        collection = _get_collection()
        return {
            "count": collection.count(),
            "video_ids": get_video_ids(),
            "name": collection.name,
        }
    except Exception:
        return {"count": 0, "video_ids": set(), "name": "videos"}


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_tag_string(tags) -> str:
    """Build sentinel-delimited tag string for Chroma metadata."""
    if isinstance(tags, list):
        return "|" + "|".join(str(t).strip().lower() for t in tags if t) + "|"
    if isinstance(tags, str):
        tags = tags.strip()
        if tags.startswith("|"):
            return tags
        return "|" + "|".join(t.strip().lower() for t in tags.split(",") if t.strip()) + "|"
    return "|"


def _safe_int(value, default=-1):
    """Safely convert a value to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
