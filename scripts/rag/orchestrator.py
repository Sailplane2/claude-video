"""LangGraph orchestration: the RAG pipeline's ingest + retrieval flows as
explicit state graphs.

This module replaces the hand-written `for` loops in rebuild.py / sync.py with
LangGraph state graphs. Two graphs, deliberately separate — they have different
call patterns and different state shapes:

  - build_indexing_graph()  — batch ingest.  scan → [fan-out per video] → chunk+store.
  - build_retrieval_graph() — interactive. query → embed → retrieve.

Design rules honoured here:

  1. Node bodies REUSE the existing stage functions (scanner.scan,
     chunker.chunk_summary, embedder.embed_single, and the store's
     upsert_chunks / search_chunks) rather than reimplementing them.
  2. Backend-agnostic: every node resolves its store via utils.get_store(backend),
     so the same graphs drive Chroma or Neo4j with zero changes.
  3. `upsert_chunks` embeds internally (it calls embed_texts), so the
     "scan → chunk → embed → store" chain collapses the embed step INTO the
     store node. The retrieval graph is the mirror image: embedding is its own
     explicit node because there is no store call to fold it into.

The indexing graph is a map-reduce fan-out: `scan` produces a list of videos,
a conditional edge expands that list into one `Send` per video, and the reducer
(operator.add) accumulates each per-video summary back into a single report.
"""

from pathlib import Path
from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .scanner import scan
from .chunker import chunk_summary
from .embedder import embed_single
from .utils import get_store

# Single source of truth for the embed/chunker identifiers (shared with rebuild.py).
from .rebuild import EMBED_MODEL, CHUNKER_VERSION


# ── Indexing graph (batch) ─────────────────────────────────────────────────

class IndexingState(TypedDict):
    """State threaded through the ingest graph.

    `summaries` uses operator.add as its reducer, so every per-video node's
    single-element list is concatenated into the running aggregate — that is
    the LangGraph way to collect results across a fan-out.
    """
    backend: str                    # "chroma" | "neo4j"
    videos_dir: str | None          # optional scan override (isolated testing)
    videos: list                    # scanned VideoRecords, set by the scan node
    summaries: Annotated[list[dict], operator.add]   # per-video result dicts
    report: dict                    # aggregate totals, set by the aggregate node


def _scan_node(state: IndexingState) -> dict:
    """Node 1: scan the corpus. Outputs the list of VideoRecords to fan out over.

    If the caller already scanned (rebuild.py does, to report issues first),
    the pre-supplied `videos` list is used as-is instead of re-scanning.
    """
    if state.get("videos"):
        return {}
    override = Path(state["videos_dir"]) if state.get("videos_dir") else None
    result = scan(videos_dir_override=override)
    return {"videos": result.videos}


def _continue_to_videos(state: IndexingState) -> list[Send]:
    """Conditional edge: expand the scanned list into one Send per video.

    Each Send carries the single video plus the backend, so the per-video node
    is self-contained and runnable in parallel. (Send payloads are per-node
    inputs — they are not automatically merged with the parent state.)
    """
    return [
        Send("process_video", {"video": v, "backend": state["backend"]})
        for v in state.get("videos", [])
    ]


def _process_video_node(state: dict) -> dict:
    """Node 2 (one per video): chunk + embed + store a single video."""
    video = state["video"]
    store = get_store(state["backend"])

    if video.status in ("archived", "deleted"):
        return {"summaries": [{"video_id": video.video_id, "skipped": True}]}

    try:
        chunks = chunk_summary(video.body, video.frontmatter)
        # embed happens inside upsert_chunks — the "embed" stage is folded here.
        store.upsert_chunks(chunks, video, EMBED_MODEL, CHUNKER_VERSION)
        return {"summaries": [{"video_id": video.video_id, "chunks": len(chunks)}]}
    except Exception as e:  # keep one bad video from aborting the whole batch
        return {"summaries": [{"video_id": video.video_id, "error": str(e)}]}


def _aggregate_node(state: IndexingState) -> dict:
    """Node 3: join point. Roll the per-video summaries up into one report."""
    summaries = state.get("summaries", [])
    skipped = [s for s in summaries if s.get("skipped")]
    errors = [s for s in summaries if "error" in s]
    processed = [s for s in summaries if not s.get("skipped") and "error" not in s]
    report = {
        "videos_scanned": len(state.get("videos", [])),
        "videos_processed": len(processed),
        "videos_skipped": len(skipped),
        "videos_failed": len(errors),
        "total_chunks": sum(s.get("chunks", 0) for s in summaries),
    }
    return {"report": report}


def build_indexing_graph():
    """Compile the batch ingest graph: scan → fan-out(process_video) → aggregate."""
    g = StateGraph(IndexingState)
    g.add_node("scan", _scan_node)
    g.add_node("process_video", _process_video_node)
    g.add_node("aggregate", _aggregate_node)
    g.add_edge(START, "scan")
    g.add_conditional_edges("scan", _continue_to_videos, ["process_video"])
    g.add_edge("process_video", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# ── Retrieval graph (interactive) ──────────────────────────────────────────

class RetrievalState(TypedDict):
    query: str
    k: int
    backend: str
    embedding: list[float]          # set by the embed node
    raw: dict                       # store.search_chunks result, set by retrieve


def _embed_query_node(state: RetrievalState) -> dict:
    """Node 1: embed the user query into a 1536-dim vector."""
    return {"embedding": embed_single(state["query"])}


def _retrieve_node(state: RetrievalState) -> dict:
    """Node 2: vector-search the store and hand back its raw result shape."""
    store = get_store(state["backend"])
    return {"raw": store.search_chunks(state["embedding"], k=state["k"])}


def build_retrieval_graph():
    """Compile the interactive retrieval graph: query → embed → retrieve."""
    g = StateGraph(RetrievalState)
    g.add_node("embed_query", _embed_query_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_edge(START, "embed_query")
    g.add_edge("embed_query", "retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


# ── Convenience entry points ────────────────────────────────────────────────

def run_indexing(backend: str = "neo4j", videos_dir: str | None = None) -> dict:
    """Run a full ingest pass and return the aggregate report."""
    app = build_indexing_graph()
    return app.invoke({"backend": backend, "videos_dir": videos_dir})


def run_retrieval(query: str, k: int = 25, backend: str = "neo4j") -> dict:
    """Run a single query through the retrieval graph and return the raw store result."""
    app = build_retrieval_graph()
    return app.invoke({"query": query, "k": k, "backend": backend})
