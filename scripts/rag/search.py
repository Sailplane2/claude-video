"""Search core: embed query, search store, group/filter, format output."""

import json
import sys
import time
from dataclasses import dataclass, field

from . import embedder
from .scanner import scan
from .utils import date_to_epoch, get_store


@dataclass
class ChunkMatch:
    video_id: str
    chunk_index: str
    section_title: str
    title: str
    channel: str
    source: str
    tags: str
    date_iso: str
    date_epoch: int
    rating: int
    summary_path: str
    text: str
    score: float
    source_type: str = "video"
    document_id: str = ""
    author: str = ""


@dataclass
class VideoMatch:
    video_id: str
    title: str
    channel: str
    source: str
    tags: str
    date_iso: str
    rating: int
    summary_path: str
    best_score: float
    chunk_count: int
    matched_chunks: list[ChunkMatch] = field(default_factory=list)
    source_type: str = "video"
    author: str = ""


@dataclass
class SearchResult:
    query: str
    mode: str
    results: list  # list[VideoMatch] or list[ChunkMatch]
    total_videos: int
    matched_count: int
    elapsed_ms: int
    fallback_used: bool = False


def _parse_raw_results(raw: dict) -> list[ChunkMatch]:
    """Parse the store's raw query response into ChunkMatch objects."""
    matches = []
    ids_list = raw.get("ids", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    documents = raw.get("documents", [[]])[0]

    for i, chunk_id in enumerate(ids_list):
        meta = metadatas[i] if i < len(metadatas) else {}
        doc = documents[i] if i < len(documents) else ""
        dist = distances[i] if i < len(distances) else 0.0
        # Cosine distance → similarity (1.0 is identical, 0.0 is orthogonal)
        similarity = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)

        matches.append(ChunkMatch(
            video_id=meta.get("video_id", ""),
            chunk_index=meta.get("chunk_index", ""),
            section_title=meta.get("section_title", ""),
            title=meta.get("title", ""),
            channel=meta.get("channel", ""),
            source=meta.get("source", ""),
            tags=meta.get("tags", ""),
            date_iso=meta.get("date_iso", ""),
            date_epoch=meta.get("date_epoch", 0),
            rating=meta.get("rating", -1),
            summary_path=meta.get("summary_path", ""),
            text=doc,
            score=similarity,
            source_type=meta.get("source_type", "video"),
            document_id=meta.get("document_id", ""),
            author=meta.get("author", ""),
        ))

    return matches


def _group_by_video(chunks: list[ChunkMatch], k: int) -> list[VideoMatch]:
    """Group chunk matches by source id (video_id or document_id), ranked by best score."""
    groups: dict[str, VideoMatch] = {}

    for c in chunks:
        gid = c.video_id if c.video_id else c.document_id
        if gid not in groups:
            groups[gid] = VideoMatch(
                video_id=gid,
                title=c.title,
                channel=c.channel,
                source=c.source,
                tags=c.tags,
                date_iso=c.date_iso,
                rating=c.rating,
                summary_path=c.summary_path,
                best_score=c.score,
                chunk_count=0,
                source_type=c.source_type,
                author=c.author,
            )
        group = groups[gid]
        group.chunk_count += 1
        group.best_score = max(group.best_score, c.score)
        group.matched_chunks.append(c)

    ranked = sorted(groups.values(), key=lambda g: g.best_score, reverse=True)
    return ranked[:k]


def _apply_filters(chunks: list[ChunkMatch], tags=None, channel=None, platform=None,
                   video_type=None, after_date=None, before_date=None, min_rating=None) -> list[ChunkMatch]:
    """Apply metadata filters in memory."""
    filtered = chunks

    if tags:
        for tag in tags:
            tag_lower = tag.strip().lower()
            filtered = [c for c in filtered if f"|{tag_lower}|" in c.tags.lower() or tag_lower in c.tags.lower()]

    if channel:
        ch_lower = channel.lower()
        filtered = [c for c in filtered if ch_lower in c.channel.lower()]

    if platform:
        pl_lower = platform.lower()
        filtered = [c for c in filtered if pl_lower in c.channel.lower() or pl_lower in c.title.lower()]

    if video_type:
        vt = video_type.lower()
        filtered = [c for c in filtered if vt in c.tags.lower()]

    if after_date:
        after_epoch = date_to_epoch(after_date)
        if after_epoch:
            filtered = [c for c in filtered if c.date_epoch >= after_epoch]

    if before_date:
        before_epoch = date_to_epoch(before_date)
        if before_epoch:
            filtered = [c for c in filtered if c.date_epoch <= before_epoch]

    if min_rating is not None:
        try:
            mr = int(min_rating)
            filtered = [c for c in filtered if c.rating >= mr]
        except (ValueError, TypeError):
            pass

    return filtered


def _get_coverage(store) -> dict:
    """Get coverage stats for status line."""
    result = scan()
    indexed = store.get_video_ids()
    return {
        "total": len(result.videos),
        "indexed": len(indexed),
    }


def _make_obsidian_link(path_str: str) -> str:
    """Create an Obsidian deep link from a file path."""
    if not path_str:
        return ""
    vault = "michael-brain"
    try:
        from pathlib import Path
        from .utils import vault_root
        p = Path(path_str)
        rel = p.relative_to(vault_root())
        encoded = str(rel).replace(" ", "%20")
        return f"obsidian://open?vault={vault}&file={encoded}"
    except (ValueError, OSError):
        filename = Path(path_str).name
        return f"obsidian://open?vault={vault}&file={filename}"


def _search_with_grep(query: str, tags=None, channel=None, video_type=None, store=None) -> SearchResult:
    """Fallback: grep-based search through all summary bodies (video-only)."""
    result = scan()
    start = time.time()

    terms = query.lower().split()
    matches: list[tuple] = []

    for video in result.videos:
        fm = video.frontmatter
        if channel and channel.lower() not in fm.get("channel", "").lower():
            continue
        if tags:
            video_tags = fm.get("tags", [])
            if isinstance(video_tags, str):
                video_tags = [t.strip() for t in video_tags.split(",")]
            video_tags_lower = [t.lower().strip() for t in video_tags]
            if not all(t.lower().strip() in video_tags_lower for t in tags):
                continue

        body_lower = video.body.lower()
        match_count = sum(1 for t in terms if t in body_lower)
        if match_count > 0:
            matches.append((video, match_count / len(terms)))

    matches.sort(key=lambda x: x[1], reverse=True)

    chunk_matches = []
    for video, score in matches[:25]:
        fm = video.frontmatter
        chunk_matches.append(ChunkMatch(
            video_id=video.video_id,
            chunk_index="grep_fallback",
            section_title="full",
            title=fm.get("title", video.video_id),
            channel=fm.get("channel", ""),
            source=fm.get("source", ""),
            tags="|" + "|".join(fm.get("tags", [])) + "|" if isinstance(fm.get("tags"), list) else fm.get("tags", ""),
            date_iso=fm.get("date", ""),
            date_epoch=date_to_epoch(fm.get("date", "")),
            rating=fm.get("rating", -1) if fm.get("rating") else -1,
            summary_path=str(video.summary_path),
            text=video.body[:500],
            score=score,
        ))

    elapsed = int((time.time() - start) * 1000)
    coverage = _get_coverage(store) if store is not None else {"total": len(result.videos), "indexed": 0}

    return SearchResult(
        query=query,
        mode="videos",
        results=_group_by_video(chunk_matches, 25),
        total_videos=coverage["total"],
        matched_count=len(chunk_matches),
        elapsed_ms=elapsed,
        fallback_used=True,
    )


# ── Public API ─────────────────────────────────────────────────────────────

def search(query: str, mode: str = "videos", k: int = 25,
           tags: list[str] | None = None, channel: str | None = None,
           platform: str | None = None, video_type: str | None = None,
           after_date: str | None = None, before_date: str | None = None,
           min_rating: int | None = None, backend: str = "neo4j",
           kind: str = "all") -> SearchResult:
    """Core search function — the API that programmatic callers use.

    Args:
        query: Natural language search query.
        mode: "videos" (grouped) or "passages" (individual chunks).
        k: Number of results to return.
        tags: Filter to content with ALL these tags.
        channel: Filter by channel name (case-insensitive substring).
        platform: Filter by platform.
        video_type: Filter by video type.
        after_date: YYYY-MM-DD lower bound.
        before_date: YYYY-MM-DD upper bound.
        min_rating: Minimum rating (1-5).
        backend: "chroma" or "neo4j".
        kind: "video", "document", or "all".

    Returns:
        SearchResult with results, coverage info, and timing.
    """
    store = get_store(backend)
    start = time.time()

    # Embed + vector search via the LangGraph retrieval graph (orchestrator.py)
    from .orchestrator import run_retrieval
    try:
        raw = run_retrieval(query, k=k * 2, backend=backend)["raw"]
    except Exception as e:
        print(f"Vector retrieval failed ({e}). Falling back to grep.", file=sys.stderr)
        return _search_with_grep(query, tags, channel, video_type, store)

    raw_chunks = _parse_raw_results(raw)

    # Filter by content type
    if kind != "all":
        raw_chunks = [c for c in raw_chunks if c.source_type == kind]

    # Fall back to grep if the index returns nothing (empty index)
    if not raw_chunks:
        return _search_with_grep(query, tags, channel, video_type, store)

    filtered = _apply_filters(raw_chunks, tags, channel, platform, video_type,
                              after_date, before_date, min_rating)

    if mode == "videos":
        results = _group_by_video(filtered, k)
        matched_count = len({c.video_id or c.document_id for c in filtered})
    else:
        results = filtered[:k]
        matched_count = len({c.video_id or c.document_id for c in filtered})

    elapsed = int((time.time() - start) * 1000)
    coverage = _get_coverage(store)

    return SearchResult(
        query=query,
        mode=mode,
        results=results,
        total_videos=coverage["total"],
        matched_count=matched_count,
        elapsed_ms=elapsed,
    )


# ── CLI renderer ───────────────────────────────────────────────────────────

def run(query: str, mode: str = "videos", k: int = 25,
        tags: str | None = None, channel: str | None = None,
        platform: str | None = None, video_type: str | None = None,
        after_date: str | None = None, before_date: str | None = None,
        min_rating: int | None = None, json_output: bool = False,
        no_fallback: bool = False, backend: str = "neo4j",
        kind: str = "all"):
    """CLI entry point for search. Formats and prints results."""

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    result = search(
        query=query, mode=mode, k=k,
        tags=tag_list, channel=channel, platform=platform,
        video_type=video_type, after_date=after_date, before_date=before_date,
        min_rating=min_rating, backend=backend, kind=kind,
    )

    if json_output:
        output = _to_json(result)
        print(json.dumps(output, indent=2))
        return

    _render_markdown(result)


def _to_json(result: SearchResult) -> dict:
    """Convert SearchResult to JSON-serializable dict."""
    items = []
    for r in result.results:
        if isinstance(r, VideoMatch):
            items.append({
                "type": r.source_type,
                "video_id": r.video_id,
                "title": r.title,
                "channel": r.channel,
                "author": r.author,
                "source": r.source,
                "tags": r.tags,
                "date": r.date_iso,
                "rating": r.rating,
                "best_score": round(r.best_score, 4),
                "chunk_count": r.chunk_count,
                "obsidian_link": _make_obsidian_link(r.summary_path),
                "snippets": [mc.text[:200] for mc in r.matched_chunks[:2]],
            })
        else:
            items.append({
                "type": r.source_type,
                "video_id": r.video_id,
                "document_id": r.document_id,
                "title": r.title,
                "author": r.author,
                "section": r.section_title,
                "score": round(r.score, 4),
                "text_snippet": r.text[:200],
                "obsidian_link": _make_obsidian_link(r.summary_path),
            })

    return {
        "query": result.query,
        "mode": result.mode,
        "results": items,
        "coverage": {
            "total_videos": result.total_videos,
            "matched": result.matched_count,
        },
        "elapsed_ms": result.elapsed_ms,
        "fallback_used": result.fallback_used,
    }


def _render_markdown(result: SearchResult):
    """Render search results as markdown for CLI display."""
    print()
    print(f"# Search: \"{result.query}\"")
    print(f"Mode: {result.mode} | Results: {len(result.results)} | "
          f"Time: {result.elapsed_ms}ms", end="")
    if result.fallback_used:
        print(" (grep fallback)")
    else:
        print()
    print()

    if not result.results:
        print("*[no results found]*")
        print()
        print(f"*Searched {result.total_videos} videos, 0 matched.*")
        return

    for i, r in enumerate(result.results):
        if isinstance(r, VideoMatch):
            link = _make_obsidian_link(r.summary_path)
            tags_display = r.tags.replace("|", "` `").strip("`").strip()
            print(f"### {i + 1}. {r.title}")
            if r.source_type == "document":
                print(f"**Type:** document | **Author:** {r.author} | **Date:** {r.date_iso} | "
                      f"**Score:** {r.best_score:.3f}")
            else:
                print(f"**Channel:** {r.channel} | **Date:** {r.date_iso} | "
                      f"**Rating:** {'⭐' * r.rating if r.rating > 0 else '—'} | "
                      f"**Score:** {r.best_score:.3f}")
            if tags_display:
                print(f"`{tags_display}`")
            if link:
                print(f"[Open in Obsidian]({link})")
            if r.matched_chunks:
                snippet = r.matched_chunks[0].text.replace(
                    r.matched_chunks[0].text.split("\n\n")[0], ""
                ).strip()[:300]
                if snippet:
                    print(f"> {snippet}...")
            print()
        else:
            link = _make_obsidian_link(r.summary_path)
            print(f"### {i + 1}. {r.title} — {r.section_title}")
            if r.source_type == "document":
                print(f"**Type:** document | **Score:** {r.score:.3f}")
            else:
                print(f"**Score:** {r.score:.3f}")
            if link:
                print(f"[Open in Obsidian]({link})")
            print(f"> {r.text[:300]}...")
            print()

    print(f"*Searched {result.total_videos} videos, "
          f"{result.matched_count} matched above threshold.*")
