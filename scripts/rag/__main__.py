"""RAG Pipeline CLI — semantic search over video + document knowledge base.

Usage:
    python -m rag rebuild [--force] [--backend chroma|neo4j]
    python -m rag sync [--backend chroma|neo4j]
    python -m rag search <query> [opts] [--backend ...] [--kind ...]
    python -m rag status [--backend chroma|neo4j]
    python -m rag prune [--backend ...]
    python -m rag validate [--backend ...]
    python -m rag graph index [--backend ...] [--videos-dir PATH]
    python -m rag graph retrieve <query> [--backend ...] [-k N]
"""

import argparse
import sys


def _add_backend(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["chroma", "neo4j"], default="neo4j",
                   help="Storage backend (default: neo4j). chroma is the frozen fallback.")


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Video + Document Knowledge Base RAG Pipeline — semantic search.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── rebuild ──
    p = subparsers.add_parser("rebuild", help="Full index rebuild from scratch")
    p.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    _add_backend(p)

    # ── sync ──
    p = subparsers.add_parser("sync", help="Incremental: add new, update stale, prune deleted")
    _add_backend(p)

    # ── search ──
    p = subparsers.add_parser("search", help="Search the knowledge base")
    p.add_argument("query", nargs="+", help="Search query terms")
    p.add_argument("--mode", choices=["videos", "passages"], default="videos",
                   help="Result grouping (default: videos)")
    p.add_argument("-k", type=int, default=25, help="Max results (default: 25)")
    p.add_argument("--tags", help="Filter: comma-separated tags (AND semantics)")
    p.add_argument("--channel", help="Filter by channel name")
    p.add_argument("--platform", help="Filter by platform (youtube, local, etc.)")
    p.add_argument("--type", dest="video_type", help="Filter by video type (tutorial, talk, etc.)")
    p.add_argument("--after", help="Date lower bound: YYYY-MM-DD")
    p.add_argument("--before", help="Date upper bound: YYYY-MM-DD")
    p.add_argument("--min-rating", type=int, help="Minimum rating (1-5)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--no-fallback", action="store_true", help="Skip grep fallback on error")
    _add_backend(p)
    p.add_argument("--kind", choices=["video", "document", "all"], default="all",
                   help="Content type filter (default: all; documents require --backend neo4j)")

    # ── status ──
    p = subparsers.add_parser("status", help="Print coverage and consistency report")
    _add_backend(p)

    # ── prune ──
    p = subparsers.add_parser("prune", help="Remove deleted/archived videos from index")
    _add_backend(p)

    # ── validate ──
    p = subparsers.add_parser("validate", help="Cross-validate filesystem vs store")
    _add_backend(p)

    # ── graph (LangGraph orchestration) ──
    p = subparsers.add_parser("graph", help="Run the LangGraph orchestration layer directly")
    gsub = p.add_subparsers(dest="graph_command")

    gi = gsub.add_parser("index", help="Batch ingest graph (scan → chunk → store). "
                                       "Idempotent upsert, no drop.")
    gi.add_argument("--videos-dir", help="Override videos directory (isolated testing)")
    _add_backend(gi)

    gr = gsub.add_parser("retrieve", help="Interactive retrieval graph (query → embed → retrieve)")
    gr.add_argument("query", nargs="+", help="Search query terms")
    gr.add_argument("-k", type=int, default=25, help="Max results (default: 25)")
    _add_backend(gr)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "rebuild":
        from . import rebuild
        sys.exit(rebuild.run(force=args.force, backend=args.backend))

    elif args.command == "sync":
        from . import sync
        sys.exit(sync.run(backend=args.backend))

    elif args.command == "search":
        if args.kind != "all" and args.backend != "neo4j":
            print("error: document search requires --backend neo4j", file=sys.stderr)
            sys.exit(2)
        from . import search
        search.run(
            query=" ".join(args.query),
            mode=args.mode,
            k=args.k,
            tags=args.tags,
            channel=args.channel,
            platform=args.platform,
            video_type=args.video_type,
            after_date=args.after,
            before_date=args.before,
            min_rating=args.min_rating,
            json_output=args.json,
            no_fallback=args.no_fallback,
            backend=args.backend,
            kind=args.kind,
        )

    elif args.command == "status":
        from . import status
        status.run(backend=args.backend)

    elif args.command == "prune":
        from . import prune
        sys.exit(prune.run(backend=args.backend))

    elif args.command == "validate":
        from . import validate
        sys.exit(validate.run(backend=args.backend))

    elif args.command == "graph":
        from . import orchestrator
        if args.graph_command == "index":
            result = orchestrator.run_indexing(backend=args.backend, videos_dir=args.videos_dir)
            r = result["report"]
            print(f"Indexing complete: {r['videos_scanned']} scanned, "
                  f"{r['videos_processed']} processed, {r['videos_skipped']} skipped, "
                  f"{r['videos_failed']} failed → {r['total_chunks']} chunks")
        elif args.graph_command == "retrieve":
            query = " ".join(args.query)
            result = orchestrator.run_retrieval(query, k=args.k, backend=args.backend)
            raw = result["raw"]
            ids = raw["ids"][0]
            distances = raw["distances"][0]
            metadatas = raw["metadatas"][0]
            print(f"{len(ids)} hits for: {query}")
            for i, cid in enumerate(ids):
                title = metadatas[i].get("title", "(untitled)")
                print(f"  {i + 1}. {title}  (distance {distances[i]:.4f})  [{cid}]")
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
