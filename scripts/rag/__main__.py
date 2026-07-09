"""RAG Pipeline CLI — semantic search over video knowledge base.

Usage:
    python -m rag rebuild [--force]     Full index rebuild
    python -m rag sync                   Incremental update
    python -m rag search <query> [opts]  Search videos
    python -m rag status                 Coverage report
    python -m rag prune                  Remove deleted/archived
    python -m rag validate               Cross-check consistency
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Video Knowledge Base RAG Pipeline — semantic search over saved videos.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── rebuild ──
    p = subparsers.add_parser("rebuild", help="Full index rebuild from scratch")
    p.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # ── sync ──
    subparsers.add_parser("sync", help="Incremental: add new, update stale, prune deleted")

    # ── search ──
    p = subparsers.add_parser("search", help="Search the video knowledge base")
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

    # ── status ──
    subparsers.add_parser("status", help="Print coverage and consistency report")

    # ── prune ──
    subparsers.add_parser("prune", help="Remove deleted/archived videos from index")

    # ── validate ──
    subparsers.add_parser("validate", help="Cross-validate filesystem vs Chroma")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "rebuild":
        from . import rebuild
        sys.exit(rebuild.run(force=args.force))

    elif args.command == "sync":
        from . import sync
        sys.exit(sync.run())

    elif args.command == "search":
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
        )

    elif args.command == "status":
        from . import status
        status.run()

    elif args.command == "prune":
        from . import prune
        sys.exit(prune.run())

    elif args.command == "validate":
        from . import validate
        sys.exit(validate.run())


if __name__ == "__main__":
    main()
