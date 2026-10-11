#!/usr/bin/env python3
"""Gemini-native video understanding: URL -> Gemini -> structured summary.

No download, no frames, no separate summarization. Gemini reads the video
directly from its URL (native YouTube understanding), so this bypasses
yt-dlp throttling entirely — the 403/429 blocking that stalls the local
pipeline never touches this path.

Produces the same `summary_body.md` format as the local watch-save pipeline
(same section structure, same `<!--TAGS:-->` / `<!--TYPE:-->` metadata lines),
so the downstream (tags/type extraction, vault write, index.json) is unchanged.

Usage:
    python3 gemini_native.py <url> [--title <text>] [--question <text>]
                            [--out <path.md>] [--thumb <path.jpg>] [--model <id>]

Exit 0 on success (writes + prints the summary); non-zero on failure.

API key: GEMINI_API_KEY in ~/.config/watch/.env (never printed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "gemini-2.5-flash"
MAX_TOKENS = 4096

SYSTEM_PROMPT = (
    "You are an expert video analyst. Watch the provided video and produce a "
    "comprehensive, well-structured markdown summary. Include:\n"
    "- An **Overview** section with the main topic and thesis\n"
    "- Key moments with approximate timestamps\n"
    "- Notable visuals\n"
    "- A summary of spoken content\n"
    "- If a specific question was asked, answer it directly in a dedicated "
    "section\n"
    "Be concise and factual. Use bullet points and section headers.\n\n"
    "At the very end of your response, ALWAYS append two metadata lines:\n"
    "<!--TAGS: tag1, tag2, tag3-->\n"
    "<!--TYPE: tutorial|talk|interview|lecture|documentary|other-->\n"
    "Choose 3-5 lowercase tags that best describe the video's topics and "
    "techniques. Use single words or hyphenated compounds only (no spaces), "
    "e.g. 'video-editing' not 'video editing'. Pick the single most accurate "
    "type. These lines must appear exactly as shown, on their own lines, "
    "after all other content."
)


def load_key() -> str:
    env_file = Path.home() / ".config/watch/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    return v.strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not found in ~/.config/watch/.env")


def normalize_url(url: str) -> str:
    """Convert youtu.be/ID short links to the canonical watch form.

    Gemini handles the full ``www.youtube.com/watch?v=`` form reliably; the
    short form is occasionally rejected.
    """
    m = re.match(r"https?://(?:www\.)?youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    return url


def extract_video_id(url: str) -> str | None:
    """Return the 11-char YouTube id from any common URL form, or None."""
    m = re.search(r"(?:v=|/shorts/|/embed/|youtu\.be/|/v/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def fetch_thumbnail(video_id: str, out_path: Path) -> bool:
    """Fetch the video's thumbnail from YouTube's static CDN endpoint.

    The thumbnail lives on a lightweight image endpoint, not the throttled
    video stream, so this succeeds even when yt-dlp's download 403s. Tries
    maxresdefault first, then hqdefault. Returns True on success.
    """
    for name in ("maxresdefault", "hqdefault"):
        url = f"https://i.ytimg.com/vi/{video_id}/{name}.jpg"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            continue
        if len(data) > 1024:  # skip empty/placeholder bodies
            out_path.write_bytes(data)
            return True
    return False


def build_text(title: str | None, question: str | None) -> str:
    parts = [SYSTEM_PROMPT]
    if title:
        parts.append(f"\n\nVideo title: {title}")
    if question and question.strip():
        parts.append(f"\n\n## Question\n{question.strip()}")
    else:
        parts.append("\n\nProduce a full summary of this video.")
    return "\n".join(parts)


def extract_text(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates in response. Keys: {list(data.keys())}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text:
        raise ValueError(f"No text in response: {json.dumps(data)[:500]}")
    return text


def call_gemini(video_url: str, text: str, api_key: str, model: str, max_tokens: int) -> str:
    body = json.dumps({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": text},
                {"fileData": {"fileUri": video_url, "mimeType": "video/mp4"}},
            ],
        }],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }).encode()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return extract_text(data)
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        raise SystemExit(f"Gemini API HTTP {e.code}: {err[:800]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Gemini API connection error: {e.reason}")
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
        raise SystemExit(f"Gemini API unexpected response: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="gemini_native",
        description="Gemini-native video understanding (URL -> summary, no download).",
    )
    ap.add_argument("url", help="Video URL (YouTube native; no download required)")
    ap.add_argument("--title", default=None, help="Video title (for context)")
    ap.add_argument("--question", default=None, help="Specific question to answer, else full summary")
    ap.add_argument("--out", default=None, help="Write summary to this file (default: print only)")
    ap.add_argument("--thumb", default=None, help="Save the video thumbnail (JPEG) to this path")
    ap.add_argument("--model", default=MODEL, help=f"Gemini model (default {MODEL})")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max output tokens")
    args = ap.parse_args()

    api_key = load_key()
    video_url = normalize_url(args.url)
    video_id = extract_video_id(args.url) or extract_video_id(video_url)
    text = build_text(args.title, args.question)

    if args.thumb and video_id:
        thumb_path = Path(args.thumb).expanduser().resolve()
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        if fetch_thumbnail(video_id, thumb_path):
            print(f"Thumbnail saved to {thumb_path}", file=sys.stderr)
        else:
            print("Thumbnail fetch failed (continuing without)", file=sys.stderr)

    print(f"Watching via Gemini ({args.model}) — no download…", file=sys.stderr)
    summary = call_gemini(video_url, text, api_key, args.model, args.max_tokens)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary, encoding="utf-8")
        print(f"Summary written to {out_path}", file=sys.stderr)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
