#!/usr/bin/env python3
"""Summarize a video using DeepSeek V4 Pro (Anthropic-compatible endpoint).

Reads frame descriptions and optional transcript, calls DeepSeek's API,
and outputs a structured markdown summary.

Usage:
    python3 summarize_video.py \\
        --descriptions <frame_descriptions.txt> \\
        [--transcript-file <transcript.txt>] \\
        [--title <video title>] \\
        [--question <user question>] \\
        [--out <output.md>]

API key: DEEPSEEK_API_KEY env var, or ~/.config/watch/.env
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "deepseek-v4-pro"
MAX_TOKENS = 4096
API_URL = "https://api.deepseek.com/anthropic/v1/messages"

SYSTEM_PROMPT = (
    "You are an expert video analyst. Given timestamped frame descriptions "
    "and optionally a transcript, produce a comprehensive, well-structured "
    "summary of the video in markdown format. Include:\n"
    "- An **Overview** section with the main topic and thesis\n"
    "- Key moments with approximate timestamps\n"
    "- Notable visuals mentioned in frame descriptions\n"
    "- A summary of spoken content (if transcript provided)\n"
    "- If a specific question was asked, answer it directly in a dedicated section\n"
    "Be concise and factual. Use bullet points and section headers for readability.\n\n"
    "At the very end of your response, ALWAYS append two metadata lines:\n"
    "<!--TAGS: tag1, tag2, tag3-->\n"
    "<!--TYPE: tutorial|talk|interview|lecture|documentary|other-->\n"
    "Choose 5-8 concise kebab-case tags (lowercase, hyphen-separated, no "
    "spaces, no '#' prefix, e.g. 'machine-learning') that best describe the "
    "video's topics and techniques — prefer specific single-concept tags over "
    "generic ones. Pick the single most accurate type. These lines must appear "
    "exactly as shown, on their own lines, after all other content."
)


def load_api_key():
    """Load DeepSeek API key from environment or ~/.config/watch/.env."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    env_file = Path.home() / ".config/watch/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "DEEPSEEK_API_KEY":
                    return v.strip().strip('"').strip("'")

    raise SystemExit(
        "DEEPSEEK_API_KEY not found in environment or ~/.config/watch/.env"
    )


def call_deepseek(api_key: str, prompt: str) -> str:
    """Call DeepSeek V4 Pro via Anthropic-compatible endpoint. Returns response text."""
    body = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "disabled"},
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())

        # Anthropic-compatible response: content is a list of blocks
        content = data.get("content", [])
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "thinking":
                # Should not appear with thinking disabled, but handle gracefully
                pass

        if not text_parts:
            raise ValueError(f"No text content in response. Keys: {list(data.keys())}")

        return "\n".join(text_parts)

    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise SystemExit(f"DeepSeek API HTTP {e.code}: {body[:500]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"DeepSeek API connection error: {e.reason}")
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
        raise SystemExit(f"DeepSeek API unexpected response: {e}")


def build_prompt(
    descriptions: str,
    transcript: str | None,
    title: str | None,
    question: str | None,
) -> str:
    """Assemble the user prompt from available inputs."""
    parts = []

    if title:
        parts.append(f"Video title: {title}\n")

    parts.append("## Frame Descriptions\n")
    parts.append(descriptions)

    if transcript:
        parts.append("\n## Transcript\n")
        parts.append(transcript)

    if question and question.strip():
        parts.append(f"\n## Question\n{question.strip()}")
    else:
        parts.append("\nProduce a full summary of this video.")

    return "\n".join(parts)


def main():
    args = sys.argv[1:]

    descriptions_path = None
    transcript_path = None
    title = None
    question = None
    out_path = None

    i = 0
    while i < len(args):
        if args[i] == "--descriptions" and i + 1 < len(args):
            descriptions_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--transcript-file" and i + 1 < len(args):
            transcript_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--title" and i + 1 < len(args):
            title = args[i + 1]
            i += 2
        elif args[i] == "--question" and i + 1 < len(args):
            question = args[i + 1]
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
        else:
            print(f"Unknown or incomplete flag: {args[i]}", file=sys.stderr)
            print(
                "Usage: python3 summarize_video.py --descriptions <path> "
                "[--transcript-file <path>] [--title <text>] "
                "[--question <text>] [--out <path>]",
                file=sys.stderr,
            )
            sys.exit(2)

    if not descriptions_path:
        print(
            "Usage: python3 summarize_video.py --descriptions <path> "
            "[--transcript-file <path>] [--title <text>] "
            "[--question <text>] [--out <path>]",
            file=sys.stderr,
        )
        sys.exit(2)

    if not descriptions_path.exists():
        print(f"Descriptions file not found: {descriptions_path}", file=sys.stderr)
        sys.exit(1)

    descriptions = descriptions_path.read_text().strip()
    if not descriptions:
        print(f"Descriptions file is empty: {descriptions_path}", file=sys.stderr)
        sys.exit(1)

    transcript = None
    if transcript_path:
        if transcript_path.exists():
            transcript = transcript_path.read_text().strip()
            if not transcript:
                print(
                    f"Warning: transcript file is empty ({transcript_path})",
                    file=sys.stderr,
                )
                transcript = None
        else:
            print(
                f"Warning: transcript file not found ({transcript_path})",
                file=sys.stderr,
            )

    api_key = load_api_key()

    prompt = build_prompt(descriptions, transcript, title, question)

    print(f"Summarizing with {MODEL}…", file=sys.stderr)
    t0 = time.time()

    summary = call_deepseek(api_key, prompt)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    if out_path:
        out_path.write_text(summary)
        print(f"Summary written to {out_path}", file=sys.stderr)
    else:
        print(summary)


if __name__ == "__main__":
    main()
