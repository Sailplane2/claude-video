#!/usr/bin/env python3
"""Finalize video tags for /watch-save — the permanent tag program.

Reads the raw auto-tags from a summary body's `<!--TAGS: ...-->` line (or any
comma/space-separated tag string), normalizes to clean kebab-case, and — when
the summary carries no usable tags (e.g. Gemini-native video mode drops the
line) — generates tags via a DeepSeek fallback from the title + summary.

Wired into /watch-save Step 7 so every watch-save produces consistent tags
with no manual cleanup.

Usage:
    python3 make_tags.py --summary summary_body.md --title "Video Title" --json
    python3 make_tags.py --tags "reverse engineering, AI agents, claude code"

Prints comma-separated tags, or a JSON array with --json.
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

MAX_TAGS = 8
MIN_TAGS = 3
MODEL = "deepseek-v4-pro"
API_URL = "https://api.deepseek.com/anthropic/v1/messages"

STOPWORDS = set(
    """a an the and or but of to in on for with is are was were be been as at by
    from it its this that these those we you they i he she them his her my your
    our their about into over after before under between than too very just also
    not can will could should do does did how what why when where which who
    whom""".split()
)

# Bare generics that never belong as tags.
JUNK = {"video", "videos", "misc", "other", "talk", "general", "new"}


def normalize_tag(tag: str) -> str:
    """lowercase → runs of non-alphanumerics become a single hyphen → trim."""
    t = tag.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t


def is_junk(tag: str) -> bool:
    if not tag or len(tag) < 3:
        return True
    if tag in STOPWORDS or tag in JUNK:
        return True
    if tag.isdigit():
        return True
    return False


def split_tags(raw: str):
    """Comma/pipe/slash/newline-delimited lists keep multi-word tags intact;
    otherwise fall back to whitespace (single words)."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if re.search(r"[,\n|/]", raw):
        return [t.strip() for t in re.split(r"[,\n|/]+", raw) if t.strip()]
    return [t.strip() for t in raw.split() if t.strip()]


def title_keywords(title: str):
    out = []
    for w in re.findall(r"[A-Za-z0-9]+", title or ""):
        wl = w.lower()
        if len(wl) < 4 or wl in STOPWORDS or wl.isdigit():
            continue
        if wl not in out:
            out.append(wl)
    return out


def load_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    env_file = Path.home() / ".config/watch/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "DEEPSEEK_API_KEY":
                    return v.strip().strip('"').strip("'")
    return None


def llm_tags(title: str | None, summary_text: str | None) -> list[str]:
    """DeepSeek fallback: generate tags from the title + summary excerpt."""
    key = load_api_key()
    if not key:
        return []
    excerpt = (summary_text or "").strip()[:2000]
    prompt = (
        "You are a video cataloger. Given a video title and a summary, output "
        "exactly ONE line of 5-8 comma-separated kebab-case tags (lowercase, "
        "hyphen-separated, no spaces, no '#' prefix, no numbers or bullets) that "
        "best describe the video's topics and techniques. Prefer specific "
        "single-concept tags.\n\n"
        f"Title: {title}\n\nSummary:\n{excerpt}\n\nTags:"
    )
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 200,
        "thinking": {"type": "disabled"},
        "system": "You output only a comma-separated tag list, nothing else.",
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        text = "".join(
            b.get("text", "") for b in data.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text"
        )
        return split_tags(text)
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, KeyError, ValueError):
        return []


def make_tags(raw: str | None, title: str | None = None,
              summary_text: str | None = None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for t in split_tags(raw):
        n = normalize_tag(t)
        if is_junk(n) or n in seen:
            continue
        seen.add(n)
        tags.append(n)

    # LLM fallback when the summary carried too few usable tags
    # (e.g. Gemini-native video mode omits the <!--TAGS:--> line).
    if len(tags) < MIN_TAGS:
        for t in llm_tags(title, summary_text):
            n = normalize_tag(t)
            if is_junk(n) or n in seen:
                continue
            seen.add(n)
            tags.append(n)

    # Last resort: title keywords.
    if len(tags) < MIN_TAGS and title:
        for kw in title_keywords(title):
            if kw in seen or len(tags) >= MIN_TAGS:
                continue
            seen.add(kw)
            tags.append(kw)

    return tags[:MAX_TAGS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", help="raw tag string (e.g. from <!--TAGS:-->)")
    ap.add_argument("--title", help="video title for fallback")
    ap.add_argument("--summary", help="summary body path; extracts <!--TAGS:--> if --tags absent")
    ap.add_argument("--json", action="store_true", help="emit a JSON array")
    args = ap.parse_args()

    raw = args.tags
    summary_text = None
    if args.summary:
        p = Path(args.summary)
        if p.exists():
            summary_text = p.read_text(encoding="utf-8")
            if not raw:
                m = re.search(r"<!--TAGS:\s*(.*?)-->", summary_text)
                if m:
                    raw = m.group(1)

    tags = make_tags(raw, args.title, summary_text)

    if args.json:
        print(json.dumps(tags))
    else:
        print(", ".join(tags))


if __name__ == "__main__":
    main()
