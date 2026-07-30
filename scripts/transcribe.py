#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    segments: list[dict] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = _to_seconds(*match.groups()[:4])
        end = _to_seconds(*match.groups()[4:])
        i += 1

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        cue_text = " ".join(cue_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


MIN_OVERLAP_WORDS = 4
OVERLAP_SEARCH_WINDOW = 20  # only look near the boundary, never deep into merged history


def _word_overlap(prev_text: str, next_text: str) -> int:
    """Length (in words) of the longest suffix of prev_text that's also a prefix of next_text.

    Catches YouTube's 2-line rolling-caption format: each cue shows [chunk N, chunk N+1],
    the next cue shows [chunk N+1, chunk N+2] — the tail of one cue is the head of the next,
    not a growing prefix (already handled below) and not an exact repeat.

    Requires >= MIN_OVERLAP_WORDS AND at least half of next_text's words, so a coincidental
    short common phrase ("of the", "you know") can't false-positive a merge across two
    genuinely unrelated cues — real rolling-caption overlap covers roughly half a cue.
    Only searches the last OVERLAP_SEARCH_WINDOW words of prev_text, not its full (possibly
    long, already-merged) history, so one real match can't cascade into swallowing everything
    that follows.
    """
    prev_words = prev_text.split()[-OVERLAP_SEARCH_WINDOW:]
    next_words = next_text.split()
    min_needed = max(MIN_OVERLAP_WORDS, (len(next_words) + 1) // 2)
    max_check = min(len(prev_words), len(next_words))
    for k in range(max_check, min_needed - 1, -1):
        if prev_words[-k:] == next_words[:k]:
            return k
    return 0


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        if out:
            k = _word_overlap(out[-1]["text"], seg["text"])
            if k > 0:
                new_words = seg["text"].split()[k:]
                if new_words:
                    out[-1]["text"] = out[-1]["text"] + " " + " ".join(new_words)
                out[-1]["end"] = seg["end"]
                continue
        out.append(seg)
    return out


def filter_range(
    segments: list[dict],
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict]:
    """Return segments whose time range overlaps [start, end]."""
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [seg for seg in segments if seg["end"] >= lo and seg["start"] <= hi]


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = int(seg["start"])
        stamp = f"[{start // 60:02d}:{start % 60:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
