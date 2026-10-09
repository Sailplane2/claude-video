#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from download import download, is_url  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, deduplicate_frames, extract, format_time, get_metadata, parse_time  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


# ---- Mode defaults ----------------------------------------------------------

MODE_DEFAULTS = {
    "transcript": {"keyframe": False, "scene_detect": False, "dedup": False, "max_frames": 0},
    "efficient":  {"keyframe": True,  "scene_detect": False, "dedup": True,  "max_frames": 50},
    "balanced":   {"keyframe": False, "scene_detect": True,  "dedup": True,  "max_frames": 80},
    "deep":       {"keyframe": False, "scene_detect": True,  "dedup": False, "max_frames": 100},
}

# ---- Deictic scanning -------------------------------------------------------

DEICTIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\blook (here|at this)\b",
        r"\bas you (can |)see\b",
        r"\bwatch (what|this)\b",
        r"\b(check this|check it) out\b",
        r"\bsee (what|that|this|here)\b",
        r"\bright (here|there)\b",
        r"\bnotice (how|that|this)\b",
        r"\bobserve (how|that|this)\b",
        r"\b(you'll|you will) (see|notice)\b",
        r"\bfocus on (this|here)\b",
        r"\b(pay|paying) attention to (this|here)\b",
        r"\bthis (part|section|clip|moment|one)\b",
        r"\bthat (part|section|clip|moment)\b",
        r"\bthe (section|part) (where|when)\b",
        r"\b(here's|here is) (what|how|where|why)\b",
        r"\bthis is (what|how|where|why)\b",
        r"\blet me (show|demonstrate)\b",
        r"\bi'?ll (show|demonstrate|pull up)\b",
        r"\bif you look\b",
    ]
]

DEICTIC_WINDOW_SECONDS: float = 0.5
"""Grab a frame 0.5s before the deictic cue (accounts for caption lag)."""

DEICTIC_MAX_FRAMES: int = 20
"""Hard cap on deictic frames to avoid blowing the token budget."""


def _find_deictic_timestamps(segments: list[dict]) -> list[float]:
    """Scan transcript segments for deictic language and return timestamps.

    Returns a deduplicated, sorted list of seconds where the speaker is
    likely referencing something on screen.
    """
    if not segments:
        return []

    timestamps: list[float] = []
    seen: set[int] = set()  # dedup by whole-second bucket

    for seg in segments:
        text = seg.get("text", "")
        start = seg.get("start", 0.0)

        for pat in DEICTIC_PATTERNS:
            if pat.search(text):
                ts = max(0.0, start - DEICTIC_WINDOW_SECONDS)
                bucket = int(ts)
                if bucket not in seen:
                    seen.add(bucket)
                    timestamps.append(ts)
                break  # one match per segment is enough

    timestamps.sort()

    # Cap and even-sample if needed
    if len(timestamps) > DEICTIC_MAX_FRAMES:
        step = max(1, len(timestamps) // DEICTIC_MAX_FRAMES)
        timestamps = timestamps[::step][:DEICTIC_MAX_FRAMES]

    return timestamps


def _extract_frames_at_timestamps(
    video_path: str,
    out_dir: Path,
    timestamps: list[float],
    resolution: int = 512,
    start_index: int = 9000,
) -> list[Path]:
    """Grab a single frame at each timestamp using ffmpeg seeking.

    Each timestamp gets one frame.  Writes ``frame_NNNN.jpg`` starting at
    *start_index*.  Returns the list of frame Paths actually written.
    """
    if not timestamps:
        return []

    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_written: list[Path] = []

    for i, ts in enumerate(timestamps):
        idx = start_index + i
        out_path = out_dir / f"frame_{idx:04d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vf", f"scale={resolution}:-2",
            "-vframes", "1",
            "-q:v", "4",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists():
            frames_written.append(out_path)
        else:
            err = result.stderr.strip()[:120]
            print(f"[watch] deictic frame at {format_time(ts)} failed: {err}",
                  file=sys.stderr)

    return frames_written


# ---- Main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument(
        "--mode",
        choices=["transcript", "efficient", "balanced", "deep"],
        default="efficient",
        help=(
            "Extraction mode. transcript: no frames, transcript only. "
            "efficient: keyframes + dedup (~40x faster). "
            "balanced: scene-detect + dedup (default). "
            "deep: scene-detect, uncapped, no dedup (token-burner)."
        ),
    )
    ap.add_argument("--max-frames", type=int, default=None, help="Override mode's default frame cap (hard max 100)")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps (timer mode only)")
    ap.add_argument(
        "--scene-detect", action="store_true",
        help="Override: use scene detection instead of mode default",
    )
    ap.add_argument(
        "--scene-threshold", type=float, default=0.3,
        help="Scene change sensitivity (0.1-1.0, lower = more frames). Default 0.3.",
    )
    ap.add_argument(
        "--keyframe", action="store_true",
        help="Override: extract only keyframes (I-frames). ~40x faster than scene-detect.",
    )
    ap.add_argument(
        "--dedup", action="store_true",
        help="Override: enable frame deduplication via perceptual hashing.",
    )
    ap.add_argument(
        "--no-dedup", action="store_true",
        help="Override: disable dedup even if the mode defaults to it.",
    )
    ap.add_argument(
        "--no-deictic", action="store_true",
        help="Disable deictic reference scanning.",
    )
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper", action="store_true",
        help="Disable Whisper fallback. Report frames-only if no captions available.",
    )
    ap.add_argument(
        "--whisper", choices=["groq", "openai"], default=None,
        help="Force a specific Whisper backend. Default: prefer Groq, fall back to OpenAI.",
    )
    args = ap.parse_args()

    # ---- Resolve mode -------------------------------------------------------

    mode_cfg = MODE_DEFAULTS[args.mode].copy()

    # CLI overrides
    if args.scene_detect:
        mode_cfg["scene_detect"] = True
        mode_cfg["keyframe"] = False
    if args.keyframe:
        mode_cfg["keyframe"] = True
        mode_cfg["scene_detect"] = False
    if args.dedup:
        mode_cfg["dedup"] = True
    if args.no_dedup:
        mode_cfg["dedup"] = False

    do_deictic = not args.no_deictic and mode_cfg["max_frames"] > 0
    # transcript mode = no frames, no deictic
    if mode_cfg["max_frames"] == 0:
        do_deictic = False

    max_frames = min(args.max_frames if args.max_frames is not None else mode_cfg["max_frames"], 100)

    # ---- Setup --------------------------------------------------------------

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    print(
        "[watch] downloading via yt-dlp…" if is_url(args.source) else "[watch] using local file…",
        file=sys.stderr,
    )
    dl = download(args.source, work / "download")
    video_path = dl["video_path"]

    meta = get_metadata(video_path)
    full_duration = meta["duration_seconds"]

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )

    # ---- Transcript (parse early — needed for deictic scan) -----------------

    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    if dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    if not transcript_segments and not args.no_whisper:
        backend, api_key = load_api_key(args.whisper)
        if backend and api_key:
            try:
                all_segments, used_backend = transcribe_video(
                    video_path, work / "audio.mp3", backend=backend, api_key=api_key,
                )
                transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                transcript_text = format_transcript(transcript_segments)
                transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                print(f"[watch] whisper fallback failed: {exc}", file=sys.stderr)
        else:
            hint = (
                f"--whisper {args.whisper} was set but the matching API key is missing"
                if args.whisper else
                "no subtitles and no Whisper API key found"
            )
            setup_py = SCRIPT_DIR / "setup.py"
            print(
                f"[watch] {hint} — run `python3 {setup_py}` to enable the Whisper fallback",
                file=sys.stderr,
            )

    # Persist the transcript to a file so it survives the download-dir cleanup.
    # The caption VTT lives inside download/ (which the caller deletes); once it
    # is gone, recovering the text means a fresh yt-dlp caption request that
    # YouTube may 429-rate-limit. Writing transcript.txt here makes that
    # re-request unnecessary.
    if transcript_text:
        transcript_path = work / "transcript.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        print(f"[watch] transcript saved → {transcript_path}", file=sys.stderr)

    # ---- Frame extraction ---------------------------------------------------

    frames: list[dict] = []

    if mode_cfg["max_frames"] > 0:
        if mode_cfg["keyframe"]:
            label = f"mode={args.mode} (keyframe"
        elif mode_cfg["scene_detect"]:
            label = f"mode={args.mode} (scene-detect"
        else:
            label = f"mode={args.mode} (timer, {fps:.3f} fps"

        if mode_cfg["dedup"]:
            label += " + dedup"
        label += f", max {max_frames})"
        print(f"[watch] extracting frames over {scope} — {label}…", file=sys.stderr)

        frames = extract(
            video_path,
            work / "frames",
            fps=fps,
            resolution=args.resolution,
            max_frames=max_frames,
            start_seconds=start_sec,
            end_seconds=end_sec,
            scene_detect=mode_cfg["scene_detect"],
            scene_threshold=args.scene_threshold,
            keyframe=mode_cfg["keyframe"],
            dedup=mode_cfg["dedup"],
        )
    else:
        print(f"[watch] mode=transcript — skipping frame extraction", file=sys.stderr)

    # ---- Deictic scanning ---------------------------------------------------

    deictic_timestamps: list[float] = []
    deictic_frames: list[Path] = []
    if do_deictic and transcript_segments:
        deictic_timestamps = _find_deictic_timestamps(transcript_segments)
        if deictic_timestamps:
            print(
                f"[watch] deictic: found {len(deictic_timestamps)} potential screen-reference moments",
                file=sys.stderr,
            )
            deictic_frames = _extract_frames_at_timestamps(
                video_path, work / "frames", deictic_timestamps, resolution=args.resolution,
            )
            if deictic_frames:
                # Merge: append deictic frames, sort by frame number, assign timestamps
                existing_paths = {f["path"] for f in frames}
                next_idx = len(frames)
                # Map deictic frames to their closest timestamp for labeling
                ts_map: dict[str, float] = {}
                for i, dp in enumerate(sorted(deictic_frames)):
                    ts = deictic_timestamps[min(i, len(deictic_timestamps) - 1)]
                    ts_map[str(dp)] = ts

                for dp in sorted(deictic_frames):
                    dp_str = str(dp)
                    if dp_str not in existing_paths:
                        frames.append({
                            "index": next_idx,
                            "timestamp_seconds": round(ts_map.get(dp_str, 0.0), 2),
                            "path": dp_str,
                            "deictic": True,
                        })
                        next_idx += 1
                        existing_paths.add(dp_str)
                # Re-sort by filename (which preserves chronological order
                # since deictic frames have high index numbers)
                frames.sort(key=lambda f: f["path"])
                for i, f in enumerate(frames):
                    f["index"] = i

                print(
                    f"[watch] deictic: added {len(deictic_frames)} frames "
                    f"(total now {len(frames)})",
                    file=sys.stderr,
                )

    # ---- Report -------------------------------------------------------------

    info = dl.get("info") or {}

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source}")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")

    mode_label = "focused" if focused else "full"
    method_parts = [args.mode]
    if mode_cfg["dedup"]:
        method_parts.append("dedup")
    if deictic_frames:
        method_parts.append(f"+{len(deictic_frames)}deictic")
    print(f"- **Frames:** {len(frames)} @ {' '.join(method_parts)}, {mode_label} mode (max {max_frames})")
    print(f"- **Frame size:** {args.resolution}px wide")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
    else:
        print("- **Transcript:** none available")

    if not focused and full_duration > 600:
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length — "
            "accuracy degrades noticeably on anything over 10 minutes. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a specific section."
        )

    if frames:
        print()
        print("## Frames")
        print()
        print(f"Frames live at: `{work / 'frames'}`")
        print()
        print(
            "**Read each frame path below with the Read tool to view the image.** "
            "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video."
        )
        print()
        for frame in frames:
            print(f"- `{frame['path']}` (t={format_time(frame['timestamp_seconds'])})")

    print()
    print("## Transcript")
    print()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            print(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            print(f"_Source: {label}._")
        print()
        print("```")
        print(transcript_text)
        print("```")
    elif focused and dl.get("subtitle_path"):
        print(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        print(
            "_No transcript available — proceed with frames only. "
            "Captions were missing and the Whisper fallback was unavailable "
            "(no API key set, or `--no-whisper` was used). "
            f"Run `python3 {setup_py}` to enable Whisper, then re-run._"
        )

    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
