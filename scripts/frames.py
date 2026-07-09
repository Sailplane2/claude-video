#!/usr/bin/env python3
"""Probe video metadata and extract frames at an auto-scaled fps or via scene detection.

Auto-fps targets a frame budget, not a fixed rate. Token cost scales with frame
count, so budget-by-duration keeps short videos dense and long videos capped.
When a user-specified range is passed, focused-mode budgets denser (they are
zooming in for detail).

Scene detection (--scene-detect) uses ffmpeg's select filter to grab a frame
whenever the visual scene changes, rather than on a fixed timer. This is
vastly better for long videos (1hr+ courses, lectures) where timer-based
extraction misses content between widely-spaced intervals.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


MAX_FPS = 2.0


def _even_sample(total: int, keep: int) -> set[int]:
    """Return evenly-spaced indices to keep when reducing total → keep frames."""
    if keep >= total:
        return set(range(total))
    step = (total - 1) / (keep - 1) if keep > 1 else 0
    return {min(total - 1, int(round(i * step))) for i in range(keep)}


def _clamp_fps(fps: float, duration_seconds: float, max_frames: int) -> tuple[float, int]:
    fps = min(fps, MAX_FPS)
    target = min(max_frames, max(1, int(round(fps * duration_seconds))))
    return fps, target


def parse_time(value: str | float | int | None) -> float | None:
    """Parse SS, MM:SS, or HH:MM:SS (with optional .ms) into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise SystemExit(f"Cannot parse time value: {value!r} (expected SS, MM:SS, or HH:MM:SS)")


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def get_metadata(video_path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
        "size_bytes": int(fmt.get("size") or 0),
        "has_audio": audio_stream is not None,
    }


def auto_fps(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Pick fps that targets a sensible frame budget for full-video scans."""
    if duration_seconds <= 0:
        return 1.0, 1

    if duration_seconds <= 30:
        target = min(max_frames, max(12, int(round(duration_seconds))))
    elif duration_seconds <= 60:
        target = min(max_frames, 40)
    elif duration_seconds <= 180:  # 3 min
        target = min(max_frames, 60)
    elif duration_seconds <= 600:  # 10 min
        target = min(max_frames, 80)
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def auto_fps_focus(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Denser budget for user-specified ranges — they are zooming in for detail."""
    if duration_seconds <= 0:
        return min(MAX_FPS, 2.0), 2

    if duration_seconds <= 5:
        target = min(max_frames, max(10, int(round(duration_seconds * 6))))
    elif duration_seconds <= 15:
        target = min(max_frames, max(30, int(round(duration_seconds * 4))))
    elif duration_seconds <= 30:
        target = min(max_frames, 60)
    elif duration_seconds <= 60:
        target = min(max_frames, 80)
    elif duration_seconds <= 180:
        target = max_frames
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def _average_hash(img_path: Path, hash_size: int = 8) -> int:
    """Compute the average hash (aHash) of an image.

    Returns a 64-bit integer where each bit is 1 if the corresponding pixel
    is brighter than the mean.  Uses PIL; returns 0 if PIL is unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        return 0

    img = Image.open(img_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    mean = sum(pixels) / len(pixels)
    bits = [1 if p > mean else 0 for p in pixels]
    return int("".join(str(b) for b in bits), 2)


def _hamming_distance(a: int, b: int) -> int:
    """Count the number of differing bits between two integers (hashes)."""
    return (a ^ b).bit_count()


def deduplicate_frames(frame_paths: list[Path], threshold: int = 10) -> list[Path]:
    """Remove near-duplicate adjacent frames using perceptual hashing.

    Compares each frame to the last *kept* frame.  If the Hamming distance
    between their average hashes is below *threshold* (out of 64 bits), the
    frame is discarded.

    Returns the subset of *frame_paths* that survived deduplication.  Always
    keeps the first frame.
    """
    if len(frame_paths) <= 1:
        return frame_paths

    kept: list[Path] = [frame_paths[0]]
    last_hash = _average_hash(frame_paths[0])

    for p in frame_paths[1:]:
        h = _average_hash(p)
        if _hamming_distance(last_hash, h) >= threshold:
            kept.append(p)
            last_hash = h
        else:
            p.unlink()  # remove duplicate from disk

    return kept


def extract(
    video_path: str,
    out_dir: Path,
    fps: float,
    resolution: int = 512,
    max_frames: int = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    scene_detect: bool = False,
    scene_threshold: float = 0.3,
    keyframe: bool = False,
    dedup: bool = False,
    dedup_threshold: int = 10,
) -> list[dict]:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
    ]

    # -ss before -i = fast seek (keyframe-snap, good enough for preview frames).
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]

    cmd += ["-i", video_path]

    if keyframe:
        # Keyframe-only: select I-frames (pre-encoded full snapshots).
        # No decoding of delta frames — ~40× faster than scene-detect.
        vf = f"select='eq(pict_type,I)',scale={resolution}:-2"
        cmd += [
            "-vf", vf,
            "-vsync", "vfr",
            "-q:v", "4",
            output_pattern,
        ]
    elif scene_detect:
        # Scene detection: grab a frame whenever the visual scene changes.
        vf = f"select='gt(scene,{scene_threshold})',scale={resolution}:-2"
        cmd += [
            "-vf", vf,
            "-vsync", "vfr",
            "-q:v", "4",
            output_pattern,
        ]
    else:
        cmd += [
            "-vf", f"fps={fps},scale={resolution}:-2",
            "-frames:v", str(max_frames),
            "-q:v", "4",
            output_pattern,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    frames = sorted(out_dir.glob("frame_*.jpg"))

    # Keyframe / scene detection can produce more frames than max_frames. Sample evenly.
    if (scene_detect or keyframe) and len(frames) > max_frames:
        keep = _even_sample(len(frames), max_frames)
        for i, p in enumerate(frames):
            if i not in keep:
                p.unlink()
        frames = [p for i, p in enumerate(frames) if i in keep]

    # Deduplicate: discard near-identical adjacent frames (talking-head killer).
    if dedup and len(frames) > 1:
        before = len(frames)
        frames = deduplicate_frames(frames, threshold=dedup_threshold)
        if len(frames) < before:
            # Renumber remaining frames to keep sequential naming
            for i, p in enumerate(frames):
                new_name = out_dir / f"frame_{i + 1:04d}.jpg"
                if p != new_name:
                    p.rename(new_name)
            frames = sorted(out_dir.glob("frame_*.jpg"))

    offset = start_seconds or 0.0
    return [
        {
            "index": i,
            "timestamp_seconds": round(offset + (i / fps if fps > 0 else 0.0), 2),
            "path": str(p),
        }
        for i, p in enumerate(frames)
    ]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: frames.py <video-path> <out-dir> [--fps F] [--resolution W] "
            "[--max-frames N] [--start T] [--end T]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    out = Path(sys.argv[2])
    args = sys.argv[3:]

    fps_override = None
    resolution = 512
    max_frames = 100
    start_arg = None
    end_arg = None
    i = 0
    while i < len(args):
        if args[i] == "--fps":
            fps_override = float(args[i + 1]); i += 2
        elif args[i] == "--resolution":
            resolution = int(args[i + 1]); i += 2
        elif args[i] == "--max-frames":
            max_frames = int(args[i + 1]); i += 2
        elif args[i] == "--start":
            start_arg = args[i + 1]; i += 2
        elif args[i] == "--end":
            end_arg = args[i + 1]; i += 2
        else:
            i += 1

    meta = get_metadata(video)
    start_sec = parse_time(start_arg)
    end_sec = parse_time(end_arg)
    full_duration = meta["duration_seconds"]

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)

    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if fps_override is not None:
        fps = fps_override
        target = max(1, int(round(fps * effective_duration)))

    frames = extract(
        video, out,
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )
    print(json.dumps(
        {"meta": meta, "fps": fps, "target": target, "focused": focused, "frames": frames},
        indent=2,
    ))
