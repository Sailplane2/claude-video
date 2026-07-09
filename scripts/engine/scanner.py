"""Scanner — discover and validate video source files.

Walks a source folder recursively, runs ``ffprobe`` on each candidate file,
extracts technical metadata, computes a content hash, and validates that the
file is suitable for downstream processing.

Usage::

    from engine.scanner import Scanner, ScanResult

    scanner = Scanner("/path/to/footage/")
    result: ScanResult = scanner.run()

    for clip in result.validated:
        print(clip.id, clip.duration.to_seconds())
    for rejected in result.rejected:
        print(rejected.path, rejected.rejection_reason)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .rational_time import RationalTime
from .schemas import ScanStatus, SourceClip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SUPPORTED_CODECS: set[str] = {
    "h264", "h265", "hevc", "prores", "dnxhd", "dnxhr",
    "mpeg4", "av1", "vp9",
}

FFPROBE_TIMEOUT_SECONDS: int = 10
"""Per-file timeout for ffprobe calls."""

MIN_CLIP_DURATION_SECONDS: float = 1.0
"""Files shorter than this are rejected."""

CONTENT_HASH_CHUNK_BYTES: int = 65536
"""Number of bytes from the start of the file to hash."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Aggregated output of a scanner run."""

    validated: list[SourceClip] = field(default_factory=list)
    """Clips that passed all validation checks."""

    rejected: list[SourceClip] = field(default_factory=list)
    """Clips that were rejected, each with a *rejection_reason*."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal warnings surfaced during the scan."""

    duration_seconds: float = 0.0
    """Wall-clock time spent scanning."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_ffprobe() -> str:
    """Locate the ``ffprobe`` binary.  Raises ``SystemError`` if not found."""
    for candidate in ("ffprobe", "/usr/local/bin/ffprobe", "/opt/homebrew/bin/ffprobe"):
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise SystemError(
        "ffprobe not found.  Install ffmpeg: `brew install ffmpeg`"
    )


_COMPUTED_FFPROBE: str | None = None


def _ffprobe_path() -> str:
    global _COMPUTED_FFPROBE
    if _COMPUTED_FFPROBE is None:
        _COMPUTED_FFPROBE = _find_ffprobe()
    return _COMPUTED_FFPROBE


def _run_ffprobe(path: str) -> dict:
    """Run ffprobe on *path* and return the parsed JSON dict.

    Raises ``RuntimeError`` on failure (timeout, corrupt file, etc.).
    """
    cmd = [
        _ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=FFPROBE_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        stderr = proc.stderr.strip() or "ffprobe returned no output"
        raise RuntimeError(f"ffprobe failed: {stderr}")

    data = json.loads(proc.stdout)
    return data


def _compute_hash(path: str) -> str:
    """SHA-256 hash of the first 64 kB of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(CONTENT_HASH_CHUNK_BYTES)
        h.update(chunk)
    return h.hexdigest()


def _is_vfr(stream: dict) -> bool:
    """Detect variable frame-rate by comparing ``r_frame_rate`` and
    ``avg_frame_rate`` from an ffprobe stream dict.

    If they differ by more than 0.5 % it is considered VFR.
    """
    r_frac = stream.get("r_frame_rate", "0/1")
    a_frac = stream.get("avg_frame_rate", "0/1")

    try:
        r_num, r_den = map(int, r_frac.split("/"))
        a_num, a_den = map(int, a_frac.split("/"))
    except (ValueError, ZeroDivisionError):
        return True  # unparseable → assume VFR

    if r_den == 0 or a_den == 0:
        return True

    r_fps = r_num / r_den
    a_fps = a_num / a_den

    if r_fps == 0:
        return True

    return abs(r_fps - a_fps) / r_fps > 0.005


def _parse_fps(stream: dict) -> tuple[int, int]:
    """Extract frame-rate numerator and denominator from an ffprobe stream.

    Prefers ``r_frame_rate``; falls back to ``avg_frame_rate``.
    """
    for key in ("r_frame_rate", "avg_frame_rate"):
        frac = stream.get(key, "")
        if "/" in frac:
            parts = frac.split("/")
            if len(parts) == 2:
                try:
                    num = int(parts[0])
                    den = int(parts[1])
                    if num > 0 and den > 0:
                        return num, den
                except ValueError:
                    continue
    return 30, 1  # sensible fallback


def _ffprobe_to_source_clip(path: str, data: dict) -> SourceClip:
    """Convert the raw ffprobe JSON output for a single file into a
    ``SourceClip`` instance."""
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    # Find the first video stream
    video_stream: dict = {}
    audio_stream: dict = {}
    for s in streams:
        if s.get("codec_type") == "video" and not video_stream:
            video_stream = s
        elif s.get("codec_type") == "audio" and not audio_stream:
            audio_stream = s

    # Duration
    duration_sec = float(fmt.get("duration", 0))
    fps_num, fps_den = _parse_fps(video_stream)
    fps_rate = RationalTime(0, 1, fps_num, fps_den)
    duration = RationalTime.from_seconds(duration_sec, fps_rate)

    clip_id = _make_clip_id(path, fmt.get("size", "0"))

    return SourceClip(
        id=clip_id,
        path=os.path.abspath(path),
        file_size_bytes=int(fmt.get("size", 0)),
        source_hash=_compute_hash(path),
        duration=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps_numerator=fps_num,
        fps_denominator=fps_den,
        codec=video_stream.get("codec_name", "unknown"),
        bitrate=int(fmt.get("bit_rate", 0)),
        audio_codec=audio_stream.get("codec_name"),
        audio_channels=int(audio_stream.get("channels", 0)),
        audio_sample_rate=int(audio_stream.get("sample_rate", 0)),
        variable_frame_rate=_is_vfr(video_stream),
    )


def _make_clip_id(path: str, size: str) -> str:
    """Produce a short, unique clip identifier from the path and file size."""
    stem = Path(path).stem
    # Hash the full path + size for uniqueness across folders
    h = hashlib.md5(f"{path}:{size}".encode()).hexdigest()[:8]
    return f"clip_{stem}_{h}"


# ---------------------------------------------------------------------------
# Scanner class
# ---------------------------------------------------------------------------

class Scanner:
    """Discover and validate video source files.

    Parameters
    ----------
    source_folder:
        Root folder to scan recursively.
    extensions:
        Set of file extensions to consider (default ``.mov, .mp4, .mxf, .m4v``).
    parallel:
        Number of parallel ffprobe workers (default 4).
    """

    def __init__(
        self,
        source_folder: str,
        extensions: set[str] | None = None,
        parallel: int = 4,
    ) -> None:
        self.source_folder = os.path.abspath(source_folder)
        self.extensions = extensions or {".mov", ".mp4", ".mxf", ".m4v"}
        self.parallel = parallel

    # ---- Public API --------------------------------------------------------

    def run(self) -> ScanResult:
        """Execute the scan.  Returns a ``ScanResult``."""
        start = time.time()
        clip_paths = self._discover_files()

        if not clip_paths:
            logger.warning("No video files found in %s", self.source_folder)
            result = ScanResult(duration_seconds=time.time() - start)
            result.warnings.append(f"No video files found in {self.source_folder}")
            return result

        validated: list[SourceClip] = []
        rejected: list[SourceClip] = []
        warnings: list[str] = []

        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            fut_to_path = {
                pool.submit(self._process_file, p): p for p in clip_paths
            }
            for fut in as_completed(fut_to_path):
                try:
                    clip, issues = fut.result()
                    if clip.scan_status == ScanStatus.VALIDATED:
                        validated.append(clip)
                    else:
                        rejected.append(clip)
                    warnings.extend(issues)
                except Exception as exc:
                    path = fut_to_path[fut]
                    rejected.append(
                        SourceClip(
                            id=_make_clip_id(path, "0"),
                            path=os.path.abspath(path),
                            file_size_bytes=0,
                            source_hash="",
                            duration=RationalTime(0, 1),
                            width=0, height=0,
                            fps_numerator=0, fps_denominator=1,
                            codec="unknown", bitrate=0,
                            scan_status=ScanStatus.ERROR,
                            rejection_reason=str(exc),
                        )
                    )
                    warnings.append(f"Error processing {path}: {exc}")

        result = ScanResult(
            validated=validated,
            rejected=rejected,
            warnings=warnings,
            duration_seconds=time.time() - start,
        )

        logger.info(
            "Scan complete: %d validated, %d rejected in %.1fs",
            len(validated), len(rejected), result.duration_seconds,
        )
        return result

    # ---- Internal ----------------------------------------------------------

    def _discover_files(self) -> list[str]:
        """Recursively walk *source_folder* and return matching file paths,
        sorted for deterministic order."""
        paths: list[str] = []
        for root, _dirs, files in os.walk(self.source_folder):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext in self.extensions:
                    paths.append(os.path.join(root, fn))
        return paths

    def _process_file(self, path: str) -> tuple[SourceClip, list[str]]:
        """Run ffprobe on a single file and return ``(SourceClip, warnings)``.

        The returned clip will have *scan_status* set to either ``VALIDATED``
        or ``REJECTED``.
        """
        issues: list[str] = []

        # Basic file-level check
        if not os.path.isfile(path):
            return (
                SourceClip(
                    id=_make_clip_id(path, "0"),
                    path=os.path.abspath(path),
                    file_size_bytes=0,
                    source_hash="",
                    duration=RationalTime(0, 1),
                    width=0, height=0,
                    fps_numerator=0, fps_denominator=1,
                    codec="unknown", bitrate=0,
                    scan_status=ScanStatus.REJECTED,
                    rejection_reason="Not a regular file",
                ),
                issues,
            )

        try:
            ffprobe_data = _run_ffprobe(path)
        except subprocess.TimeoutExpired:
            return (
                SourceClip(
                    id=_make_clip_id(path, "0"),
                    path=os.path.abspath(path),
                    file_size_bytes=0,
                    source_hash="",
                    duration=RationalTime(0, 1),
                    width=0, height=0,
                    fps_numerator=0, fps_denominator=1,
                    codec="unknown", bitrate=0,
                    scan_status=ScanStatus.REJECTED,
                    rejection_reason="ffprobe timed out (> 10 s)",
                ),
                issues,
            )
        except (RuntimeError, json.JSONDecodeError, OSError) as exc:
            return (
                SourceClip(
                    id=_make_clip_id(path, "0"),
                    path=os.path.abspath(path),
                    file_size_bytes=0,
                    source_hash="",
                    duration=RationalTime(0, 1),
                    width=0, height=0,
                    fps_numerator=0, fps_denominator=1,
                    codec="unknown", bitrate=0,
                    scan_status=ScanStatus.REJECTED,
                    rejection_reason=f"ffprobe error: {exc}",
                ),
                issues,
            )

        # Build the SourceClip from ffprobe data
        try:
            clip = _ffprobe_to_source_clip(path, ffprobe_data)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            return (
                SourceClip(
                    id=_make_clip_id(path, "0"),
                    path=os.path.abspath(path),
                    file_size_bytes=0,
                    source_hash="",
                    duration=RationalTime(0, 1),
                    width=0, height=0,
                    fps_numerator=0, fps_denominator=1,
                    codec="unknown", bitrate=0,
                    scan_status=ScanStatus.REJECTED,
                    rejection_reason=f"Metadata parse error: {exc}",
                ),
                issues,
            )

        # ---- Validation checks ---------------------------------------------

        # 1. Minimum duration
        if clip.duration.to_seconds() < MIN_CLIP_DURATION_SECONDS:
            clip.scan_status = ScanStatus.REJECTED
            clip.rejection_reason = (
                f"Duration {clip.duration.to_seconds():.2f}s < "
                f"{MIN_CLIP_DURATION_SECONDS}s minimum"
            )
            return clip, issues

        # 2. Codec support
        if clip.codec.lower() not in SUPPORTED_CODECS:
            clip.scan_status = ScanStatus.REJECTED
            clip.rejection_reason = f"Unsupported codec: {clip.codec}"
            return clip, issues

        # 3. Resolution sanity
        if clip.width == 0 or clip.height == 0:
            clip.scan_status = ScanStatus.REJECTED
            clip.rejection_reason = "Zero frame dimensions detected"
            return clip, issues

        # 4. VFR warning (not a rejection)
        if clip.variable_frame_rate:
            issues.append(
                f"{clip.id}: variable frame-rate detected — will surface in summary"
            )

        clip.scan_status = ScanStatus.VALIDATED
        return clip, issues


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def scan_source_folder(
    source_folder: str,
    extensions: set[str] | None = None,
    parallel: int = 4,
) -> ScanResult:
    """Convenience function: create a ``Scanner`` and run it immediately."""
    return Scanner(source_folder, extensions, parallel).run()
