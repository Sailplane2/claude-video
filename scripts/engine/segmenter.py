"""Segmenter — detect scene and motion boundaries in source clips.

Uses PySceneDetect for content-aware scene detection and supplements it
with frame-difference motion analysis to identify additional cut points.
Outputs segment boundaries with min/max duration enforcement and extracts
JPEG frame samples per segment.

If PySceneDetect is not installed, falls back to uniform 30-second
segmentation with a warning.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

from .rational_time import RATE_23_976, RationalTime
from .schemas import Segment, SourceClip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SEGMENT_DURATION_SECONDS = 3.0
MAX_SEGMENT_DURATION_SECONDS = 60.0
MOTION_BOUNDARY_PROMOTION_WINDOW = 30.0  # seconds
MOTION_ANALYSIS_FPS = 0.5  # fps for motion sampling (was 1.0 — halved for 4K perf)
MOTION_BOUNDARY_THRESHOLD = 0.30  # normalised histogram-diff threshold
MOTION_HIGH_THRESHOLD = 0.60  # motion intensity above which frame rate increases

# Frame extraction rates
FRAME_EXTRACT_INTERVAL_LOW_MOTION = 5.0  # seconds per frame for low motion
FRAME_EXTRACT_FPS_HIGH_MOTION = 1.0  # fps for high motion (> 0.6)
MIN_FRAMES_PER_SEGMENT = 1
MAX_FRAMES_PER_SEGMENT = 10  # 4K H.264 decode is heavy — keep it tight

FFMPEG_FRAME_EXTRACT_TIMEOUT = 15  # per-segment, not per-clip

# ---------------------------------------------------------------------------
# PySceneDetect check & wrapper
# ---------------------------------------------------------------------------

_HAS_SCENEDETECT: bool = False
try:
    from scenedetect import (  # type: ignore[import-untyped]
        AdaptiveDetector,
        SceneManager,
        open_video,
    )
    from scenedetect.frame_timecode import FrameTimecode  # type: ignore[import-untyped]

    _HAS_SCENEDETECT = True
except ImportError:
    _HAS_SCENEDETECT = False


def _find_ffmpeg() -> str:
    """Locate the ``ffmpeg`` binary."""
    for candidate in ("ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
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
        "ffmpeg not found.  Install ffmpeg: `brew install ffmpeg`"
    )


_COMPUTED_FFMPEG: str | None = None


def _ffmpeg_path() -> str:
    global _COMPUTED_FFMPEG
    if _COMPUTED_FFMPEG is None:
        _COMPUTED_FFMPEG = _find_ffmpeg()
    return _COMPUTED_FFMPEG


# ---------------------------------------------------------------------------
# Motion analysis helpers
# ---------------------------------------------------------------------------

def _extract_frames_ffmpeg(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = 300,
    seek_offset: float = 0.0,
    duration: float | None = None,
) -> list[str]:
    """Extract JPEG frames from *video_path* at *fps* using ffmpeg.

    Returns sorted list of frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%04d.jpg")

    cmd = [_ffmpeg_path(), "-y"]
    if seek_offset > 0:
        cmd.extend(["-ss", str(seek_offset)])
    cmd.extend([
        "-i", video_path,
        "-vf", f"fps={fps},scale=320:-1",
        "-frames:v", str(max_frames),
        "-qscale:v", "5",
        pattern,
    ])

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=FFMPEG_FRAME_EXTRACT_TIMEOUT)

    # Collect produced frames
    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )
    return frames


def _compute_histogram_diff(path_a: str, path_b: str) -> float:
    """Compute normalised histogram difference between two JPEG frames.

    Returns 0.0 (identical) to 1.0 (completely different).
    """
    if Image is None:
        return 0.0  # can't compute without Pillow

    try:
        im_a = Image.open(path_a).convert("L")  # grayscale
        im_b = Image.open(path_b).convert("L")
        hist_a = im_a.histogram()
        hist_b = im_b.histogram()
        # Chi-squared distance
        diff = sum(
            ((a - b) ** 2) / (a + b + 1)
            for a, b in zip(hist_a, hist_b)
        )
        # Normalise to 0-1
        n_pixels = im_a.size[0] * im_a.size[1]
        max_diff = 256.0 * n_pixels  # rough upper bound
        return min(diff / max_diff, 1.0)
    except Exception:
        return 0.0


def _detect_motion_boundaries(
    frames: list[str],
    threshold: float = MOTION_BOUNDARY_THRESHOLD,
    window_size: int = 5,
) -> list[int]:
    """Detect motion boundaries in a list of frame paths.

    Computes histogram differences between consecutive frames and identifies
    significant changes using a rolling-window average.  Returns 0-based
    frame indices where boundaries occur.
    """
    if len(frames) < 3:
        return []

    diffs: list[float] = []
    for i in range(1, len(frames)):
        d = _compute_histogram_diff(frames[i - 1], frames[i])
        diffs.append(d)

    # Rolling mean
    rolling: list[float] = []
    half_w = window_size // 2
    for i in range(len(diffs)):
        lo = max(0, i - half_w)
        hi = min(len(diffs), i + half_w + 1)
        rolling.append(sum(diffs[lo:hi]) / (hi - lo))

    # Boundaries where diff significantly exceeds rolling average
    boundaries: list[int] = []
    for i in range(len(diffs)):
        if diffs[i] > threshold and diffs[i] > rolling[i] * 1.5:
            boundaries.append(i + 1)  # frame index after the diff

    return boundaries


# ---------------------------------------------------------------------------
# Boundary merging
# ---------------------------------------------------------------------------

def _merge_boundaries(
    scene_boundaries: list[float],
    motion_boundaries: list[float],
    total_duration: float,
    min_duration: float = MIN_SEGMENT_DURATION_SECONDS,
    max_duration: float = MAX_SEGMENT_DURATION_SECONDS,
    motion_promotion_window: float = MOTION_BOUNDARY_PROMOTION_WINDOW,
) -> list[float]:
    """Merge scene and motion boundary lists into a single set of segment
    breakpoints (in seconds from start).

    Rules:
    - Scene boundaries are always kept.
    - Motion boundaries are kept only if no scene boundary within
      *motion_promotion_window* seconds (±).
    - Final segments must be >= *min_duration* (shorts merged rightward).
    - Final segments must be <= *max_duration* (overs split by timestamp).
    """
    all_boundaries: set[float] = set(scene_boundaries)

    for mb in motion_boundaries:
        nearby = any(abs(mb - sb) <= motion_promotion_window for sb in scene_boundaries)
        if not nearby:
            all_boundaries.add(mb)

    # Sort and enforce end-of-video
    sorted_b = sorted(b for b in all_boundaries if 0 < b < total_duration)
    sorted_b = [0.0] + sorted_b + [total_duration]

    # Enforce minimum duration — merge short segments with the next one
    merged = [sorted_b[0]]
    for i in range(1, len(sorted_b) - 1):
        if sorted_b[i] - merged[-1] < min_duration:
            # Skip this boundary (merge with next)
            continue
        merged.append(sorted_b[i])
    merged.append(sorted_b[-1])

    # Handle last segment if too short (merge into previous)
    if len(merged) >= 2 and (merged[-1] - merged[-2]) < min_duration:
        merged.pop(-2)

    # Enforce maximum duration — insert split points
    final = [merged[0]]
    for i in range(1, len(merged)):
        seg_dur = merged[i] - final[-1]
        if seg_dur > max_duration:
            num_splits = math.ceil(seg_dur / max_duration)
            for j in range(1, num_splits):
                split_time = final[-1] + seg_dur * (j / num_splits)
                final.append(round(split_time, 6))
        final.append(merged[i])

    return final


# ---------------------------------------------------------------------------
# Frame extraction per segment
# ---------------------------------------------------------------------------

def _extract_segment_frames(
    video_path: str,
    output_dir: str,
    segment_start: float,
    segment_duration: float,
    motion_intensity: float,
) -> int:
    """Extract frame samples for a single segment.

    Returns the number of frames extracted.
    """
    if segment_duration < 0.1:
        return 0

    if segment_duration < 5.0:
        interval = segment_duration - 0.01  # extract 1 frame near start
    elif motion_intensity > MOTION_HIGH_THRESHOLD:
        interval = 1.0  # 1 fps for high motion
    else:
        interval = FRAME_EXTRACT_INTERVAL_LOW_MOTION

    frame_count = max(MIN_FRAMES_PER_SEGMENT, int(segment_duration / interval))
    frame_count = min(frame_count, MAX_FRAMES_PER_SEGMENT)

    pattern = os.path.join(output_dir, "frame_%04d.jpg")
    os.makedirs(output_dir, exist_ok=True)

    try:
        subprocess.run(
            [
                _ffmpeg_path(), "-y",
                "-ss", str(segment_start),
                "-i", video_path,
                "-vf", f"fps=1/{interval},scale=320:-1",
                "-frames:v", str(frame_count),
                "-qscale:v", "5",
                pattern,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=FFMPEG_FRAME_EXTRACT_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0

    extracted = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    return len(extracted)


# ---------------------------------------------------------------------------
# Segmenter class
# ---------------------------------------------------------------------------

class Segmenter:
    """Detect scene and motion boundaries, produce segments.

    Parameters
    ----------
    output_dir:
        Root directory for writing ``segment_map.json`` and frame samples.
    clip:
        The ``SourceClip`` to segment.
    frames_dir:
        Subdirectory within *output_dir* where frame samples are stored
        (default ``frames/<clip_id>/``).
    use_scenedetect:
        Whether to attempt PySceneDetect (default ``True``).  Falls back
        automatically if the library is not installed.
    """

    def __init__(
        self,
        output_dir: str,
        clip: SourceClip,
        frames_dir: str | None = None,
        use_scenedetect: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.clip = clip
        self.use_scenedetect = use_scenedetect and _HAS_SCENEDETECT
        self.frames_dir = frames_dir or os.path.join(
            output_dir, "frames", clip.id
        )

    # ---- Public API --------------------------------------------------------

    def run(self) -> list[Segment]:
        """Segment the clip and extract frame samples.

        Returns the list of ``Segment`` objects for this clip.
        """
        start_time = time.time()
        video_path = self.clip.path
        total_duration = self.clip.duration.to_seconds()

        if total_duration <= 0:
            logger.warning("Clip %s has zero duration, skipping", self.clip.id)
            return []

        logger.info("Segmenting %s (%.1fs)", self.clip.id, total_duration)

        # Step 1 & 2: detect boundaries
        scene_boundaries = self._detect_scene_boundaries(video_path)
        if self.use_scenedetect:
            motion_boundaries, motion_intensities = self._detect_motion_boundaries(
                video_path, total_duration
            )
        else:
            motion_boundaries = set()
            motion_intensities = []

        # Step 3: merge
        boundaries = _merge_boundaries(
            scene_boundaries, motion_boundaries, total_duration
        )

        # Build Segment objects
        clip_rate = RationalTime(0, 1, self.clip.fps_numerator, self.clip.fps_denominator)
        segments: list[Segment] = []

        for i in range(len(boundaries) - 1):
            in_sec = boundaries[i]
            out_sec = boundaries[i + 1]
            seg_duration = out_sec - in_sec

            if seg_duration < 0.001:
                continue

            in_rt = RationalTime.from_seconds(in_sec, clip_rate)
            out_rt = RationalTime.from_seconds(out_sec, clip_rate)

            seg_id = f"seg_{self.clip.id}_{i:04d}"
            seg_motion = _estimate_segment_motion(
                motion_intensities, in_sec, out_sec
            )

            segment = Segment(
                id=seg_id,
                clip_id=self.clip.id,
                index=i,
                in_point=in_rt,
                out_point=out_rt,
                motion_intensity=seg_motion,
                scene_boundary=(in_sec in scene_boundaries or in_sec == 0),
                motion_boundary=(in_sec in motion_boundaries),
            )
            segments.append(segment)

        # Step 4: extract frames per segment (only when using scene detection)
        if self.use_scenedetect:
            os.makedirs(self.frames_dir, exist_ok=True)
            for seg in segments:
                seg_start = seg.in_point.to_seconds()
                seg_dur = seg.out_point.to_seconds() - seg_start
                seg_frames_dir = os.path.join(self.frames_dir, seg.id)
                frame_count = _extract_segment_frames(
                    video_path, seg_frames_dir, seg_start, seg_dur, seg.motion_intensity,
                )
                seg.frame_count = frame_count
                seg.has_frames = frame_count > 0

        elapsed = time.time() - start_time
        logger.info(
            "  → %d segments from %s in %.1fs",
            len(segments), self.clip.id, elapsed,
        )
        return segments

    # ---- Internal: scene detection -----------------------------------------

    def _detect_scene_boundaries(self, video_path: str) -> list[float]:
        """Run PySceneDetect and return scene boundary times in seconds.

        Returns empty list if PySceneDetect is unavailable or fails.
        """
        if not self.use_scenedetect:
            logger.info(
                "Scene detection skipped. "
                "Falling back to uniform 30 s segmentation for %s",
                self.clip.id,
            )
            return []

        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
        except Exception as exc:
            logger.warning(
                "PySceneDetect failed for %s: %s. Using motion only.",
                self.clip.id, exc,
            )
            return []

        boundaries: set[float] = set()
        for start_tc, end_tc in scene_list:
            boundaries.add(float(start_tc.get_seconds()))
            boundaries.add(float(end_tc.get_seconds()))

        # Remove 0 and total_duration — _merge_boundaries adds those
        boundaries.discard(0.0)
        total_dur = self.clip.duration.to_seconds()
        boundaries.discard(total_dur)

        return sorted(boundaries)

    # ---- Internal: motion detection ----------------------------------------

    def _detect_motion_boundaries(
        self, video_path: str, total_duration: float
    ) -> tuple[list[float], list[float]]:
        """Extract frames at *MOTION_ANALYSIS_FPS* and detect motion boundaries.

        Returns ``(boundary_times, per_second_intensities)``.

        Per-second intensities are the raw histogram-diff values indexed by
        integer second offset.
        """
        if Image is None:
            logger.warning("Pillow not installed; skipping motion analysis")
            return [], [0.0] * int(total_duration)

        tmpdir = tempfile.mkdtemp(prefix="motion_")

        try:
            max_frames = min(int(total_duration * MOTION_ANALYSIS_FPS) + 1, 600)
            frames = _extract_frames_ffmpeg(
                video_path, tmpdir, fps=MOTION_ANALYSIS_FPS,
                max_frames=max_frames,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Motion frame extraction failed: %s", exc)
            return [], [0.0] * int(total_duration)

        if len(frames) < 3:
            return [], [0.0] * int(total_duration)

        # Compute per-second histogram differences
        intensities: list[float] = [0.0]
        for i in range(1, len(frames)):
            diff = _compute_histogram_diff(frames[i - 1], frames[i])
            intensities.append(diff)

        # Pad or truncate to match total_duration
        expected = int(total_duration)
        while len(intensities) < expected:
            intensities.append(0.0)
        intensities = intensities[:expected]

        # Detect motion boundaries
        boundary_indices = _detect_motion_boundaries(frames, threshold=0.30)
        boundary_times = [
            idx / MOTION_ANALYSIS_FPS for idx in boundary_indices
        ]

        # Cleanup temp directory
        try:
            for f in frames:
                os.remove(f)
            os.rmdir(tmpdir)
        except OSError:
            pass

        return boundary_times, intensities


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _estimate_segment_motion(
    intensities: list[float], in_sec: float, out_sec: float
) -> float:
    """Average motion intensity over a segment's time window."""
    lo = max(0, int(in_sec))
    hi = min(len(intensities), int(out_sec) + 1)
    if hi <= lo:
        return 0.0
    return sum(intensities[lo:hi]) / (hi - lo)


def segment_clip(
    clip: SourceClip,
    output_dir: str,
    use_scenedetect: bool = True,
) -> list[Segment]:
    """Convenience: segment a single clip and return its segments."""
    segmenter = Segmenter(
        output_dir=output_dir,
        clip=clip,
        use_scenedetect=use_scenedetect,
    )
    return segmenter.run()


def segment_clips(
    clips: list[SourceClip],
    output_dir: str,
    use_scenedetect: bool = True,
) -> list[Segment]:
    """Segment multiple clips and return the aggregated segment list."""
    all_segments: list[Segment] = []
    for clip in clips:
        if clip.scan_status.value != "validated":
            continue
        segments = segment_clip(clip, output_dir, use_scenedetect)
        all_segments.extend(segments)
    return all_segments
