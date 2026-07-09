"""Audio Analyzer — extract audio features from video segments.

Uses ``ffmpeg`` to extract segment audio and ``librosa`` (when available) for
tempo, RMS energy, and spectral analysis.  Detects silence regions longer than
1 second as candidate edit points.

If ``librosa`` is not installed or the clip has no audio track, the analyzer
falls back gracefully to ffprobe-level metadata and ffmpeg loudnorm statistics.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

from .rational_time import RationalTime
from .schemas import Segment, SourceClip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

_HAS_LIBROSA: bool = False
try:
    import librosa  # type: ignore[import-untyped]

    _HAS_LIBROSA = True
except ImportError:
    pass

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SILENCE_MIN_DURATION_SECONDS: float = 1.0
"""Silence regions at least this long are reported as edit-point candidates."""

FFMPEG_AUDIO_EXTRACT_TIMEOUT: int = 30
"""Per-segment timeout for ffmpeg audio extraction."""

AUDIO_EXTRACT_SAMPLE_RATE: int = 22050
"""Sample rate used for audio extraction (librosa expects ~22 kHz)."""

LOUDNESS_ANALYSIS_TIMEOUT: int = 15
"""Timeout for ffmpeg loudnorm analysis."""

# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

_COMPUTED_FFMPEG: str | None = None
_COMPUTED_FFPROBE: str | None = None


def _find_binary(name: str) -> str:
    """Locate an ffmpeg/ffprobe binary."""
    locations = (
        name,
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
    )
    for candidate in locations:
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
        f"{name} not found.  Install ffmpeg: `brew install ffmpeg`"
    )


def _ffmpeg_path() -> str:
    global _COMPUTED_FFMPEG
    if _COMPUTED_FFMPEG is None:
        _COMPUTED_FFMPEG = _find_binary("ffmpeg")
    return _COMPUTED_FFMPEG


def _ffprobe_path() -> str:
    global _COMPUTED_FFPROBE
    if _COMPUTED_FFPROBE is None:
        _COMPUTED_FFPROBE = _find_binary("ffprobe")
    return _COMPUTED_FFPROBE


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _extract_segment_audio(
    clip_path: str,
    segment_start: float,
    segment_duration: float,
    output_path: str,
    sample_rate: int = AUDIO_EXTRACT_SAMPLE_RATE,
) -> bool:
    """Extract a segment's audio to a WAV file using ffmpeg.

    Returns ``True`` if extraction succeeded.
    """
    if segment_duration < 0.05:
        return False

    try:
        cmd = [
            _ffmpeg_path(), "-y",
            "-ss", str(segment_start),
            "-i", clip_path,
            "-t", str(segment_duration),
            "-vn",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "wav",
            output_path,
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=FFMPEG_AUDIO_EXTRACT_TIMEOUT,
        )
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 44
    except (subprocess.TimeoutExpired, OSError):
        return False


def _has_audio_track(clip_path: str) -> bool:
    """Quick check via ffprobe: does this file have an audio stream?"""
    try:
        cmd = [
            _ffprobe_path(),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            clip_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return False
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        return len(streams) > 0
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return True  # conservative: assume audio if we can't check


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def _detect_silence(
    audio_path: str,
    min_duration: float = SILENCE_MIN_DURATION_SECONDS,
    silence_threshold_dbfs: float = -40.0,
) -> list[dict[str, float]]:
    """Detect silence regions in a WAV file using ffmpeg's silencedetect filter.

    Returns a list of ``{"start": float, "end": float}`` dicts for each
    silence region longer than *min_duration*.
    """
    try:
        cmd = [
            _ffmpeg_path(),
            "-i", audio_path,
            "-af",
            f"silencedetect=noise={silence_threshold_dbfs}dB:d={min_duration}",
            "-f", "null",
            "-",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=LOUDNESS_ANALYSIS_TIMEOUT,
        )
        stderr = proc.stderr
    except (subprocess.TimeoutExpired, OSError):
        return []

    markers: list[dict[str, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[1].strip())
            except (ValueError, IndexError):
                continue
        elif "silence_end:" in line and start is not None:
            try:
                parts = line.split()
                end = float(parts[parts.index("silence_end:") + 1].rstrip("|"))
                markers.append({"start": round(start, 3), "end": round(end, 3)})
                start = None
            except (ValueError, IndexError):
                continue

    return markers


# ---------------------------------------------------------------------------
# Loudness (EBU R128)
# ---------------------------------------------------------------------------

def _measure_loudness(audio_path: str) -> float | None:
    """Measure integrated loudness (LUFS) using ffmpeg's loudnorm filter.

    Returns ``None`` if measurement fails.
    """
    try:
        cmd = [
            _ffmpeg_path(),
            "-i", audio_path,
            "-af", "loudnorm=I=-23:LRA=7:tp=-1:print_format=json",
            "-f", "null",
            "-",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=LOUDNESS_ANALYSIS_TIMEOUT,
        )
        stderr = proc.stderr
    except (subprocess.TimeoutExpired, OSError):
        return None

    # loudnorm prints JSON to stderr after processing
    try:
        # Find the JSON block in stderr
        lines = stderr.splitlines()
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "{":
                json_start = i
                break
        if json_start == -1:
            return None
        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)
        return float(data.get("input_i", 0.0))
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Librosa analysis
# ---------------------------------------------------------------------------

def _librosa_analyze(audio_path: str, sr: int) -> dict[str, Any]:
    """Run librosa analysis on an audio file.

    Returns a dict with *tempo_bpm* and *energy_mean* keys (plus raw values
    for debugging).  Returns empty dict on failure.
    """
    if not _HAS_LIBROSA or np is None:
        return {}

    try:
        y, _sr = librosa.load(audio_path, sr=sr, mono=True)

        if len(y) < sr:  # less than 1 second
            return {"tempo_bpm": None, "energy_mean": 0.0}

        # Tempo (BPM) via onset correlation
        tempo, _beats = librosa.beat.beat_track(y=y, sr=_sr)
        tempo_bpm = float(np.round(tempo)) if not np.isnan(tempo) else None

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))

        # Spectral centroid (rough "brightness" indicator)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=_sr)[0]
        centroid_mean = float(np.mean(spectral_centroid))

        # Zero-crossing rate (speech vs. music heuristic)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        return {
            "tempo_bpm": tempo_bpm,
            "energy_mean": energy_mean,
            "spectral_centroid_mean": centroid_mean,
            "zero_crossing_rate_mean": zcr_mean,
        }
    except Exception as exc:
        logger.debug("librosa analysis failed: %s", exc)
        return {}


def _classify_audio_type(librosa_results: dict[str, Any]) -> str:
    """Classify the audio type from librosa features.

    Returns one of ``"music"``, ``"speech"``, ``"ambient"``, ``"mixed"``,
    or ``"none"``.
    """
    tempo = librosa_results.get("tempo_bpm")
    zcr = librosa_results.get("zero_crossing_rate_mean", 0.0)
    energy = librosa_results.get("energy_mean", 0.0)

    if tempo is None and energy < 0.01:
        return "none"
    if energy < 0.005:
        return "ambient"

    # Speech typically has high ZCR (> 0.1) and variable tempo
    if zcr > 0.08 and (tempo is None or tempo < 80 or tempo > 180):
        return "speech"

    # Music has consistent tempo in 60-180 BPM range
    if tempo is not None and 60 <= tempo <= 180:
        return "music"

    # Mixed: significant energy but no clear tempo
    if energy > 0.02:
        return "mixed"

    return "ambient"


# ---------------------------------------------------------------------------
# AudioAnalyzer class
# ---------------------------------------------------------------------------

class AudioAnalyzer:
    """Extract audio features from video segments.

    For each segment, the analyzer:

    1. Extracts segment audio to a temporary WAV file via ffmpeg.
    2. Measures integrated loudness (LUFS) via ffmpeg loudnorm.
    3. Detects silence regions longer than 1 s (edit-point candidates).
    4. If librosa is available, computes tempo (BPM) and RMS energy.
    5. Classifies the audio type (music / speech / ambient / mixed / none).
    6. Stores results directly on the ``Segment``'s audio fields.

    Parameters
    ----------
    clip:
        The ``SourceClip`` being analyzed (used for audio-track check).
    tmp_root:
        Optional root directory for temporary audio files.  Defaults to
        system temp dir.
    """

    def __init__(
        self,
        clip: SourceClip,
        tmp_root: str | None = None,
    ) -> None:
        self.clip = clip
        self.tmp_root = tmp_root or tempfile.gettempdir()
        self._has_audio: bool = clip.audio_codec is not None or (
            clip.audio_channels > 0
        )

    # ---- Public API --------------------------------------------------------

    def analyze_segment(
        self,
        clip_path: str,
        segment: Segment,
    ) -> dict[str, Any]:
        """Extract audio features for a single segment.

        Parameters
        ----------
        clip_path:
            Absolute path to the source video file.
        segment:
            The ``Segment`` to analyze.  Results are stored directly on the
            segment's audio fields, and also returned as a dict.

        Returns
        -------
        dict
            Audio features: ``tempo_bpm``, ``energy_mean``, ``silence_markers``,
            ``loudness_lufs``, ``type``.
        """
        if not self._has_audio:
            segment.audio_has_audio = False
            segment.audio_type = "none"
            segment.audio_analysis_status = "none"
            return self._empty_result()

        if clip_path is None or not os.path.isfile(str(clip_path)):
            segment.audio_analysis_status = "error"
            return self._empty_result()

        seg_start = segment.in_point.to_seconds()
        seg_duration = (segment.out_point - segment.in_point).to_seconds()

        if seg_duration < 0.05:
            segment.audio_has_audio = False
            segment.audio_type = "none"
            segment.audio_analysis_status = "none"
            return self._empty_result()

        # Extract audio to a temp WAV
        tmp_wav = os.path.join(
            self.tmp_root,
            f"audio_{segment.id}_{os.urandom(4).hex()}.wav",
        )
        try:
            ok = _extract_segment_audio(
                clip_path, seg_start, seg_duration, tmp_wav,
            )
            if not ok:
                segment.audio_analysis_status = "error"
                return self._empty_result()

            # Measure loudness
            loudness = _measure_loudness(tmp_wav)
            segment.audio_loudness_lufs = loudness

            # Detect silence
            silence = _detect_silence(tmp_wav)
            segment.audio_silence_markers = silence if silence else None

            # Librosa analysis (when available)
            librosa_results = _librosa_analyze(tmp_wav, AUDIO_EXTRACT_SAMPLE_RATE)
            tempo = librosa_results.get("tempo_bpm")
            energy = librosa_results.get("energy_mean", 0.0)

            segment.audio_tempo_bpm = tempo

            # Classify audio type
            audio_type = _classify_audio_type(librosa_results)
            segment.audio_type = audio_type
            segment.audio_has_audio = audio_type != "none"
            segment.audio_analysis_status = "complete"

            result: dict[str, Any] = {
                "tempo_bpm": tempo,
                "energy_mean": energy,
                "silence_markers": silence,
                "loudness_lufs": loudness,
                "type": audio_type,
            }
            return result

        except Exception as exc:
            logger.warning(
                "Audio analysis failed for segment %s: %s",
                segment.id, exc,
            )
            segment.audio_analysis_status = "error"
            return self._empty_result()

        finally:
            # Cleanup temp WAV
            try:
                if os.path.isfile(tmp_wav):
                    os.remove(tmp_wav)
            except OSError:
                pass

    def analyze_segments(
        self,
        clip_path: str,
        segments: list[Segment],
    ) -> list[Segment]:
        """Analyze audio for all segments of a clip in batch.

        Returns the list of segments with audio fields populated.
        """
        for seg in segments:
            self.analyze_segment(clip_path, seg)
        return segments

    # ---- Internal ----------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "tempo_bpm": None,
            "energy_mean": 0.0,
            "silence_markers": [],
            "loudness_lufs": None,
            "type": "none",
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_clip_audio(
    clip: SourceClip,
    segments: list[Segment],
    tmp_root: str | None = None,
) -> list[Segment]:
    """Convenience: create an ``AudioAnalyzer`` and analyze all segments."""
    analyzer = AudioAnalyzer(clip=clip, tmp_root=tmp_root)
    return analyzer.analyze_segments(clip.path, segments)
