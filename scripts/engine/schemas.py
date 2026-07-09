"""Schemas — inter-stage data contracts for the AI Video Engine.

All entities are ``dataclass``-based with ``to_dict()`` / ``from_dict()``
serialization so they can be written and read from JSON files on disk.

RationalTime values are serialised as ``{num, den, rate_num, rate_den}``
dictionaries.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .rational_time import RationalTime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIME_KEYS = {"num", "den", "rate_num", "rate_den"}


def _rational_to_dict(rt: RationalTime) -> dict[str, int]:
    return {
        "num": rt.numerator,
        "den": rt.denominator,
        "rate_num": rt.rate_numerator,
        "rate_den": rt.rate_denominator,
    }


def _dict_to_rational(d: dict[str, int]) -> RationalTime:
    return RationalTime(
        numerator=d["num"],
        denominator=d["den"],
        rate_numerator=d["rate_num"],
        rate_denominator=d["rate_den"],
    )


def _is_rational_dict(val: Any) -> bool:
    return isinstance(val, dict) and _TIME_KEYS <= val.keys()


def _to_serializable(obj: Any) -> Any:
    """Recursively convert dataclass/Enum/RationalTime to plain dicts/lists."""
    if isinstance(obj, RationalTime):
        return _rational_to_dict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_serializable(getattr(obj, f.name))
            for f in fields(obj)
            if getattr(obj, f.name) is not None
        }
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _from_serializable(cls: type, data: Any) -> Any:
    """Recursively restore a dataclass tree from plain dicts."""
    if data is None:
        return None
    if isinstance(data, dict) and _is_rational_dict(data):
        return _dict_to_rational(data)
    if is_dataclass(cls) and not isinstance(cls, type):
        # Check for union types in field annotations to handle RationalTime
        field_defs = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, val in data.items():
            if key not in field_defs:
                continue
            ftype = field_defs[key]
            kwargs[key] = _from_serializable(ftype, val)
        return cls(**kwargs)
    if isinstance(data, list):
        # Infer item type from cls if it's a generic alias
        if hasattr(cls, "__origin__") and cls.__origin__ is list:
            item_type = cls.__args__[0] if cls.__args__ else Any
            return [_from_serializable(item_type, item) for item in data]
        return data  # pragma: no cover — fallback
    if isinstance(data, dict):
        return {k: _from_serializable(Any, v) for k, v in data.items()}
    return data


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScanStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"


# ---------------------------------------------------------------------------
# SourceClip — output of the Scanner
# ---------------------------------------------------------------------------

@dataclass
class SourceClip:
    """Metadata for a single source video file discovered by the Scanner.

    Serialised as one entry in ``validated_clips.json``.
    """

    id: str
    """Unique clip identifier (derived from source_hash or file path)."""

    path: str
    """Absolute filesystem path to the source file."""

    file_size_bytes: int
    """File size in bytes."""

    source_hash: str
    """Hash of the first 64 kB of the file (SHA‑256 hex digest)."""

    duration: RationalTime
    """Total duration of the source file."""

    width: int
    """Frame width in pixels."""

    height: int
    """Frame height in pixels."""

    fps_numerator: int
    """Frame-rate numerator (e.g. 30000 for 29.97 fps)."""

    fps_denominator: int
    """Frame-rate denominator (e.g. 1001 for 29.97 fps)."""

    codec: str
    """Video codec string from ffprobe (e.g. ``h264``)."""

    bitrate: int
    """Average bitrate in bits/second."""

    audio_codec: Optional[str] = None
    """Audio codec string (e.g. ``aac``), or ``None`` if no audio track."""

    audio_channels: int = 0
    """Number of audio channels."""

    audio_sample_rate: int = 0
    """Audio sample rate in Hz."""

    variable_frame_rate: bool = False
    """Whether ffprobe reported a variable frame-rate (VFR) flag."""

    scan_status: ScanStatus = ScanStatus.PENDING
    """Validation status after scanning."""

    rejection_reason: Optional[str] = None
    """If *scan_status* is ``REJECTED``, the reason."""

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SourceClip:
        dur = _dict_to_rational(data["duration"])
        return SourceClip(
            id=data["id"],
            path=data["path"],
            file_size_bytes=data["file_size_bytes"],
            source_hash=data["source_hash"],
            duration=dur,
            width=data["width"],
            height=data["height"],
            fps_numerator=data["fps_numerator"],
            fps_denominator=data["fps_denominator"],
            codec=data["codec"],
            bitrate=data["bitrate"],
            audio_codec=data.get("audio_codec"),
            audio_channels=data.get("audio_channels", 0),
            audio_sample_rate=data.get("audio_sample_rate", 0),
            variable_frame_rate=data.get("variable_frame_rate", False),
            scan_status=ScanStatus(data.get("scan_status", "pending")),
            rejection_reason=data.get("rejection_reason"),
        )


# ---------------------------------------------------------------------------
# Segment — one contiguous unit from the Segmenter
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single contiguous segment detected by the Segmenter.

    Each segment has a rational *in_point* and *out_point* relative to the
    start of its source clip.  Later pipeline stages enrich this record with
    vision-analysis and computational-metrics fields.
    """

    id: str
    """Unique segment identifier (e.g. ``seg_<clip_id>_<index>``)."""

    clip_id: str
    """Foreign key → ``SourceClip.id``."""

    index: int
    """Zero-based segment order within the source clip."""

    in_point: RationalTime
    """Start time of the segment relative to the source clip."""

    out_point: RationalTime
    """End time of the segment relative to the source clip."""

    # ---- Populated by the Segmenter ----------------------------------------

    motion_intensity: float = 0.0
    """Aggregate motion intensity for this segment (0‑1)."""

    scene_boundary: bool = False
    """Whether this segment starts at a PySceneDetect scene cut."""

    motion_boundary: bool = False
    """Whether this segment starts at a motion-detected boundary."""

    frame_count: int = 0
    """Number of frame samples extracted for this segment."""

    has_frames: bool = True
    """False if frame extraction failed for this segment."""

    # ---- Populated by the Vision Analyzer ----------------------------------

    scene_type: Optional[str] = None
    subjects: Optional[list[str]] = None
    motion_type: Optional[str] = None
    camera_movement: Optional[str] = None
    composition_quality: Optional[float] = None
    exposure_quality: Optional[float] = None
    focus_quality: Optional[float] = None
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    human_subjects_present: Optional[bool] = None
    tags: Optional[list[str]] = None
    story_potential: Optional[float] = None
    narrative_notes: Optional[str] = None
    ai_analysis_status: str = "pending"
    """``"pending"`` | ``"complete"`` | ``"error"`` | ``"none"``."""

    # ---- Populated by the Computational Metrics component ------------------

    avg_luminance: Optional[float] = None
    """Mean per-frame luminance (0‑1)."""

    laplacian_variance: Optional[float] = None
    """Mean Laplacian variance across frames (focus metric)."""

    motion_magnitude: Optional[float] = None
    """Mean frame-to-frame absolute difference."""

    metrics_status: str = "pending"
    """``"pending"`` | ``"complete"`` | ``"error"`` | ``"none"``."""

    # ---- Populated by the Corpus Aggregator --------------------------------

    composite_quality: Optional[float] = None
    """Weighted average of vision + computational quality scores."""

    is_hero: bool = False
    """True if *composite_quality* > 0.75 *and* duration > 4 s."""

    # ---- Populated by the Audio Analyzer ------------------------------------

    audio_has_audio: bool = True
    """Whether the segment has an audio track."""

    audio_type: Optional[str] = None
    """Detected audio type (``"music"``, ``"speech"``, ``"ambient"``,
    ``"mixed"``, or ``"none"``)."""

    audio_tempo_bpm: Optional[float] = None
    """Estimated tempo in beats per minute, or ``None`` if no audio."""

    audio_loudness_lufs: Optional[float] = None
    """Integrated loudness in LUFS (EBU R128)."""

    audio_silence_markers: Optional[list[dict[str, float]]] = None
    """List of silence regions ``{"start": s, "end": s}`` longer than 1 s."""

    audio_analysis_status: str = "pending"
    """``"pending"`` | ``"complete"`` | ``"error"`` | ``"none"``."""

    analysis_warnings: list[str] = field(default_factory=list)

    # -- convenience properties -----------------------------------------------

    @property
    def duration(self) -> RationalTime:
        return self.out_point - self.in_point

    @property
    def duration_seconds(self) -> float:
        return self.duration.to_seconds()

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Segment:
        # Build kwargs manually to handle RationalTime fields
        kwargs: dict[str, Any] = {}
        for key, val in data.items():
            if key in ("in_point", "out_point", "duration"):
                kwargs[key] = _dict_to_rational(val) if isinstance(val, dict) else val
            elif key in ("subjects", "tags"):
                kwargs[key] = list(val) if val else None
            elif key == "audio_silence_markers":
                kwargs[key] = list(val) if val else None
            elif key == "analysis_warnings":
                kwargs[key] = list(val) if val else []
            else:
                kwargs[key] = val
        # Remove duration from kwargs — it's a computed property
        kwargs.pop("duration", None)
        return Segment(**kwargs)


# ---------------------------------------------------------------------------
# Act — one act within a StoryArc
# ---------------------------------------------------------------------------

@dataclass
class Act:
    """A single act within a proposed story arc."""

    index: int
    """1‑based act number."""

    name: str
    """Short human-readable name (e.g. ``"Establishing"``)."""

    description: str
    """One‑sentence summary of what this act covers."""

    duration_fraction: float
    """Fraction of total runtime this act should consume (sums to 1.0)."""

    emotional_target: str
    """Desired emotional tone for this act (e.g. ``"awe"``, ``"tension"``)."""

    clip_count_target: int
    """Target number of clips/segments for this act."""

    feasibility_score: float
    """0‑1 feasibility given available footage."""

    required_content_elements: list[str] = field(default_factory=list)
    """Scene types or tags the act needs (e.g. ``["landscape_aerial"]``)."""

    suggested_segment_ids: list[str] = field(default_factory=list)
    """Segment IDs the story generator suggested for this act."""

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Act:
        return Act(**data)


# ---------------------------------------------------------------------------
# StoryArc — one of three competing proposals from the Story Generator
# ---------------------------------------------------------------------------

@dataclass
class StoryArc:
    """A complete story arc proposal with three acts."""

    id: str
    """Arc identifier (``"arc_a"``, ``"arc_b"``, ``"arc_c"``)."""

    name: str
    """Human-readable name (e.g. ``"Classic 3-Act"``)."""

    theme: str
    """One‑sentence theme description."""

    acts: list[Act]
    """Exactly three acts."""

    arc_type: str = "unknown"
    """``"classic"`` | ``"theme_focused"`` | ``"alternate"``."""

    self_critique: str = ""
    """AI self-assessment of this arc's strengths and weaknesses."""

    overall_feasibility: float = 0.0
    """Aggregate 0‑1 feasibility score for the arc."""

    gaps: list[str] = field(default_factory=list)
    """Identified gaps between what the arc needs and what footage exists."""

    merged_from: Optional[list[str]] = None
    """If this arc was created via checkpoint merge, the source arc IDs."""

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> StoryArc:
        acts = [Act.from_dict(a) for a in data["acts"]]
        kwargs = {k: v for k, v in data.items() if k != "acts"}
        return StoryArc(**kwargs, acts=acts)


# ---------------------------------------------------------------------------
# SequenceItem — one edit decision in the timeline
# ---------------------------------------------------------------------------

@dataclass
class SequenceItem:
    """A single edit decision: which segment to use and where to place it."""

    segment_id: str
    """Foreign key → ``Segment.id``."""

    clip_id: str
    """Foreign key → ``SourceClip.id``."""

    act_index: int
    """1‑based act this item belongs to."""

    source_in: RationalTime
    """In-point within the source file (frame‑accurate)."""

    source_out: RationalTime
    """Out-point within the source file (frame‑accurate)."""

    timeline_position: RationalTime
    """Position of this clip on the timeline (offset from start)."""

    duration: RationalTime
    """Duration of this clip on the timeline."""

    transition: str = "cross_dissolve"
    """Transition type for the *outgoing* edge (default ``cross_dissolve``)."""

    transition_duration: Optional[RationalTime] = None
    """Duration of the transition (default 12 frames at 24 fps)."""

    reason: str = ""
    """Why this segment was selected (for traceability)."""

    confidence: float = 0.5
    """Selection confidence 0‑1."""

    marker_value: Optional[str] = None
    """Optional marker label for this clip position."""

    marker_note: Optional[str] = None
    """Optional marker note with additional context."""

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SequenceItem:
        kwargs: dict[str, Any] = {}
        for key, val in data.items():
            if key in ("source_in", "source_out", "timeline_position", "duration",
                       "transition_duration"):
                kwargs[key] = _dict_to_rational(val) if isinstance(val, dict) else val
            else:
                kwargs[key] = val
        return SequenceItem(**kwargs)


# ---------------------------------------------------------------------------
# EditDecision — full timeline output from the Clip Selector
# ---------------------------------------------------------------------------

@dataclass
class EditDecision:
    """The complete ordered timeline produced by the Clip Selector."""

    project_title: str
    """Project title for identification."""

    timeline_rate: RationalTime
    """Frame rate of the output timeline."""

    target_duration_seconds: float
    """Target duration of the final edit in seconds."""

    items: list[SequenceItem]
    """Ordered list of clips on the timeline."""

    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_duration(self) -> RationalTime:
        if not self.items:
            return RationalTime(0, 1, 0, 1)
        last = self.items[-1]
        return last.timeline_position + last.duration

    @property
    def clip_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EditDecision:
        tr = _dict_to_rational(data["timeline_rate"])
        items = [SequenceItem.from_dict(it) for it in data["items"]]
        created = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        return EditDecision(
            project_title=data["project_title"],
            timeline_rate=tr,
            target_duration_seconds=data["target_duration_seconds"],
            items=items,
            created_at=created,
        )


# ---------------------------------------------------------------------------
# Corpus — aggregated catalog of all footage
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    """Definitive catalog of all available footage with merged analysis.

    Produced by the Corpus Aggregator from per-segment analysis and
    computational metrics.
    """

    project_title: str
    """Project title."""

    segments: list[Segment]
    """All segments from all clips, each with merged analysis data."""

    total_segments: int = 0
    total_duration_seconds: float = 0.0
    hero_segments: list[Segment] = field(default_factory=list)
    scene_type_distribution: dict[str, int] = field(default_factory=dict)
    quality_distribution: dict[str, int] = field(default_factory=dict)
    time_of_day_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Corpus:
        segments = [Segment.from_dict(s) for s in data["segments"]]
        hero = [Segment.from_dict(s) for s in data.get("hero_segments", [])]
        created = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data else datetime.now()
        )
        return Corpus(
            project_title=data["project_title"],
            segments=segments,
            hero_segments=hero,
            total_segments=data.get("total_segments", len(segments)),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            scene_type_distribution=data.get("scene_type_distribution", {}),
            quality_distribution=data.get("quality_distribution", {}),
            time_of_day_distribution=data.get("time_of_day_distribution", {}),
            warnings=data.get("warnings", []),
            created_at=created,
        )


# ---------------------------------------------------------------------------
# ProjectConfig — user-facing project configuration
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Project-level configuration consumed by the engine.

    Serialised as ``project_config.json``.
    """

    title: str
    """Human-readable project title."""

    source_folder: str
    """Absolute path to the folder containing source video files."""

    output_folder: str
    """Absolute path where pipeline outputs will be written."""

    target_duration_seconds: float = 180.0
    """Target duration of the final edit in seconds."""

    timeline_fps: RationalTime = field(default_factory=lambda: RationalTime(0, 1, 24, 1))
    """Frame rate of the output timeline (default 24 fps)."""

    style_profile: str = "default"
    """Name of the style profile to use for story generation."""

    parallel_workers: int = 5
    """Number of parallel workers for vision analysis / metrics."""

    extensions: list[str] = field(default_factory=lambda: [".mov", ".mp4", ".mxf", ".m4v"])
    """File extensions to scan."""

    max_segments_per_file: int = 10
    """Maximum segments to select from any single source file."""

    keep_frames: bool = False
    """If True, keep frame samples after a successful run."""

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ProjectConfig:
        tfps_data = data.get("timeline_fps")
        tfps = _dict_to_rational(tfps_data) if isinstance(tfps_data, dict) else RationalTime(0, 1, 24, 1)
        return ProjectConfig(
            title=data["title"],
            source_folder=data["source_folder"],
            output_folder=data["output_folder"],
            target_duration_seconds=data.get("target_duration_seconds", 180.0),
            timeline_fps=tfps,
            style_profile=data.get("style_profile", "default"),
            parallel_workers=data.get("parallel_workers", 5),
            extensions=data.get("extensions", [".mov", ".mp4", ".mxf", ".m4v"]),
            max_segments_per_file=data.get("max_segments_per_file", 10),
            keep_frames=data.get("keep_frames", False),
        )


# ---------------------------------------------------------------------------
# RunnerManifest — per-run metadata
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """Timing and result information for one pipeline stage."""

    status: str = "pending"
    """``"pending"`` | ``"running"`` | ``"complete"`` | ``"failed"`` | ``"skipped"``."""

    duration_seconds: float = 0.0
    """Wall-clock time spent in this stage."""

    error: Optional[str] = None
    """Error message if the stage failed."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Stage-specific metrics (clips_found, segments_analyzed, etc.)."""


@dataclass
class RunnerManifest:
    """Per-run metadata, serialised as ``.manifest.json``."""

    run_id: str
    """Timestamp-based run identifier (``YYYYMMDD_HHMMSS``)."""

    project_title: str
    """Project title from config."""

    created_at: datetime = field(default_factory=datetime.now)

    stages: dict[str, StageRecord] = field(default_factory=lambda: {
        name: StageRecord()
        for name in [
            "scanner",
            "segmenter",
            "vision_analysis",
            "computational_metrics",
            "corpus_assembly",
            "story_generation",
            "checkpoint_1",
            "clip_selection",
            "fcpxml_generation",
            "narration",
            "review_package",
        ]
    })

    models_used: dict[str, str] = field(default_factory=lambda: {
        "vision": "gemini-2.5-flash",
        "story": "deepseek-v4-pro",
        "narration": "deepseek-v4-pro",
    })

    api_cost_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunnerManifest:
        stages = {}
        for name, sdata in data.get("stages", {}).items():
            stages[name] = StageRecord(
                status=sdata.get("status", "pending"),
                duration_seconds=sdata.get("duration_seconds", 0.0),
                error=sdata.get("error"),
                extra=sdata.get("extra", {}),
            )
        created = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data else datetime.now()
        )
        return RunnerManifest(
            run_id=data["run_id"],
            project_title=data["project_title"],
            created_at=created,
            stages=stages,
            models_used=data.get("models_used", {}),
            api_cost_estimate=data.get("api_cost_estimate", 0.0),
        )
