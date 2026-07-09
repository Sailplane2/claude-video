"""AI Video Engine - Foundation Layer.

Core data types, schemas, and analysis components for building
AI-assisted video editing pipelines.
"""

from .rational_time import (
    RATE_23_976,
    RATE_24,
    RATE_25,
    RATE_29_97,
    RATE_30,
    RATE_48,
    RATE_50,
    RATE_59_94,
    RATE_60,
    RationalTime,
)
from .scanner import ScanResult, Scanner, scan_source_folder
from .schemas import (
    Act,
    Corpus,
    EditDecision,
    ProjectConfig,
    RunnerManifest,
    ScanStatus,
    Segment,
    SequenceItem,
    SourceClip,
    StageRecord,
    StoryArc,
)
from .segmenter import Segmenter, segment_clip, segment_clips

__all__ = [
    "RationalTime",
    "RATE_23_976",
    "RATE_24",
    "RATE_25",
    "RATE_29_97",
    "RATE_30",
    "RATE_48",
    "RATE_50",
    "RATE_59_94",
    "RATE_60",
    "SourceClip",
    "ScanStatus",
    "Segment",
    "Act",
    "StoryArc",
    "EditDecision",
    "SequenceItem",
    "Corpus",
    "ProjectConfig",
    "RunnerManifest",
    "StageRecord",
    "Scanner",
    "ScanResult",
    "scan_source_folder",
    "Segmenter",
    "segment_clip",
    "segment_clips",
]

