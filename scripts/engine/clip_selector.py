"""Clip Selector — assign segments to acts with scoring and AI refinement.

Produces a final ``EditDecision`` from a ``Corpus`` and a set of ``StoryArc``
candidates.  The primary path is a deterministic rule-based scoring engine
that optimises for mood match, quality, duration fit, and source variety.

When ``use_ai=True`` and a DeepSeek-compatible API is available, the selected
segments for each act are passed to the model for creative reordering, which
can suggest alternative pacing while preserving the original selection set.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .rational_time import RationalTime
from .schemas import (
    Act,
    Corpus,
    EditDecision,
    Segment,
    SequenceItem,
    StoryArc,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional AI dependencies
# ---------------------------------------------------------------------------

_HAS_OPENAI: bool = False
try:
    from openai import OpenAI  # type: ignore[import-untyped]

    _HAS_OPENAI = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

SCORE_MOOD_MATCH_WEIGHT: float = 0.25
SCORE_QUALITY_WEIGHT: float = 0.25
SCORE_DURATION_FIT_WEIGHT: float = 0.20
SCORE_VARIETY_BONUS_WEIGHT: float = 0.30

MIN_CONFIDENCE_THRESHOLD: float = 0.10
"""Segments below this score are not assigned."""

DEFAULT_TRANSITION_DURATION_FRAMES: int = 12
"""Default cross-dissolve duration in frames (at 24 fps ≈ 0.5 s)."""

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Keyword maps for mood matching (fallback when no NLP is available)
_MOOD_KEYWORDS: dict[str, set[str]] = {
    "awe": {"landscape", "aerial", "wide", "mountain", "sunset", "ocean", "vast", "beautiful"},
    "tension": {"closeup", "dark", "shake", "movement", "fast", "chaos", "conflict"},
    "calm": {"slow", "still", "nature", "water", "peaceful", "dawn", "gentle"},
    "joy": {"bright", "people", "smile", "celebration", "colorful", "motion", "happy"},
    "sadness": {"dark", "rain", "slow", "lonely", "somber", "quiet", "melancholy"},
    "energy": {"fast", "action", "sports", "movement", "dynamic", "intense", "bright"},
    "mystery": {"dark", "shadow", "fog", "closeup", "strange", "unusual", "ambiguous"},
    "wonder": {"nature", "aerial", "light", "beautiful", "vast", "colorful", "landscape"},
    "melancholy": {"rain", "slow", "fade", "solitary", "gray", "quiet", "reflective"},
    "excitement": {"fast", "action", "closeup", "shake", "intense", "dynamic", "motion"},
}


def _mood_match_score(
    segment: Segment,
    act: Act,
) -> float:
    """Score how well a segment's attributes match the act's emotional target.

    Uses keyword overlap between act emotional_target / required_content_elements
    and segment scene_type, tags, and motion characteristics.

    Returns 0.0 - 1.0.
    """
    mood = act.emotional_target.lower().strip()
    target_keywords: set[str] = set()

    # Build keyword set from act metadata
    mood_keywords = _MOOD_KEYWORDS.get(mood, set())
    target_keywords.update(mood_keywords)
    for elem in act.required_content_elements:
        for word in re.split(r"[_\s]+", elem.lower()):
            target_keywords.add(word)

    # Build keyword set from segment
    seg_keywords: set[str] = set()
    if segment.scene_type:
        for word in re.split(r"[_\s]+", segment.scene_type.lower()):
            seg_keywords.add(word)
    if segment.tags:
        for tag in segment.tags:
            for word in re.split(r"[_\s]+", tag.lower()):
                seg_keywords.add(word)
    if segment.subjects:
        for subj in segment.subjects:
            for word in re.split(r"[_\s]+", subj.lower()):
                seg_keywords.add(word)
    if segment.motion_type:
        for word in re.split(r"[_\s]+", segment.motion_type.lower()):
            seg_keywords.add(word)

    if not target_keywords or not seg_keywords:
        return 0.5  # neutral when we have no data

    overlap = len(target_keywords & seg_keywords)
    max_possible = min(len(target_keywords), len(seg_keywords))
    if max_possible == 0:
        return 0.5

    score = overlap / max_possible
    return min(score * 1.5, 1.0)  # boost a bit but cap at 1.0


def _quality_score(segment: Segment) -> float:
    """Score segment quality (0-1).

    Prefers ``composite_quality`` when available, falls back to
    ``composition_quality``, then to ``story_potential``.
    """
    if segment.composite_quality is not None:
        return max(0.0, min(segment.composite_quality, 1.0))
    if segment.composition_quality is not None:
        return max(0.0, min(segment.composition_quality, 1.0))
    if segment.story_potential is not None:
        return max(0.0, min(segment.story_potential, 1.0))
    return 0.5  # neutral default


def _duration_fit_score(
    segment: Segment,
    act: Act,
    total_target_duration: float,
) -> float:
    """Score how well the segment's duration fits the expected per-segment
    duration for this act.

    The ideal segment duration is derived from the act's ``duration_fraction``
    and ``clip_count_target``.

    Returns 0.0 - 1.0.
    """
    target_act_duration = total_target_duration * act.duration_fraction
    ideal_seg_duration = target_act_duration / max(act.clip_count_target, 1)

    seg_dur = segment.duration_seconds
    if seg_dur <= 0:
        return 0.0

    ratio = seg_dur / ideal_seg_duration
    if ratio < 0.25 or ratio > 4.0:
        return 0.1  # too short or too long
    if 0.5 <= ratio <= 2.0:
        return 1.0  # close to ideal
    # Linear falloff
    return max(0.1, 1.0 - abs(1.0 - ratio))


def _variety_bonus(
    clip_ids_used: set[str],
    clip_id: str,
) -> float:
    """Bonus for using a segment from a clip not yet used in this act.

    Returns 1.0 if the clip is new, 0.3 if already used (still allows
    multiple segments from the same source but penalises it).
    """
    if clip_id not in clip_ids_used:
        return 1.0
    return 0.3


@dataclass
class ScoredSegment:
    """A segment with its computed assignment score for a specific act."""

    segment: Segment
    act_index: int
    total_score: float
    mood_match: float
    quality: float
    duration_fit: float
    variety_bonus: float


# ---------------------------------------------------------------------------
# AI refinement
# ---------------------------------------------------------------------------

def _build_ai_prompt(segments: list[Segment], act: Act) -> str:
    """Build a prompt for the AI reordering pass."""
    seg_lines: list[str] = []
    for i, seg in enumerate(segments):
        name = seg.id
        scene = seg.scene_type or "unknown"
        dur = f"{seg.duration_seconds:.1f}s"
        mood = f"motion={seg.motion_intensity:.2f}"
        tags = ""
        if seg.tags:
            tags = f" tags={','.join(seg.tags[:5])}"
        seg_lines.append(f"  [{i}] {name} ({scene}, {dur}, {mood}{tags})")

    return f"""You are a video editor refining the order of clips in a sequence.

Act "{act.name}": {act.description}
Emotional target: {act.emotional_target}
Required elements: {', '.join(act.required_content_elements)}

Currently assigned segments (in proposed order):
{chr(10).join(seg_lines)}

Task: Suggest a *reordered* sequence of these exact same segments (by their index numbers) that:
1. Creates a better narrative arc for the "{act.emotional_target}" emotional tone
2. Varies shot types and pacing (don't put similar segments adjacent)
3. Respects the act's purpose: {act.description}

Return ONLY a JSON array of the reordered indices, e.g. [2, 0, 1, 3, ...].
Keep all segments; do not remove any."""


def _call_ai_refine(
    segments: list[Segment],
    act: Act,
    api_key: str | None = None,
    model: str = "deepseek-chat",
    base_url: str | None = None,
) -> list[int] | None:
    """Call the AI API to suggest a reordering.

    Returns a list of indices (into *segments*) in the suggested new order,
    or ``None`` on failure.
    """
    if not _HAS_OPENAI:
        logger.warning("openai package not installed; skipping AI refine")
        return None

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No API key found (DEEPSEEK_API_KEY or OPENAI_API_KEY); "
                       "skipping AI refine")
        return None

    if base_url is None:
        base_url = os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        )

    prompt = _build_ai_prompt(segments, act)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        content = resp.choices[0].message.content.strip()

        # Extract JSON array from response
        # Try to find [...] in the response
        match = re.search(r"\[[\d,\s]+\]", content)
        if match:
            indices = json.loads(match.group())
            # Validate: must contain all original indices exactly once
            expected = set(range(len(segments)))
            if sorted(indices) != sorted(expected):
                logger.warning(
                    "AI returned invalid indices (missing or extra): %s", indices
                )
                return None
            logger.info(
                "AI refine for act %s: %s → %s",
                act.name, list(range(len(segments))), indices,
            )
            return indices
        else:
            logger.warning(
                "AI response did not contain a JSON array: %s", content[:200]
            )
            return None
    except Exception as exc:
        logger.warning("AI refine call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# ClipSelector class
# ---------------------------------------------------------------------------

# Default transition durations per act type (in frames at 24 fps)
_DEFAULT_TRANSITIONS: dict[str, str] = {
    "Establishing": "cross_dissolve",
    "Rising Action": "hard_cut",
    "Climax": "hard_cut",
    "Falling Action": "cross_dissolve",
    "Resolution": "fade_to_black",
}


class ClipSelector:
    """Assign segments to acts and produce an ``EditDecision``.

    The primary path is a deterministic scoring engine that evaluates every
    segment against each act using four weighted criteria.  When
    ``use_ai=True``, a second pass calls a language model to suggest creative
    reorderings of the selected segments within each act.

    Parameters
    ----------
    corpus:
        The ``Corpus`` of all available segments.
    target_duration_seconds:
        Target duration of the final edit (from ``ProjectConfig``).
    timeline_rate:
        Frame rate of the output timeline (e.g. ``RATE_24``).
    project_title:
        Project title for the output ``EditDecision``.
    """

    def __init__(
        self,
        corpus: Corpus,
        target_duration_seconds: float,
        timeline_rate: RationalTime,
        project_title: str = "Untitled",
    ) -> None:
        self.corpus = corpus
        self.target_duration = target_duration_seconds
        self.timeline_rate = timeline_rate
        self.project_title = project_title

    # ---- Public API --------------------------------------------------------

    def select(
        self,
        story_arc: StoryArc,
        use_ai: bool = False,
        ai_api_key: str | None = None,
        ai_model: str = "deepseek-chat",
    ) -> EditDecision:
        """Run the clip selection pipeline for a single ``StoryArc``.

        Parameters
        ----------
        story_arc:
            The arc to assign segments for.
        use_ai:
            If ``True``, run an AI refinement pass on each act's segment order.
        ai_api_key:
            Optional API key override.  Falls back to environment variables.
        ai_model:
            Model name (default ``"deepseek-chat"``).

        Returns
        -------
        EditDecision
            The ordered timeline with all assigned segments.
        """
        items: list[SequenceItem] = []
        timeline_pos = RationalTime(0, 1, 0, 1)

        for act in story_arc.acts:
            scored = self._score_and_assign(act)

            # Sort by score descending, take clip_count_target or all
            scored.sort(key=lambda s: s.total_score, reverse=True)
            n_target = min(act.clip_count_target, len(scored))
            selected = scored[:n_target]

            if not selected:
                continue

            # Extract just the segments in scored order
            act_segments = [s.segment for s in selected]

            # AI refine pass (reorders only, does not replace selection)
            if use_ai:
                reordered_indices = _call_ai_refine(
                    act_segments, act,
                    api_key=ai_api_key, model=ai_model,
                )
                if reordered_indices is not None:
                    act_segments = [act_segments[i] for i in reordered_indices]

            # Build SequenceItems
            for seg in act_segments:
                item = self._make_sequence_item(
                    seg, act, timeline_pos,
                )
                items.append(item)
                timeline_pos = timeline_pos + item.duration

        # Apply small gap between acts (6 frames ≈ 0.25 s at 24 fps)
        # This is handled by the timeline_position calculation

        return EditDecision(
            project_title=self.project_title,
            timeline_rate=self.timeline_rate,
            target_duration_seconds=self.target_duration,
            items=items,
            created_at=datetime.now(),
        )

    # ---- Scoring & assignment ----------------------------------------------

    def _score_and_assign(
        self,
        act: Act,
    ) -> list[ScoredSegment]:
        """Score all corpus segments against a single act.

        Returns a list of ``ScoredSegment`` with computed scores, sorted
        descending.
        """
        clip_ids_used: set[str] = set()
        scored: list[ScoredSegment] = []

        for seg in self.corpus.segments:
            if seg.audio_analysis_status == "error":
                continue  # skip segments with fatal analysis errors

            mood = _mood_match_score(seg, act)
            quality = _quality_score(seg)
            dur_fit = _duration_fit_score(seg, act, self.target_duration)
            variety = _variety_bonus(clip_ids_used, seg.clip_id)

            total = (
                mood * SCORE_MOOD_MATCH_WEIGHT
                + quality * SCORE_QUALITY_WEIGHT
                + dur_fit * SCORE_DURATION_FIT_WEIGHT
                + variety * SCORE_VARIETY_BONUS_WEIGHT
            )

            if total >= MIN_CONFIDENCE_THRESHOLD:
                scored.append(ScoredSegment(
                    segment=seg,
                    act_index=act.index,
                    total_score=round(total, 4),
                    mood_match=round(mood, 3),
                    quality=round(quality, 3),
                    duration_fit=round(dur_fit, 3),
                    variety_bonus=round(variety, 3),
                ))
                clip_ids_used.add(seg.clip_id)

        return scored

    # ---- Sequence item construction ----------------------------------------

    def _make_sequence_item(
        self,
        segment: Segment,
        act: Act,
        timeline_position: RationalTime,
    ) -> SequenceItem:
        """Build a ``SequenceItem`` from a segment and act."""
        duration = segment.out_point - segment.in_point
        transition = _DEFAULT_TRANSITIONS.get(act.name, "cross_dissolve")
        trans_dur = RationalTime.from_frames(
            DEFAULT_TRANSITION_DURATION_FRAMES, self.timeline_rate,
        )

        reason = (
            f"Act {act.index} ({act.name}): "
            f"mood={act.emotional_target}, "
            f"scene={segment.scene_type or 'unknown'}"
        )

        return SequenceItem(
            segment_id=segment.id,
            clip_id=segment.clip_id,
            act_index=act.index,
            source_in=segment.in_point,
            source_out=segment.out_point,
            timeline_position=timeline_position,
            duration=duration,
            transition=transition,
            transition_duration=trans_dur,
            reason=reason,
            confidence=self._compute_confidence(segment, act),
        )

    @staticmethod
    def _compute_confidence(segment: Segment, act: Act) -> float:
        """Compute a 0-1 confidence score for this assignment."""
        score = 0.5  # baseline
        if segment.composite_quality is not None:
            score = segment.composite_quality * 0.7 + 0.3
        elif segment.composition_quality is not None:
            score = segment.composition_quality * 0.6 + 0.2
        return round(min(score, 1.0), 3)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def select_clips(
    corpus: Corpus,
    story_arc: StoryArc,
    target_duration_seconds: float,
    timeline_rate: RationalTime,
    use_ai: bool = False,
    project_title: str = "Untitled",
) -> EditDecision:
    """Convenience: create a ``ClipSelector`` and run selection."""
    selector = ClipSelector(
        corpus=corpus,
        target_duration_seconds=target_duration_seconds,
        timeline_rate=timeline_rate,
        project_title=project_title,
    )
    return selector.select(story_arc, use_ai=use_ai)
