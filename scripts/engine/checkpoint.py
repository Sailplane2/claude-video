"""Checkpoint — merge fallback and interactive arc review.

After story generation produces 2-3 competing ``StoryArc`` proposals, the
Checkpoint stage presents them side-by-side for human review and supports
an automated merge algorithm that selects the best act from each arc by
feasibility score, producing a composite arc.

If the composite arc has a lower overall feasibility than the best single
arc, the merge is discarded and the highest-scoring arc is used instead.
"""

from __future__ import annotations

import copy
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from .schemas import Act, Corpus, StoryArc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACT_LABELS = ("Act 1", "Act 2", "Act 3")
ACT_NAMES = ("Establishing", "Rising Action / Development", "Climax / Resolution")

TABLE_BORDER = "─" * 72

# ---------------------------------------------------------------------------
# Merge result
# ---------------------------------------------------------------------------

MERGE_LABELS = ("a", "b", "c")


@dataclass
class CheckpointResult:
    """The output of a checkpoint review or merge."""

    selected_arc: StoryArc
    """The arc chosen (either a single source or a composite)."""

    is_merged: bool = False
    """``True`` if the arc was created via merge rather than selected whole."""

    merge_sources: Optional[list[str]] = None
    """The source arc IDs that contributed to a composite."""

    merged_act_indices: Optional[list[int]] = None
    """For composite arcs, the original act index used per position
    (index 0 → act index from the source arc)."""

    human_override_indices: Optional[list[int]] = None
    """Act positions where the human manually overrode the automatic merge."""

    warnings: list[str] = field(default_factory=list)

    @property
    def overall_feasibility(self) -> float:
        return self.selected_arc.overall_feasibility


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_arc_card(arc: StoryArc, label: str) -> str:
    """Format a single arc as a compact card block."""
    lines: list[str] = [
        f"[{label.upper()}] {arc.name:30s}  feasibility={arc.overall_feasibility:.3f}",
        f"       Theme: {arc.theme}",
        f"       Type:  {arc.arc_type}",
    ]
    if arc.self_critique:
        wrapped = textwrap.wrap(arc.self_critique, width=60)
        for w in wrapped:
            lines.append(f"       | {w}")

    for act in arc.acts:
        lines.append(
            f"       ├─ Act {act.index}: {act.name:25s}"
            f"  dur={act.duration_fraction:.0%}"
            f"  clips={act.clip_count_target}"
            f"  mood={act.emotional_target}"
            f"  feas={act.feasibility_score:.3f}"
        )

    if arc.gaps:
        for gap in arc.gaps[:3]:
            wrapped = textwrap.wrap(gap, width=65)
            for w in wrapped:
                lines.append(f"       ⚠  {w}")
        if len(arc.gaps) > 3:
            lines.append(f"       ⚠  ... and {len(arc.gaps) - 3} more gaps")

    return "\n".join(lines)


def _format_merge_preview(arcs: list[StoryArc], best_acts: list[tuple[int, float]]) -> str:
    """Format a preview of what the automated merge would produce."""
    lines: list[str] = ["── Merge Preview ──"]
    for pos, (arc_idx, feas) in enumerate(best_acts):
        src = arcs[arc_idx]
        lines.append(
            f"  {ACT_LABELS[pos]:20s} ← [{MERGE_LABELS[arc_idx].upper()}] "
            f"{src.acts[pos].name}  (feasibility={feas:.3f})"
        )
    lines.append(f"{'─' * 40}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merge algorithm
# ---------------------------------------------------------------------------

def _compute_best_acts(arcs: list[StoryArc]) -> list[tuple[int, float]]:
    """For each act position, pick the arc with the highest feasibility score.

    Returns a list of ``(arc_index, feasibility_score)`` tuples, one per act
    position (typically 3 for 3 acts).

    The act positions are aligned by ``act.index`` (1-based).  All arcs are
    expected to have the same number of acts.
    """
    if not arcs:
        return []

    # Determine max act count across all arcs
    max_acts = max(len(a.acts) for a in arcs)
    best: list[tuple[int, float]] = []

    for pos in range(max_acts):
        best_arc_idx = 0
        best_feas = -1.0
        tie_count = 0

        for ai, arc in enumerate(arcs):
            if pos < len(arc.acts):
                act = arc.acts[pos]
                if act.feasibility_score > best_feas:
                    best_feas = act.feasibility_score
                    best_arc_idx = ai
                    tie_count = 1
                elif act.feasibility_score == best_feas:
                    tie_count += 1
                    # On tie, prefer the arc with higher overall feasibility
                    if arc.overall_feasibility > arcs[best_arc_idx].overall_feasibility:
                        best_arc_idx = ai

        best.append((best_arc_idx, best_feas))

    return best


def _build_composite_arc(
    arcs: list[StoryArc],
    best_acts: list[tuple[int, float]],
    merged_from: Optional[list[str]] = None,
) -> StoryArc:
    """Build a composite ``StoryArc`` from the best acts across all arcs.

    The composite inherits its name, theme, and arc_type from the arc with the
    highest overall feasibility.  Each act is a deep copy from the winning arc.
    """
    if not arcs or not best_acts:
        raise ValueError("Need at least one arc and one best-act mapping")

    # Find the dominant arc (highest overall feasibility)
    dominant_idx = max(
        range(len(arcs)),
        key=lambda i: arcs[i].overall_feasibility,
    )
    dominant = arcs[dominant_idx]

    # Build composite acts
    composite_acts: list[Act] = []
    merged_act_indices: list[int] = []

    for pos, (arc_idx, _feas) in enumerate(best_acts):
        source_act = arcs[arc_idx].acts[pos]
        composite_acts.append(copy.deepcopy(source_act))
        merged_act_indices.append(arc_idx)

    # Compute composite feasibility as weighted average of act feasibilities
    if composite_acts:
        overall_feas = sum(
            a.feasibility_score for a in composite_acts
        ) / len(composite_acts)
    else:
        overall_feas = 0.0

    # Collect gaps from all source arcs
    composite_gaps: list[str] = []
    seen_gaps: set[str] = set()
    for arc in arcs:
        for gap in arc.gaps:
            if gap not in seen_gaps:
                composite_gaps.append(gap)
                seen_gaps.add(gap)

    merged_sources = merged_from or [a.id for a in arcs]

    return StoryArc(
        id=f"composite_{'_'.join(a.id for a in arcs)}",
        name=f"Composite ({' + '.join(a.name for a in arcs)})",
        theme=dominant.theme,
        acts=composite_acts,
        arc_type=dominant.arc_type,
        self_critique=f"Composite merge of: {', '.join(a.id for a in arcs)}",
        overall_feasibility=round(overall_feas, 4),
        gaps=composite_gaps,
        merged_from=merged_sources,
    )


# ---------------------------------------------------------------------------
# Checkpoint class
# ---------------------------------------------------------------------------

class Checkpoint:
    """Review and optionally merge competing story arcs.

    The checkpoint stage is the human-in-the-loop gate between story
    generation and clip selection.  It presents arcs side-by-side and
    supports:

    * **Individual arc selection** — pick the best single arc.
    * **Automated merge** — combine the best act from each arc by
      feasibility score.
    * **Human override** — after merge preview, override individual
      act choices.

    Parameters
    ----------
    arcs:
        The list of ``StoryArc`` candidates (typically 2-3).
    corpus:
        The ``Corpus`` used for feasibility validation.
    """

    def __init__(
        self,
        arcs: list[StoryArc],
        corpus: Corpus,
    ) -> None:
        self.arcs = arcs
        self.corpus = corpus

    # ---- Public API --------------------------------------------------------

    def present_arcs(self) -> str:
        """Return a side-by-side textual display of all arcs.

        The display is formatted for terminal or log output.
        """
        if not self.arcs:
            return "No arcs to display."

        label_width = max(len(a.id) for a in self.arcs) + 4
        lines: list[str] = [
            TABLE_BORDER,
            "STORY ARC CHECKPOINT".center(72),
            TABLE_BORDER,
            "",
        ]

        for i, arc in enumerate(self.arcs):
            label = MERGE_LABELS[i] if i < len(MERGE_LABELS) else str(i)
            lines.append(_format_arc_card(arc, label))
            lines.append("")

        lines.append(self._interactive_prompt_text())
        lines.append(TABLE_BORDER)
        return "\n".join(lines)

    def merge_arcs(
        self,
        human_override: dict[int, int] | None = None,
    ) -> CheckpointResult:
        """Perform an automated merge of all arcs.

        By default, merges pick the best act per position by feasibility score.
        A human can override individual positions via *human_override*.

        Parameters
        ----------
        human_override:
            Optional dict mapping act position (0-based) to arc index
            (0-based).  For example ``{1: 2}`` means "use arc index 2's
            act for position 1" instead of the automatic best.

        Returns
        -------
        CheckpointResult
            The merge result or fallback to the best single arc.
        """
        if not self.arcs:
            raise ValueError("No arcs to merge")

        if len(self.arcs) == 1:
            # Only one arc — no merge needed
            return CheckpointResult(
                selected_arc=copy.deepcopy(self.arcs[0]),
                is_merged=False,
                warnings=["Only one arc available; no merge needed."],
            )

        # Compute automatic best acts
        best_acts = _compute_best_acts(self.arcs)

        # Apply human overrides
        if human_override:
            for pos, arc_idx in human_override.items():
                if 0 <= pos < len(best_acts):
                    if 0 <= arc_idx < len(self.arcs):
                        best_acts[pos] = (
                            arc_idx,
                            self.arcs[arc_idx].acts[pos].feasibility_score,
                        )
                    else:
                        logger.warning(
                            "Human override: arc index %d out of range (0-%d)",
                            arc_idx, len(self.arcs) - 1,
                        )

        # Build composite
        composite = _build_composite_arc(self.arcs, best_acts)

        # Find the best single arc for comparison
        best_single = max(self.arcs, key=lambda a: a.overall_feasibility)

        # Fallback: if composite is worse than the best single arc, discard merge
        warnings: list[str] = []
        if composite.overall_feasibility < best_single.overall_feasibility:
            logger.info(
                "Composite feasibility (%.4f) < best single (%.4f); "
                "falling back to single arc '%s'",
                composite.overall_feasibility,
                best_single.overall_feasibility,
                best_single.id,
            )
            warnings.append(
                f"Merge feasibility ({composite.overall_feasibility:.4f}) "
                f"is worse than best single arc '{best_single.id}' "
                f"({best_single.overall_feasibility:.4f}). Using single arc."
            )
            return CheckpointResult(
                selected_arc=copy.deepcopy(best_single),
                is_merged=False,
                warnings=warnings,
            )

        # Detect if the merge actually picks from different arcs
        source_arcs = set(ai for ai, _ in best_acts)
        is_merged = len(source_arcs) > 1

        return CheckpointResult(
            selected_arc=composite,
            is_merged=is_merged,
            merge_sources=list(source_arcs),
            merged_act_indices=[ai for ai, _ in best_acts],
            human_override_indices=list(human_override.keys()) if human_override else None,
            warnings=warnings,
        )

    def select_best_single(self) -> CheckpointResult:
        """Select the best single arc by overall feasibility."""
        if not self.arcs:
            raise ValueError("No arcs available")

        best = max(self.arcs, key=lambda a: a.overall_feasibility)
        return CheckpointResult(
            selected_arc=copy.deepcopy(best),
            is_merged=False,
        )

    # ---- Internal ----------------------------------------------------------

    @staticmethod
    def _interactive_prompt_text() -> str:
        """Return the prompt text shown at the bottom of the arc display."""
        return (
            "Actions:\n"
            "  [a][b][c]  — select the corresponding arc as-is\n"
            "  [m]        — merge (best act from each arc)\n"
            "  [r]        — reject all and restart story generation\n"
            "  [1-9]      — override specific act positions (interactive)"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def merge_arcs(
    arcs: list[StoryArc],
    corpus: Corpus,
    human_override: dict[int, int] | None = None,
) -> CheckpointResult:
    """Convenience: create a ``Checkpoint`` and run the merge."""
    cp = Checkpoint(arcs=arcs, corpus=corpus)
    return cp.merge_arcs(human_override=human_override)
