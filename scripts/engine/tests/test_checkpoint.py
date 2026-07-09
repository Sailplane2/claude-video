"""Comprehensive unit tests for the Checkpoint merge algorithm.

Tests cover:
- ``_compute_best_acts``: picking the highest-feasibility act per position
- ``_build_composite_arc``: constructing a composite from best acts
- ``Checkpoint.merge_arcs``: full merge pipeline including fallback logic
- ``Checkpoint.present_arcs``: formatting output (smoke test)
- Edge cases: ties, single-arc, human overrides, fallback when merge is worse
"""

import copy
from dataclasses import dataclass

import pytest

from engine.checkpoint import (
    Checkpoint,
    CheckpointResult,
    _build_composite_arc,
    _compute_best_acts,
    merge_arcs,
)
from engine.schemas import Act, Corpus, Segment, StoryArc


# ===================================================================
# Fixtures
# ===================================================================


def _make_act(
    index: int,
    name: str,
    feasibility_score: float,
    duration_fraction: float = 0.33,
    emotional_target: str = "neutral",
    clip_count_target: int = 5,
) -> Act:
    return Act(
        index=index,
        name=name,
        description=f"Act {index} description",
        duration_fraction=duration_fraction,
        emotional_target=emotional_target,
        clip_count_target=clip_count_target,
        feasibility_score=feasibility_score,
    )


def _make_arc(
    arc_id: str,
    name: str,
    theme: str,
    acts: list[Act],
    arc_type: str = "classic",
    overall_feasibility: float | None = None,
    gaps: list[str] | None = None,
) -> StoryArc:
    if overall_feasibility is None and acts:
        overall_feasibility = sum(a.feasibility_score for a in acts) / len(acts)
    return StoryArc(
        id=arc_id,
        name=name,
        theme=theme,
        acts=acts,
        arc_type=arc_type,
        overall_feasibility=overall_feasibility or 0.0,
        gaps=gaps or [],
    )


@pytest.fixture
def arcs_three() -> list[StoryArc]:
    """Three arcs with clear winners per act position.

    Arc A: Act 1=0.9, Act 2=0.3, Act 3=0.3  → wins Act 1
    Arc B: Act 1=0.5, Act 2=0.8, Act 3=0.4  → wins Act 2
    Arc C: Act 1=0.4, Act 2=0.4, Act 3=0.9  → wins Act 3
    """
    a = _make_arc(
        "arc_a", "Classic", "Standard arc",
        acts=[
            _make_act(1, "Establishing", 0.9),
            _make_act(2, "Rising Action", 0.3),
            _make_act(3, "Climax", 0.3),
        ],
    )
    b = _make_arc(
        "arc_b", "Thematic", "Theme-focused arc",
        acts=[
            _make_act(1, "Setup", 0.5),
            _make_act(2, "Development", 0.8),
            _make_act(3, "Resolution", 0.4),
        ],
    )
    c = _make_arc(
        "arc_c", "Alternate", "Alternate arc",
        acts=[
            _make_act(1, "Intro", 0.4),
            _make_act(2, "Body", 0.4),
            _make_act(3, "Finale", 0.9),
        ],
    )
    return [a, b, c]


@pytest.fixture
def arcs_tied() -> list[StoryArc]:
    """Two arcs where act feasibilities are identical, but overall
    feasibility differs (ties should resolve to the arc with higher
    overall feasibility)."""
    a = _make_arc(
        "arc_a", "Arc A", "First arc",
        acts=[
            _make_act(1, "Begin", 0.7),
            _make_act(2, "Middle", 0.7),
            _make_act(3, "End", 0.7),
        ],
        overall_feasibility=0.7,
    )
    b = _make_arc(
        "arc_b", "Arc B", "Second arc",
        acts=[
            _make_act(1, "Start", 0.7),
            _make_act(2, "Mid", 0.7),
            _make_act(3, "Finish", 0.7),
        ],
        overall_feasibility=0.85,
    )
    return [a, b]


@pytest.fixture
def arcs_single() -> list[StoryArc]:
    """Only one arc — no merge possible."""
    a = _make_arc(
        "arc_a", "Solo", "Only arc",
        acts=[
            _make_act(1, "Intro", 0.6),
            _make_act(2, "Body", 0.7),
            _make_act(3, "Outro", 0.8),
        ],
    )
    return [a]


@pytest.fixture
def arcs_merge_worse() -> list[StoryArc]:
    """Arcs where the merge produces a composite *worse* than the best
    single arc, triggering the fallback.

    Arc B dominates everywhere, so the composite is identical or worse.
    """
    a = _make_arc(
        "arc_a", "Weak", "Weak arc",
        acts=[
            _make_act(1, "Intro", 0.3),
            _make_act(2, "Middle", 0.2),
            _make_act(3, "End", 0.4),
        ],
    )
    b = _make_arc(
        "arc_b", "Strong", "Dominant arc",
        acts=[
            _make_act(1, "Opening", 0.9),
            _make_act(2, "Rising", 0.8),
            _make_act(3, "Finale", 0.85),
        ],
    )
    return [a, b]


@pytest.fixture
def empty_corpus() -> Corpus:
    return Corpus(
        project_title="Test",
        segments=[],
    )


# ===================================================================
# _compute_best_acts
# ===================================================================


class TestComputeBestActs:
    def test_picks_highest_feasibility_per_position(self, arcs_three):
        best = _compute_best_acts(arcs_three)
        assert len(best) == 3
        # Position 0: arc_a wins (feas=0.9)
        assert best[0] == (0, 0.9)
        # Position 1: arc_b wins (feas=0.8)
        assert best[1] == (1, 0.8)
        # Position 2: arc_c wins (feas=0.9)
        assert best[2] == (2, 0.9)

    def test_empty_list(self):
        assert _compute_best_acts([]) == []

    def test_tie_breaks_by_overall_feasibility(self, arcs_tied):
        """When two arcs have the same act feasibility, pick the arc with
        the higher overall feasibility."""
        best = _compute_best_acts(arcs_tied)
        # All three positions are tied and should resolve to arc_b
        # (overall 0.85 > 0.70)
        for pos in range(3):
            assert best[pos][0] == 1  # arc_b wins every position

    def test_single_arc(self, arcs_single):
        best = _compute_best_acts(arcs_single)
        assert len(best) == 3
        for pos in range(3):
            assert best[pos][0] == 0  # only arc

    def test_arcs_with_uneven_act_counts(self):
        """If arcs have different numbers of acts, handle gracefully."""
        a = _make_arc(
            "arc_a", "Short", "2-act arc",
            acts=[
                _make_act(1, "A1", 0.8),
                _make_act(2, "A2", 0.7),
            ],
        )
        b = _make_arc(
            "arc_b", "Long", "3-act arc",
            acts=[
                _make_act(1, "B1", 0.6),
                _make_act(2, "B2", 0.9),
                _make_act(3, "B3", 0.5),
            ],
        )
        best = _compute_best_acts([a, b])
        assert len(best) == 3
        # Position 0: arc_a wins (0.8 > 0.6)
        assert best[0] == (0, 0.8)
        # Position 1: arc_b wins (0.9 > 0.7)
        assert best[1] == (1, 0.9)
        # Position 2: only arc_b has act 3
        assert best[2] == (1, 0.5)


# ===================================================================
# _build_composite_arc
# ===================================================================


class TestBuildCompositeArc:
    def test_builds_from_best_acts(self, arcs_three):
        best = [(0, 0.9), (1, 0.8), (2, 0.9)]
        composite = _build_composite_arc(arcs_three, best)

        assert composite.id.startswith("composite_")
        assert "Composite" in composite.name

        # Acts should come from the winning arcs
        assert composite.acts[0].name == arcs_three[0].acts[0].name  # arc_a's act 1
        assert composite.acts[1].name == arcs_three[1].acts[1].name  # arc_b's act 2
        assert composite.acts[2].name == arcs_three[2].acts[2].name  # arc_c's act 3

        # Feasibility should be the average of the three winning acts
        expected_feas = round((0.9 + 0.8 + 0.9) / 3, 4)
        assert composite.overall_feasibility == expected_feas

    def test_inherits_from_dominant_arc(self, arcs_three):
        """The composite should inherit its name/theme from the arc with
        highest overall feasibility."""
        best = [(0, 0.9), (1, 0.8), (2, 0.9)]
        composite = _build_composite_arc(arcs_three, best)

        # arcs_three[0] has the highest overall (0.9+0.3+0.3)/3=0.5
        # arcs_three[1] has (0.5+0.8+0.4)/3≈0.567
        # arcs_three[2] has (0.4+0.4+0.9)/3≈0.567
        # arc_b and arc_c tie at ~0.567, but arc_b has higher single act
        # Actually: arc_a=0.5, arc_b=0.567, arc_c=0.567
        # The max() picks the first of the tie, which is arc_b (index 1)
        # Actually wait — arc_b and arc_c both have avg 0.567, but arc_b
        # is earlier in the list. Let me check: max(range(len(arcs)), key=lambda i: arcs[i].overall_feasibility)
        # arcs[0].overall_feasibility = 0.5, arcs[1] = 0.567, arcs[2] = 0.567
        # First max is index 1 (arc_b)
        assert composite.arc_type == arcs_three[1].arc_type

    def test_merged_from_set(self, arcs_three):
        best = [(0, 0.9), (1, 0.8), (2, 0.9)]
        composite = _build_composite_arc(arcs_three, best)
        assert composite.merged_from is not None
        assert all(aid in composite.merged_from for aid in ["arc_a", "arc_b", "arc_c"])

    def test_collects_gaps_from_all_arcs(self):
        a = _make_arc(
            "arc_a", "A", "QA",
            acts=[_make_act(1, "A1", 0.5), _make_act(2, "A2", 0.6)],
            gaps=["No establishing shot", "Low light in act 2"],
        )
        b = _make_arc(
            "arc_b", "B", "QB",
            acts=[_make_act(1, "B1", 0.9), _make_act(2, "B2", 0.8)],
            gaps=["Missing dialogue", "Low light in act 2"],
        )
        best = [(0, 0.5), (1, 0.8)]
        composite = _build_composite_arc([a, b], best)
        # Should have 3 unique gaps (duplicates deduplicated)
        assert len(composite.gaps) == 3
        assert "Low light in act 2" in composite.gaps

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one arc"):
            _build_composite_arc([], [])

    def test_deep_copy_decoupling(self, arcs_three):
        """Modifying the composite's acts should not affect the originals."""
        best = [(0, 0.9), (1, 0.8), (2, 0.9)]
        composite = _build_composite_arc(arcs_three, best)

        original_name = arcs_three[0].acts[0].name
        composite.acts[0].name = "Modified"
        assert arcs_three[0].acts[0].name == original_name


# ===================================================================
# Checkpoint.merge_arcs
# ===================================================================


class TestMergeArcs:
    def test_merge_from_three_arcs(self, arcs_three, empty_corpus):
        cp = Checkpoint(arcs_three, empty_corpus)
        result = cp.merge_arcs()

        assert result.is_merged
        assert len(result.merge_sources or []) == 3
        assert result.selected_arc.overall_feasibility == round((0.9 + 0.8 + 0.9) / 3, 4)

    def test_merge_with_human_override(self, arcs_three, empty_corpus):
        """Override position 0 to use arc_b's act instead of arc_a's."""
        cp = Checkpoint(arcs_three, empty_corpus)
        result = cp.merge_arcs(human_override={0: 1})

        # Position 0 should now use arc_b's act (feasibility 0.5)
        assert result.is_merged
        assert result.human_override_indices == [0]
        assert result.selected_arc.acts[0].name == arcs_three[1].acts[0].name

    def test_single_arc_no_merge(self, arcs_single, empty_corpus):
        cp = Checkpoint(arcs_single, empty_corpus)
        result = cp.merge_arcs()

        assert not result.is_merged
        assert "Only one arc" in result.warnings[0]

    def test_fallback_when_merge_is_worse(self, arcs_merge_worse, empty_corpus):
        """When composite feasibility < best single arc, fall back to
        best single."""
        cp = Checkpoint(arcs_merge_worse, empty_corpus)
        result = cp.merge_arcs()

        # Should fall back to arc_b (the strong arc)
        assert not result.is_merged
        assert result.selected_arc.id == "arc_b"
        assert len(result.warnings) >= 1
        assert "worse than" in result.warnings[0]

    def test_merge_with_all_from_one_arc_detects_no_merge(self):
        """If all best acts come from the same arc, is_merged=False."""
        a = _make_arc(
            "arc_a", "Dominant", "Everything wins",
            acts=[
                _make_act(1, "A1", 0.9),
                _make_act(2, "A2", 0.9),
                _make_act(3, "A3", 0.9),
            ],
        )
        b = _make_arc(
            "arc_b", "Weak", "Loses everywhere",
            acts=[
                _make_act(1, "B1", 0.1),
                _make_act(2, "B2", 0.1),
                _make_act(3, "B3", 0.1),
            ],
        )
        cp = Checkpoint([a, b], Corpus(project_title="T", segments=[]))
        result = cp.merge_arcs()
        assert not result.is_merged  # all from arc_a
        assert result.selected_arc.name == "Composite (Dominant + Weak)"
        # The composite picks all from arc_a anyway

    def test_empty_arcs_raises(self, empty_corpus):
        cp = Checkpoint([], empty_corpus)
        with pytest.raises(ValueError, match="No arcs to merge"):
            cp.merge_arcs()

    def test_merge_arcs_convenience(self, arcs_three, empty_corpus):
        result = merge_arcs(arcs_three, empty_corpus)
        assert isinstance(result, CheckpointResult)
        assert result.is_merged

    def test_select_best_single(self, arcs_three, empty_corpus):
        cp = Checkpoint(arcs_three, empty_corpus)
        result = cp.select_best_single()
        assert not result.is_merged
        # arc_b and arc_c tie at 0.567, but arc_b is first in list
        assert result.selected_arc.id == "arc_b" or result.selected_arc.id == "arc_c"


# ===================================================================
# Checkpoint.present_arcs
# ===================================================================


class TestPresentArcs:
    def test_returns_string_with_arc_ids(self, arcs_three, empty_corpus):
        cp = Checkpoint(arcs_three, empty_corpus)
        output = cp.present_arcs()
        assert isinstance(output, str)
        assert "arc_a" in output
        assert "arc_b" in output
        assert "arc_c" in output

    def test_empty_arcs_message(self, empty_corpus):
        cp = Checkpoint([], empty_corpus)
        output = cp.present_arcs()
        assert "No arcs" in output

    def test_includes_prompt_text(self, arcs_three, empty_corpus):
        cp = Checkpoint(arcs_three, empty_corpus)
        output = cp.present_arcs()
        assert "[a]" in output
        assert "[b]" in output
        assert "[m]" in output
        assert "[r]" in output

    def test_includes_feasibility_scores(self, arcs_three, empty_corpus):
        cp = Checkpoint(arcs_three, empty_corpus)
        output = cp.present_arcs()
        assert "feasibility=" in output


# ===================================================================
# CheckpointResult properties
# ===================================================================


class TestCheckpointResult:
    def test_overall_feasibility_proxy(self):
        arc = _make_arc("test", "Test", "Theme", acts=[_make_act(1, "A", 0.5)])
        result = CheckpointResult(selected_arc=arc)
        assert result.overall_feasibility == arc.overall_feasibility
