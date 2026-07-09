"""Comprehensive unit tests for ``RationalTime``.

Tests cover creation, conversion, arithmetic, comparison, and FCPXML
serialisation at the common NTSC and integer frame rates.

All tests are pure (no external dependencies beyond ``fractions.Fraction``).
"""

import pickle
from fractions import Fraction

import pytest

from engine.rational_time import (
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

# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_defaults_to_23_976(self):
        rt = RationalTime(300, 1)
        assert rt.numerator == 300
        assert rt.denominator == 1
        assert rt.rate_numerator == 24000
        assert rt.rate_denominator == 1001

    def test_explicit_parameters(self):
        rt = RationalTime(300300, 1001, 30000, 1001)
        assert rt.numerator == 300300
        assert rt.denominator == 1001
        assert rt.rate_numerator == 30000
        assert rt.rate_denominator == 1001

    def test_frozen_dataclass(self):
        rt = RationalTime(10, 1)
        with pytest.raises(AttributeError):
            rt.numerator = 20  # type: ignore[misc]

    def test_hashable(self):
        rt1 = RationalTime(10, 1, 24, 1)
        rt2 = RationalTime(10, 1, 24, 1)
        assert hash(rt1) == hash(rt2)
        assert len({rt1, rt2}) == 1

    def test_pickle_roundtrip(self):
        rt = RationalTime(300, 1, 24000, 1001)
        assert pickle.loads(pickle.dumps(rt)) == rt


# ===================================================================
# to_seconds
# ===================================================================


class TestToSeconds:
    def test_simple(self):
        rt = RationalTime(300, 1)
        assert rt.to_seconds() == 300.0

    def test_rational(self):
        rt = RationalTime(300300, 1001)
        assert abs(rt.to_seconds() - 300.0) < 1e-9

    def test_zero(self):
        rt = RationalTime(0, 1)
        assert rt.to_seconds() == 0.0

    def test_fractional_seconds(self):
        rt = RationalTime(1, 4)  # 0.25 s
        assert rt.to_seconds() == 0.25


# ===================================================================
# to_frames
# ===================================================================


class TestToFrames:
    def test_exact_at_24fps(self):
        """12 seconds at 24 fps = 288 frames exactly."""
        rt = RationalTime(12, 1, 24, 1)
        assert rt.to_frames() == 288

    def test_exact_at_30fps(self):
        """10 seconds at 30 fps = 300 frames exactly."""
        rt = RationalTime(10, 1, 30, 1)
        assert rt.to_frames() == 300

    def test_at_23_976(self):
        """300 seconds at 24000/1001 fps."""
        rt = RationalTime(300, 1, 24000, 1001)
        # 300 * 24000/1001 ≈ 7192.8 → rounds to 7193
        assert rt.to_frames() == 7193

    def test_at_29_97(self):
        """10 seconds at 30000/1001 fps."""
        rt = RationalTime(10, 1, 30000, 1001)
        # 10 * 30000/1001 ≈ 299.7 → rounds to 300
        assert rt.to_frames() == 300

    def test_at_59_94(self):
        """5 seconds at 60000/1001 fps."""
        rt = RationalTime(5, 1, 60000, 1001)
        # 5 * 60000/1001 ≈ 299.7 → rounds to 300
        assert rt.to_frames() == 300

    def test_zero_frames(self):
        rt = RationalTime(0, 1, 24, 1)
        assert rt.to_frames() == 0

    def test_one_frame_at_24fps(self):
        """1/24 s = 1 frame at 24 fps."""
        rt = RationalTime(1, 24, 24, 1)
        assert rt.to_frames() == 1

    def test_one_frame_at_23_976(self):
        """1 frame = 1001/24000 s at 23.976 fps."""
        rt = RationalTime(1001, 24000, 24000, 1001)
        assert rt.to_frames() == 1

    def test_rounding_midpoint(self):
        """0.5 frames rounds to nearest at rate level."""
        # 1/48 s at 24 fps = 0.5 frames → rounds to 0 or 1
        rt = RationalTime(1, 48, 24, 1)
        # 1/48 * 24 = 0.5 → round(0.5) = 0 (Python rounds to even)
        assert rt.to_frames() == 0


# ===================================================================
# to_fcpxml_duration
# ===================================================================


class TestToFcpxmlDuration:
    def test_no_timeline_rate_returns_raw_time(self):
        """Matches the design doc example: 300300/1001s."""
        rt = RationalTime(300300, 1001, 24000, 1001)
        assert rt.to_fcpxml_duration() == "300300/1001s"

    def test_raw_time_simple(self):
        rt = RationalTime(10, 1, 24, 1)
        assert rt.to_fcpxml_duration() == "10/1s"

    def test_frame_aligned_24fps(self):
        """10 seconds at 24fps → 240 frames → 240/24s = 10/1s."""
        rt = RationalTime.from_seconds(10.0, RATE_24)
        result = rt.to_fcpxml_duration(RATE_24)
        # 10 * 24 = 240 frames, 240 * 1 / 24 = 10/1
        assert result == "10/1s"

    def test_frame_aligned_23_976(self):
        """10 seconds at 23.976 fps → round(10*24000/1001) = 240 frames.
        240 * 1001 / 24000 = 240240/24000 which simplifies to 1001/100."""
        rt = RationalTime.from_seconds(10.0, RATE_23_976)
        result = rt.to_fcpxml_duration(RATE_23_976)
        assert result == "1001/100s"
        # Verify: 240240/24000 = 10.01 s (off by ~0.01 due to frame alignment)

    def test_frame_aligned_29_97(self):
        """10 seconds at 29.97 fps → round(10*30000/1001) ≈ 300 frames.
        300 * 1001 / 30000 = 300300/30000 = 1001/100."""
        rt = RationalTime.from_seconds(10.0, RATE_29_97)
        result = rt.to_fcpxml_duration(RATE_29_97)
        assert result == "1001/100s"
        # 1001/100 = 10.01 s

    def test_aligned_with_different_timeline_rate(self):
        """10 sec clip at 29.97 fps placed in a 24 fps timeline.
        10 seconds at 24 fps = 240 frames → 240 * 1/24 = 10/1."""
        rt = RationalTime.from_seconds(10.0, RATE_29_97)
        result = rt.to_fcpxml_duration(RATE_24)
        assert result == "10/1s"

    def test_exact_frame_count(self):
        """Exactly 10 frames at 24 fps."""
        rt = RationalTime.from_frames(10, RATE_24)
        result = rt.to_fcpxml_duration(RATE_24)
        # 10 * 1 / 24 = 10/24 = 5/12
        assert result == "5/12s"


# ===================================================================
# from_frames
# ===================================================================


class TestFromFrames:
    def test_at_24fps(self):
        rt = RationalTime.from_frames(240, RATE_24)
        assert rt.to_frames() == 240
        assert abs(rt.to_seconds() - 10.0) < 1e-9

    def test_at_23_976(self):
        rt = RationalTime.from_frames(240, RATE_23_976)
        assert rt.to_frames() == 240
        # 240 / (24000/1001) = 240 * 1001 / 24000 = 10.01
        assert abs(rt.to_seconds() - 10.01) < 0.01

    def test_at_29_97(self):
        rt = RationalTime.from_frames(300, RATE_29_97)
        assert rt.to_frames() == 300
        # 300 / (30000/1001) = 300 * 1001 / 30000 = 10.01
        assert abs(rt.to_seconds() - 10.01) < 0.01

    def test_zero_frames(self):
        rt = RationalTime.from_frames(0, RATE_24)
        assert rt.to_seconds() == 0.0
        assert rt.to_frames() == 0

    def test_preserves_rate(self):
        rt = RationalTime.from_frames(100, RATE_59_94)
        assert rt.rate_numerator == 60000
        assert rt.rate_denominator == 1001


# ===================================================================
# from_seconds
# ===================================================================


class TestFromSeconds:
    def test_from_float_24fps(self):
        rt = RationalTime.from_seconds(10.5, RATE_24)
        assert abs(rt.to_seconds() - 10.5) < 1e-6
        assert rt.rate_numerator == 24
        assert rt.rate_denominator == 1

    def test_from_int_24fps(self):
        rt = RationalTime.from_seconds(10, RATE_24)
        assert rt.to_seconds() == 10.0

    def test_from_fraction_24fps(self):
        rt = RationalTime.from_seconds(Fraction(21, 2), RATE_24)
        assert rt.to_seconds() == 10.5

    def test_zero_seconds(self):
        rt = RationalTime.from_seconds(0.0, RATE_24)
        assert rt.to_seconds() == 0.0

    def test_large_float(self):
        rt = RationalTime.from_seconds(3600.0, RATE_24)  # 1 hour
        assert abs(rt.to_seconds() - 3600.0) < 0.001

    def test_precision(self):
        """Float with limited precision should round-trip closely."""
        rt = RationalTime.from_seconds(12.345, RATE_29_97)
        assert abs(rt.to_seconds() - 12.345) < 0.001


# ===================================================================
# Arithmetic
# ===================================================================


class TestArithmetic:
    def test_add(self):
        a = RationalTime.from_seconds(10.0, RATE_24)
        b = RationalTime.from_seconds(5.0, RATE_24)
        c = a + b
        assert abs(c.to_seconds() - 15.0) < 1e-9
        assert c.rate_numerator == 24

    def test_add_different_rates_preserves_left_rate(self):
        a = RationalTime.from_seconds(10.0, RATE_24)
        b = RationalTime.from_seconds(5.0, RATE_29_97)
        c = a + b
        assert c.rate_numerator == 24

    def test_radd_with_zero(self):
        rt = RationalTime.from_seconds(10.0, RATE_24)
        result = sum([rt, rt], RationalTime(0, 1, 24, 1))
        assert abs(result.to_seconds() - 20.0) < 1e-9

    def test_sub(self):
        a = RationalTime.from_seconds(10.0, RATE_24)
        b = RationalTime.from_seconds(3.0, RATE_24)
        c = a - b
        assert abs(c.to_seconds() - 7.0) < 1e-9

    def test_sub_negative_raises(self):
        a = RationalTime.from_seconds(3.0, RATE_24)
        b = RationalTime.from_seconds(10.0, RATE_24)
        with pytest.raises(ValueError, match="non-positive"):
            _ = a - b

    def test_sub_equal_raises(self):
        a = RationalTime.from_seconds(5.0, RATE_24)
        with pytest.raises(ValueError, match="non-positive"):
            _ = a - a

    def test_mul_scalar(self):
        rt = RationalTime.from_seconds(10.0, RATE_24)
        c = rt * 2
        assert abs(c.to_seconds() - 20.0) < 1e-9

    def test_mul_fraction(self):
        rt = RationalTime.from_seconds(10.0, RATE_24)
        c = rt * Fraction(1, 2)
        assert abs(c.to_seconds() - 5.0) < 1e-9

    def test_mul_float(self):
        rt = RationalTime.from_seconds(10.0, RATE_24)
        c = rt * 0.5
        assert abs(c.to_seconds() - 5.0) < 0.001


# ===================================================================
# Comparison
# ===================================================================


class TestComparison:
    def test_eq_same_rate(self):
        a = RationalTime.from_seconds(10.0, RATE_24)
        b = RationalTime.from_seconds(10.0, RATE_24)
        assert a == b

    def test_eq_different_rate_checks_both(self):
        """Even if time is the same, different rates make them unequal."""
        a = RationalTime(10, 1, 24, 1)
        b = RationalTime(10, 1, 30000, 1001)
        assert a != b

    def test_lt(self):
        a = RationalTime.from_seconds(5.0, RATE_24)
        b = RationalTime.from_seconds(10.0, RATE_24)
        assert a < b
        assert not b < a

    def test_le(self):
        a = RationalTime.from_seconds(5.0, RATE_24)
        b = RationalTime.from_seconds(10.0, RATE_24)
        c = RationalTime.from_seconds(5.0, RATE_24)
        assert a <= b
        assert a <= c

    def test_gt(self):
        a = RationalTime.from_seconds(10.0, RATE_24)
        b = RationalTime.from_seconds(5.0, RATE_24)
        assert a > b

    def test_ge(self):
        a = RationalTime.from_seconds(10.0, RATE_24)
        b = RationalTime.from_seconds(5.0, RATE_24)
        c = RationalTime.from_seconds(10.0, RATE_24)
        assert a >= b
        assert a >= c

    def test_not_implemented(self):
        rt = RationalTime(10, 1)
        assert rt.__eq__("not a time") is NotImplemented
        assert rt.__lt__("not a time") is NotImplemented


# ===================================================================
# with_rate
# ===================================================================


class TestWithRate:
    def test_preserves_time_changes_rate(self):
        # 10 seconds at 24 fps → 10 seconds at 29.97 fps
        rt = RationalTime.from_seconds(10.0, RATE_24)
        rt2 = rt.with_rate(RATE_29_97)
        assert abs(rt2.to_seconds() - 10.0) < 1e-9
        assert rt2.rate_numerator == 30000
        assert rt2.rate_denominator == 1001
        # Frame count changes because rate changed
        assert rt.to_frames() == 240   # 10 s @ 24 fps
        assert rt2.to_frames() == 300  # 10 s @ 29.97 fps


# ===================================================================
# Property-based round-trip tests
# ===================================================================


class TestRoundTrip:
    """Rigorous round-trip tests for common frame rates."""

    @pytest.mark.parametrize("rate", [
        RATE_23_976,
        RATE_24,
        RATE_25,
        RATE_29_97,
        RATE_30,
        RATE_48,
        RATE_50,
        RATE_59_94,
        RATE_60,
    ])
    @pytest.mark.parametrize("seconds", [1, 5, 10, 30, 60, 300, 600, 3600])
    def test_seconds_round_trip(self, rate, seconds):
        """from_seconds(s, rate).to_seconds() ≈ s within float precision."""
        rt = RationalTime.from_seconds(float(seconds), rate)
        recovered = rt.to_seconds()
        # Allow small tolerance for float → Fraction → float
        assert abs(recovered - seconds) / max(seconds, 1) < 1e-6

    @pytest.mark.parametrize("rate", [
        RATE_23_976,
        RATE_24,
        RATE_29_97,
        RATE_30,
    ])
    @pytest.mark.parametrize("frames", [1, 10, 100, 1000, 10000])
    def test_frames_round_trip(self, rate, frames):
        """from_frames(f, rate).to_frames() == f."""
        rt = RationalTime.from_frames(frames, rate)
        recovered = rt.to_frames()
        assert recovered == frames


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_very_large_duration(self):
        """24-hour timeline at 23.976 fps should not overflow."""
        hours_24 = 24 * 3600  # 86400 s
        rt = RationalTime.from_seconds(float(hours_24), RATE_23_976)
        assert abs(rt.to_seconds() - hours_24) < 1.0

    def test_to_fcpxml_large_timeline_rate_differs(self):
        """24-hour at 29.97 fps expressed in 24 fps timeline."""
        hours_24 = 24 * 3600
        rt = RationalTime.from_seconds(float(hours_24), RATE_29_97)
        result = rt.to_fcpxml_duration(RATE_24)
        # Should produce valid rational, not overflow
        assert "/" in result
        assert result.endswith("s")

    def test_different_rates_equal_time_not_equal(self):
        """Same seconds, different rates → not equal (different frame counts)."""
        a = RationalTime(10, 1, 24, 1)       # 10 s @ 24 fps
        b = RationalTime(10, 1, 30000, 1001)  # 10 s @ 29.97 fps
        assert a.to_seconds() == b.to_seconds()
        assert a != b

    def test_repr(self):
        rt = RationalTime(300, 1)
        r = repr(rt)
        assert "RationalTime" in r
        assert "300/1" in r
        assert "24000/1001" in r


# ===================================================================
# Rate constants
# ===================================================================


class TestRateConstants:
    def test_23_976_rate(self):
        assert RATE_23_976.rate_numerator == 24000
        assert RATE_23_976.rate_denominator == 1001

    def test_24_rate(self):
        assert RATE_24.rate_numerator == 24
        assert RATE_24.rate_denominator == 1

    def test_29_97_rate(self):
        assert RATE_29_97.rate_numerator == 30000
        assert RATE_29_97.rate_denominator == 1001

    def test_59_94_rate(self):
        assert RATE_59_94.rate_numerator == 60000
        assert RATE_59_94.rate_denominator == 1001

    def test_60_rate(self):
        assert RATE_60.rate_numerator == 60
        assert RATE_60.rate_denominator == 1
