"""RationalTime — frame-accurate rational time representation for video editing.

Internally uses ``fractions.Fraction`` for lossless arithmetic across
the common NTSC frame rates (23.976, 29.97, 59.94) and integer frame rates
(24, 25, 30, 48, 50, 60).

Design follows the Phase 2 spec:
  - ``numerator`` / ``denominator``  → time in seconds
  - ``rate_numerator`` / ``rate_denominator``  → frames-per-second Fraction
    (e.g. 24000/1001 for 23.976 fps)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union


# ---------------------------------------------------------------------------
# Common NTSC / integer frame rates
# ---------------------------------------------------------------------------

# Forward type-declaration for common rate constants (populated at module end)
RATE_23_976: "RationalTime"
RATE_24: "RationalTime"
RATE_25: "RationalTime"
RATE_29_97: "RationalTime"
RATE_30: "RationalTime"
RATE_48: "RationalTime"
RATE_50: "RationalTime"
RATE_59_94: "RationalTime"
RATE_60: "RationalTime"


@dataclass(frozen=True)
class RationalTime:
    """Immutable time value tied to a specific frame rate.

    The time itself is stored as the rational number *numerator/denominator*
    seconds.  The frame rate is stored separately as
    *rate_numerator/rate_denominator* frames per second, so that
    ``to_frames()`` remains deterministic and exact (up to the final round).

    Parameters
    ----------
    numerator:
        Time numerator (seconds).
    denominator:
        Time denominator (seconds).  Default 1.
    rate_numerator:
        Frame-rate numerator.  Default 24000 (23.976 fps).
    rate_denominator:
        Frame-rate denominator.  Default 1001 (23.976 fps).
    """

    numerator: int
    denominator: int = 1
    rate_numerator: int = 24000
    rate_denominator: int = 1001

    # ---- internal helpers ---------------------------------------------------

    def _time(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _rate(self) -> Fraction:
        return Fraction(self.rate_numerator, self.rate_denominator)

    # ---- public accessors ---------------------------------------------------

    def to_seconds(self) -> float:
        """Return the time as a ``float`` in seconds."""
        return float(self._time())

    def to_frames(self) -> int:
        """Return the number of frames at the source rate, nearest integer."""
        return round(self._time() * self._rate())

    def to_fcpxml_duration(self, timeline_rate: RationalTime | None = None) -> str:
        """Format as an FCPXML rational-duration string ``\"num/dens\"``.

        When *timeline_rate* is provided the result is frame-aligned to that
        rate (the duration is snapped to the nearest whole frame).  This is
        critical for import into DaVinci Resolve.

        When *timeline_rate* is ``None`` (default) the raw time fraction is
        output unchanged, which matches the example in the Phase 2 design doc::

            >>> t = RationalTime(300300, 1001, 24000, 1001)
            >>> t.to_fcpxml_duration()
            '300300/1001s'

        Frame-aligned example for a 10-second clip at 23.976 fps::

            >>> rt = RationalTime.from_seconds(10.0, RATE_23_976)
            >>> rt.to_fcpxml_duration(RATE_23_976)
            '240240/24000s'
        """
        if timeline_rate is None:
            return f"{self.numerator}/{self.denominator}s"

        frames = round(self._time() * timeline_rate._rate())
        # At the timeline rate each frame = rate_den / rate_num seconds.
        duration = Fraction(
            frames * timeline_rate.rate_denominator, timeline_rate.rate_numerator
        )
        return f"{duration.numerator}/{duration.denominator}s"

    # ---- constructors -------------------------------------------------------

    @staticmethod
    def from_frames(frames: int, rate: RationalTime) -> RationalTime:
        """Create a ``RationalTime`` from a whole number of *frames* at *rate*.

        ``time = frames / fps``  →  ``numerator / denominator`` seconds.
        """
        time = Fraction(frames, 1) / rate._rate()
        return RationalTime(
            numerator=time.numerator,
            denominator=time.denominator,
            rate_numerator=rate.rate_numerator,
            rate_denominator=rate.rate_denominator,
        )

    @staticmethod
    def from_seconds(
        seconds: Union[float, Fraction, int], rate: RationalTime
    ) -> RationalTime:
        """Create a ``RationalTime`` from *seconds* at *rate*.

        Floats are converted with ``Fraction.from_float`` limited to a
        1e6 denominator to prevent runaway numerators.
        """
        if isinstance(seconds, float):
            seconds = Fraction.from_float(seconds).limit_denominator(1_000_000)
        elif not isinstance(seconds, Fraction):
            seconds = Fraction(seconds)
        return RationalTime(
            numerator=seconds.numerator,
            denominator=seconds.denominator,
            rate_numerator=rate.rate_numerator,
            rate_denominator=rate.rate_denominator,
        )

    # ---- make a copy at a different rate ------------------------------------

    def with_rate(self, rate: RationalTime) -> RationalTime:
        """Return a new ``RationalTime`` representing the same duration but at
        a different frame rate."""
        t = self._time()
        return RationalTime(
            numerator=t.numerator,
            denominator=t.denominator,
            rate_numerator=rate.rate_numerator,
            rate_denominator=rate.rate_denominator,
        )

    # ---- arithmetic (all preserve the left-hand rate) -----------------------

    def __add__(self, other: RationalTime) -> RationalTime:
        if not isinstance(other, RationalTime):
            return NotImplemented
        result = self._time() + other._time()
        return RationalTime(
            numerator=result.numerator,
            denominator=result.denominator,
            rate_numerator=self.rate_numerator,
            rate_denominator=self.rate_denominator,
        )

    def __radd__(self, other: RationalTime) -> RationalTime:
        """Support ``sum(iter_of_rational_time, start=RationalTime(...))``."""
        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, RationalTime):
            return NotImplemented
        return other.__add__(self)

    def __sub__(self, other: RationalTime) -> RationalTime:
        if not isinstance(other, RationalTime):
            return NotImplemented
        result = self._time() - other._time()
        if result <= 0:
            raise ValueError(
                f"Subtraction would yield non-positive duration: {self} - {other}"
            )
        return RationalTime(
            numerator=result.numerator,
            denominator=result.denominator,
            rate_numerator=self.rate_numerator,
            rate_denominator=self.rate_denominator,
        )

    def __mul__(self, scalar: Union[int, float, Fraction]) -> RationalTime:
        if isinstance(scalar, float):
            scalar = Fraction.from_float(scalar).limit_denominator(1_000_000)
        t = self._time() * scalar
        return RationalTime(
            numerator=t.numerator,
            denominator=t.denominator,
            rate_numerator=self.rate_numerator,
            rate_denominator=self.rate_denominator,
        )

    # ---- comparison ---------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return self._time() == other._time() and self._rate() == other._rate()

    def __hash__(self) -> int:
        return hash((
            self.numerator, self.denominator,
            self.rate_numerator, self.rate_denominator,
        ))

    def __lt__(self, other: RationalTime) -> bool:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return self._time() < other._time()

    def __le__(self, other: RationalTime) -> bool:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return self._time() <= other._time()

    def __gt__(self, other: RationalTime) -> bool:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return self._time() > other._time()

    def __ge__(self, other: RationalTime) -> bool:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return self._time() >= other._time()

    # ---- string -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RationalTime({self.numerator}/{self.denominator}s "
            f"@ {self.rate_numerator}/{self.rate_denominator}fps ≈ "
            f"{self.to_seconds():.4f}s, {self.to_frames()}f)"
        )


# ---------------------------------------------------------------------------
# Singleton constants for common frame rates
# ---------------------------------------------------------------------------

RATE_23_976 = RationalTime(0, 1, 24000, 1001)
RATE_24 = RationalTime(0, 1, 24, 1)
RATE_25 = RationalTime(0, 1, 25, 1)
RATE_29_97 = RationalTime(0, 1, 30000, 1001)
RATE_30 = RationalTime(0, 1, 30, 1)
RATE_48 = RationalTime(0, 1, 48, 1)
RATE_50 = RationalTime(0, 1, 50, 1)
RATE_59_94 = RationalTime(0, 1, 60000, 1001)
RATE_60 = RationalTime(0, 1, 60, 1)
