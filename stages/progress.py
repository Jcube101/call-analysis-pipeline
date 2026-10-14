"""
Optional progress reporting for Stages 1 and 3.

A stage's run() takes progress=None. When given, it is called as
progress(phase, percent, message) -- the same contract as diarize.run():

  - on each phase change, with percent 0 for a phase that counts its work
    (chunks, segments) and None for one that does not;
  - while a counted phase runs, at most every 5% and every 10 s, with the
    rounded percent of that phase done.

A phase is finished when the next one begins; 100 is never sent. Errors the
callback raises are printed and ignored, so it cannot fail the stage. Without
a callback every method is a no-op.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

ProgressCallback = Callable[[str, Optional[int], str], None]


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class PhaseProgress:
    """Phase changes and throttled percent updates for one stage."""

    STEP = 0.05      # an update every 5% ...
    MIN_GAP = 10.0   # ... but no more often than every 10 s

    def __init__(self, callback: Optional[ProgressCallback], log_prefix: str) -> None:
        self._callback = callback
        self._log_prefix = log_prefix
        self.phase: Optional[str] = None
        self._phase_start = 0.0
        self._last_update = 0.0
        self._last_fraction = 0.0

    def begin(self, phase: str, counted: bool = False) -> None:
        now = time.monotonic()
        self.phase = phase
        self._phase_start = self._last_update = now
        self._last_fraction = 0.0
        self._notify(phase, 0 if counted else None, f"Phase: {phase}")

    def update(self, completed: int, total: int, unit: str) -> None:
        """Report `completed` of `total` units done in the current phase."""
        if self._callback is None or not total or self.phase is None:
            return
        fraction = min(completed, total) / total
        now = time.monotonic()
        if (
            fraction >= 1.0  # the next phase follows immediately
            or fraction - self._last_fraction < self.STEP
            or now - self._last_update < self.MIN_GAP
        ):
            return
        elapsed = now - self._phase_start
        left = elapsed * (1 - fraction) / fraction if fraction > 0 else 0.0
        percent = round(fraction * 100)
        self._notify(
            self.phase, percent,
            f"{self.phase} {percent}% ({completed}/{total} {unit}), "
            f"{_fmt_elapsed(elapsed)} elapsed, ~{_fmt_elapsed(left)} left",
        )
        self._last_update = now
        self._last_fraction = fraction

    def _notify(self, phase: str, percent: Optional[int], message: str) -> None:
        if self._callback is None:
            return
        try:
            self._callback(phase, percent, message)
        except Exception as e:
            print(f"{self._log_prefix} Progress callback failed (ignored): {e}", flush=True)
