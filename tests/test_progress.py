"""Tests for stages/progress.py -- the Stage 1 and 3 progress throttle."""

import pytest

from stages import progress as progress_mod
from stages.progress import PhaseProgress


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(progress_mod.time, "monotonic", c)
    return c


def test_no_callback_is_a_no_op(clock):
    p = PhaseProgress(None, "[Stage 1]")
    p.begin("noise reduction", counted=True)
    p.update(5, 10, "chunks")  # must not raise


def test_begin_sends_zero_for_counted_and_none_otherwise(clock):
    calls = []
    p = PhaseProgress(lambda *a: calls.append(a), "[Stage 1]")
    p.begin("decode")
    p.begin("noise reduction", counted=True)
    assert calls == [("decode", None, "Phase: decode"), ("noise reduction", 0, "Phase: noise reduction")]


def test_update_throttles_by_step_and_gap(clock):
    calls = []
    p = PhaseProgress(lambda *a: calls.append(a), "[Stage 3]")
    p.begin("transcription", counted=True)
    calls.clear()
    clock.now += 5
    p.update(50, 100, "segments")       # < 10 s since begin
    clock.now += 6
    p.update(52, 100, "segments")       # 11 s, 52% -> sent
    clock.now += 20
    p.update(54, 100, "segments")       # only 2% more
    p.update(70, 100, "segments")       # 20 s, 18% more -> sent
    clock.now += 1
    p.update(90, 100, "segments")       # 1 s since last send
    clock.now += 10
    p.update(100, 100, "segments")      # 100 is never sent
    assert [c[1] for c in calls] == [52, 70]
    assert calls[0][2].startswith("transcription 52% (52/100 segments), 0:11 elapsed")


def test_percent_is_rounded(clock):
    calls = []
    p = PhaseProgress(lambda *a: calls.append(a), "[Stage 1]")
    p.begin("noise reduction", counted=True)
    clock.now += 11
    p.update(2, 7, "chunks")  # 28.57%
    assert calls[-1][1] == 29


def test_callback_errors_are_swallowed(clock, capsys):
    def boom(*a):
        raise RuntimeError("socket gone")
    p = PhaseProgress(boom, "[Stage 3]")
    p.begin("model load")
    assert "[Stage 3] Progress callback failed (ignored): socket gone" in capsys.readouterr().out
