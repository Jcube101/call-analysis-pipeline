"""
Tests for Stage 2's speaker re-identification outcome (stages/diarize.py).

An exception to the rule against unit tests for stages/diarize.py: nothing here
downloads a model or needs a GPU. pyannote, torch and huggingface_hub are
replaced by stubs, the audio is a few seconds of generated tones, and only the
MFCC + KMeans re-identification pass runs for real (librosa, scikit-learn).
Failures are forced by patching the feature step or the clustering step, not by
running out of memory.

diarize.py is loaded under a private module name, because tests/test_api.py
puts an empty stub at sys.modules["stages.diarize"] for the rest of the session.
"""

import importlib.util
import io
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from stages import reid_outcome

SR = 16000
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Loading diarize.py without pyannote
# ---------------------------------------------------------------------------

def _load_diarize():
    pyannote = types.ModuleType("pyannote")
    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = object  # replaced per test via _FakePipeline
    pyannote.audio = audio
    saved = {k: sys.modules.get(k) for k in ("pyannote", "pyannote.audio")}
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = audio
    try:
        spec = importlib.util.spec_from_file_location(
            "_diarize_under_test", os.path.join(REPO, "stages", "diarize.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


diarize = _load_diarize()


# ---------------------------------------------------------------------------
# Fakes for pyannote's pipeline, torch and huggingface_hub
# ---------------------------------------------------------------------------

class _Segment:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Annotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for i, (start, end, label) in enumerate(self._tracks):
            yield _Segment(start, end), i, label


class _FakePipeline:
    tracks: list = []

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def to(self, device):
        return self

    def __call__(self, path, hook=None, **kwargs):
        return _Annotation(self.tracks)


class _FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def empty_cache():
        pass


@pytest.fixture
def stubbed(monkeypatch):
    """Stub everything run() touches outside re-identification."""
    torch = types.ModuleType("torch")
    torch.cuda = _FakeCuda()
    torch.device = lambda name: name
    # scipy's array-API helpers look up torch.Tensor in sys.modules["torch"].
    torch.Tensor = type("Tensor", (), {})
    hub = types.ModuleType("huggingface_hub")
    hub.login = lambda token=None: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(diarize, "Pipeline", _FakePipeline)
    monkeypatch.setattr(diarize, "_check_memory_headroom", lambda path: None)
    monkeypatch.setattr(diarize.settings, "validate_for_diarization", lambda: None)
    monkeypatch.setattr(diarize, "vram_status", lambda: None)
    return _FakePipeline


# ---------------------------------------------------------------------------
# Audio: two clearly different "voices", alternating one second at a time
# ---------------------------------------------------------------------------

def _voice(freq, seconds, seed):
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    tone = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 6))
    return (0.3 * tone + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


# Truth: A B A B A B A B, one second each.
TRUTH = ["SPEAKER_00", "SPEAKER_01"] * 4


@pytest.fixture
def wav(tmp_path):
    chunks = [_voice(140 if i % 2 == 0 else 900, 1.0, seed=i) for i in range(8)]
    path = str(tmp_path / "clean.wav")
    sf.write(path, np.concatenate(chunks), SR)
    return path


# pyannote's labels: two speakers, assigned wrongly and named so that any
# rewrite by re-identification (which emits SPEAKER_00/01) is visible.
PYANNOTE = ["SPEAKER_07", "SPEAKER_05", "SPEAKER_05", "SPEAKER_07"] * 2


def _tracks(labels=None):
    """pyannote's view: eight one-second segments, labelled wrongly on purpose."""
    labels = labels or PYANNOTE
    return [(float(i), float(i + 1), labels[i]) for i in range(8)]


def _run(stubbed, wav, tracks, num_speakers=2):
    stubbed.tracks = tracks
    outcomes = []
    segments = diarize.run(
        clean_wav_path=wav,
        output_dir=os.path.dirname(wav),
        num_speakers=num_speakers,
        on_reidentification=outcomes.append,
    )
    assert len(outcomes) == 1
    return segments, outcomes[0]


# ---------------------------------------------------------------------------
# ok
# ---------------------------------------------------------------------------

def test_ok_relabels_by_voice_and_reports_ok(stubbed, wav):
    segments, outcome = _run(stubbed, wav, _tracks())
    assert [s["label"] for s in segments] == TRUTH
    assert [s["speaker"] for s in segments] == ["Speaker A", "Speaker B"] * 4
    assert outcome["status"] == "ok"
    assert outcome["reason"] is None
    assert outcome["total_segments"] == 8
    assert outcome["clustered_segments"] == 8
    assert outcome["failed_segments"] == 0
    assert outcome["short_segments"] == 0
    assert outcome["speakers_requested"] == 2
    assert outcome["speakers_found"] == 2
    assert reid_outcome.warnings_for(outcome) == []


def test_run_without_callback_still_returns_plain_segment_list(stubbed, wav):
    stubbed.tracks = _tracks()
    segments = diarize.run(clean_wav_path=wav, output_dir=os.path.dirname(wav), num_speakers=2)
    assert isinstance(segments, list)
    assert set(segments[0]) == {"start", "end", "label", "speaker"}


def test_callback_error_is_swallowed(stubbed, wav):
    stubbed.tracks = _tracks()

    def boom(outcome):
        raise RuntimeError("callback broke")

    segments = diarize.run(
        clean_wav_path=wav, output_dir=os.path.dirname(wav), num_speakers=2,
        on_reidentification=boom,
    )
    assert len(segments) == 8


def test_short_segments_do_not_count_as_failures(stubbed, wav):
    tracks = _tracks() + [(7.6, 7.9, "SPEAKER_05")]
    segments, outcome = _run(stubbed, wav, tracks)
    assert outcome["status"] == "ok"
    assert outcome["short_segments"] == 1
    assert outcome["clustered_segments"] == 8
    assert segments[-1]["label"] == segments[7]["label"]


def test_non_utf8_stdout_no_longer_breaks_reidentification(stubbed, wav, monkeypatch):
    """KNOWN-ISSUES #3: the feature line used to print a character cp1252 cannot encode."""
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    segments, outcome = _run(stubbed, wav, _tracks())
    assert outcome["status"] == "ok"
    assert [s["label"] for s in segments] == TRUTH


# ---------------------------------------------------------------------------
# partial
# ---------------------------------------------------------------------------

def _fail_first_mfcc_calls(monkeypatch, n):
    import librosa

    real = librosa.feature.mfcc
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= n:
            raise MemoryError("Unable to allocate 1.26 MiB (forced)")
        return real(*args, **kwargs)

    monkeypatch.setattr(librosa.feature, "mfcc", flaky)


def test_partial_when_some_segments_fail_feature_extraction(stubbed, wav, monkeypatch):
    _fail_first_mfcc_calls(monkeypatch, 3)
    segments, outcome = _run(stubbed, wav, _tracks())
    assert outcome["status"] == "partial"
    assert outcome["reason"] == "feature_extraction"
    assert outcome["failed_segments"] == 3
    assert outcome["clustered_segments"] == 5
    assert "Unable to allocate" in outcome["error"]
    # Clustering still ran: no segment keeps pyannote's label.
    assert not {s["label"] for s in segments} & set(PYANNOTE)
    assert reid_outcome.warnings_for(outcome) == [
        "Speaker re-identification failed for 3 of 8 segments; labels may be less accurate."
    ]


def test_one_failed_segment_is_enough_for_partial(stubbed, wav, monkeypatch):
    _fail_first_mfcc_calls(monkeypatch, 1)
    _, outcome = _run(stubbed, wav, _tracks())
    assert outcome["status"] == "partial"
    assert outcome["failed_segments"] == 1


# ---------------------------------------------------------------------------
# failed -- labels must be pyannote's, untouched
# ---------------------------------------------------------------------------

def _assert_pyannote_labels(segments, tracks):
    assert [s["label"] for s in segments] == [t[2] for t in tracks]


def test_failed_case1_sklearn_missing(stubbed, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    tracks = _tracks()
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "sklearn_missing"
    assert "scikit-learn is not installed" in reid_outcome.warnings_for(outcome)[0]


def test_failed_case2_exception_before_any_audio_is_read(stubbed, wav, monkeypatch):
    """What KNOWN-ISSUES #3 used to look like: the stage raises before extraction."""
    monkeypatch.setitem(sys.modules, "librosa", None)
    tracks = _tracks()
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "exception"
    assert outcome["clustered_segments"] == 0
    assert reid_outcome.warnings_for(outcome) == [
        "Speaker re-identification failed; speaker labels come straight from pyannote "
        "and may be less accurate."
    ]


def test_failed_case3_wav_cannot_be_opened(stubbed, wav, monkeypatch):
    real = sf.SoundFile

    def unreadable(*args, **kwargs):
        # Only re-identification opens the file with mode="r"; sf.info(), which
        # run() uses for the sample rate, goes through SoundFile too.
        if kwargs.get("mode") == "r":
            raise RuntimeError("Error opening clean.wav (forced)")
        return real(*args, **kwargs)

    monkeypatch.setattr(sf, "SoundFile", unreadable)
    tracks = _tracks()
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "exception"
    assert "Error opening" in outcome["error"]


def test_failed_case5_too_few_usable_segments(stubbed, wav, monkeypatch):
    _fail_first_mfcc_calls(monkeypatch, 7)
    tracks = _tracks()
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "too_few_features"
    assert outcome["clustered_segments"] == 1
    assert outcome["failed_segments"] == 7
    assert reid_outcome.warnings_for(outcome)[0].startswith(
        "Speaker re-identification skipped: only 1 of 8 segments could be analysed for 2 speakers"
    )


def test_failed_case5_all_segments_short(stubbed, wav):
    tracks = [(i * 0.4, i * 0.4 + 0.3, "SPEAKER_00" if i % 2 else "SPEAKER_01") for i in range(6)]
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "too_few_features"
    assert outcome["short_segments"] == 6
    assert outcome["failed_segments"] == 0


class _ExplodingKMeans:
    def __init__(self, *args, **kwargs):
        pass

    def fit_predict(self, X):
        raise OSError("exception: access violation writing 0x0000000000000000 (forced)")


def test_failed_case6_clustering_raises_keeps_failed_counts(stubbed, wav, monkeypatch):
    """The 2026-10-14 job: some segments fail extraction, then clustering crashes."""
    import sklearn.cluster

    _fail_first_mfcc_calls(monkeypatch, 5)
    monkeypatch.setattr(sklearn.cluster, "KMeans", _ExplodingKMeans)
    tracks = _tracks()
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "exception"
    assert outcome["failed_segments"] == 5
    assert "access violation" in outcome["error"]
    assert reid_outcome.warnings_for(outcome) == [
        "Speaker re-identification failed for 5 of 8 segments and then stopped; "
        "speaker labels come straight from pyannote and may be less accurate."
    ]


class _PoisonedEnd(float):
    """Subtracts normally (so duration works) but cannot be added (midpoint fails)."""

    def __add__(self, other):
        raise ArithmeticError("forced failure while labelling short segments")

    __radd__ = __add__


def test_case7_failure_after_clustering_leaves_every_label_untouched(stubbed, wav):
    # Eight long segments get cluster labels first; the short one after them
    # then raises while its inherited label is worked out. Labels used to be
    # written as they were computed, so the long ones came back relabelled.
    tracks = _tracks() + [(7.6, _PoisonedEnd(7.9), "SPEAKER_05")]
    segments, outcome = _run(stubbed, wav, tracks)
    _assert_pyannote_labels(segments, tracks)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "exception"
    assert "forced failure" in outcome["error"]


# ---------------------------------------------------------------------------
# Fewer speakers than requested
# ---------------------------------------------------------------------------

class _OneClusterKMeans:
    def __init__(self, *args, **kwargs):
        pass

    def fit_predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_fewer_speakers_than_requested_warns(stubbed, wav, monkeypatch):
    import sklearn.cluster

    monkeypatch.setattr(sklearn.cluster, "KMeans", _OneClusterKMeans)
    _, outcome = _run(stubbed, wav, _tracks())
    assert outcome["status"] == "ok"
    assert outcome["speakers_found"] == 1
    assert reid_outcome.warnings_for(outcome) == [
        "Only 1 speaker found but 2 were requested; one speaker may have absorbed the other."
    ]


def test_fewer_speakers_in_auto_detect_mode_does_not_warn(stubbed, wav, monkeypatch):
    import sklearn.cluster

    monkeypatch.setattr(sklearn.cluster, "KMeans", _OneClusterKMeans)
    tracks = _tracks(["SPEAKER_00", "SPEAKER_01"] * 4)
    _, outcome = _run(stubbed, wav, tracks, num_speakers=None)
    assert outcome["speakers_requested"] is None
    assert outcome["speakers_found"] == 1
    assert reid_outcome.warnings_for(outcome) == []


# ---------------------------------------------------------------------------
# reid_outcome on its own
# ---------------------------------------------------------------------------

def test_partial_threshold_is_one_failed_segment():
    assert reid_outcome.PARTIAL_MIN_FAILED_SEGMENTS == 1
    assert reid_outcome.status_after_clustering(0) == "ok"
    assert reid_outcome.status_after_clustering(1) == "partial"


def test_warning_for_the_2026_10_14_numbers():
    outcome = reid_outcome.new_outcome(102, 2)
    outcome.update(status="partial", failed_segments=80, clustered_segments=22, speakers_found=2)
    assert reid_outcome.warnings_for(outcome) == [
        "Speaker re-identification failed for 80 of 102 segments; labels may be less accurate."
    ]


def test_no_outcome_no_warnings():
    assert reid_outcome.warnings_for(None) == []
    assert reid_outcome.warnings_for({}) == []
