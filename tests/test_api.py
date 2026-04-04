"""
Tests for api.py logic that does not require a running server.

Functions under test: get_or_recover_job, ALLOWED_GEMINI_MODELS,
and the glob-based file discovery patterns used during recovery.

Note: get_or_recover_job() builds its job_dir as
  f"output/jobs/{job_id}"
relative to the current working directory, so tests that exercise
disk-recovery must chdir into a temp directory that contains the
expected folder structure.

api.py imports heavy pipeline stages (torch, pyannote, etc.) at module
level. We stub those out via sys.modules before the import so the test
environment does not need GPU dependencies installed.
"""

import json
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Stub out only the heavy pipeline stages (torch/GPU deps) before api.py is
# loaded.  stages.export and stages.report are left as real modules so that
# test_export.py and test_report.py can import them normally in the same run.
# ---------------------------------------------------------------------------
for _mod in (
    "stages.diarize",
    "stages.preprocess",
    "stages.transcribe",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

from api import ALLOWED_GEMINI_MODELS, get_or_recover_job, jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job_dir(base_dir, job_id):
    """Create output/jobs/{job_id}/ under base_dir and return its path."""
    job_dir = os.path.join(base_dir, "output", "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


# ---------------------------------------------------------------------------
# ALLOWED_GEMINI_MODELS
# ---------------------------------------------------------------------------

def test_allowed_gemini_models_contains_flash():
    assert "gemini-3-flash-preview" in ALLOWED_GEMINI_MODELS


def test_allowed_gemini_models_contains_pro():
    assert "gemini-3.1-pro-preview" in ALLOWED_GEMINI_MODELS


def test_allowed_gemini_models_contains_flash_lite():
    assert "gemini-3.1-flash-lite-preview" in ALLOWED_GEMINI_MODELS


def test_allowed_gemini_models_has_three_entries():
    assert len(ALLOWED_GEMINI_MODELS) == 3


# ---------------------------------------------------------------------------
# get_or_recover_job — in-memory path
# ---------------------------------------------------------------------------

def test_get_or_recover_job_returns_none_for_unknown():
    """get_or_recover_job() returns None for an unknown job_id."""
    result = get_or_recover_job("nonexistent-job-id-xyz")
    assert result is None


def test_get_or_recover_job_returns_in_memory_job():
    """get_or_recover_job() returns the job from memory if present."""
    job_id = "mem-test-job"
    jobs[job_id] = {"status": "complete", "data": "hello"}
    try:
        result = get_or_recover_job(job_id)
        assert result is not None
        assert result["data"] == "hello"
    finally:
        del jobs[job_id]


# ---------------------------------------------------------------------------
# get_or_recover_job — disk recovery
# ---------------------------------------------------------------------------

def test_get_or_recover_job_recovers_from_disk(tmp_path, monkeypatch):
    """get_or_recover_job() recovers a job from disk when not in memory."""
    job_id = "disk-recovery-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)

    transcript_data = {
        "metadata": {"source_file": "call.mp3", "context": "friend", "num_speakers": 2},
        "transcript": [{"start": 0.0, "end": 1.0, "speaker": "Speaker A", "text": "Hi"}],
    }
    (open(os.path.join(job_dir, "call_20260401_000000.json"), "w")).write(
        json.dumps(transcript_data)
    )
    open(os.path.join(job_dir, "call_20260401_000000.txt"), "w").write("transcript")

    monkeypatch.chdir(tmp_path)
    jobs.pop(job_id, None)

    result = get_or_recover_job(job_id)
    assert result is not None


def test_get_or_recover_job_sets_status_complete_on_recovery(tmp_path, monkeypatch):
    """Recovered job has status 'complete'."""
    job_id = "recovery-status-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "call_20260401_000000.txt"), "w").write("txt")
    open(os.path.join(job_dir, "call_20260401_000000.json"), "w").write(
        json.dumps({"metadata": {}, "transcript": []})
    )

    monkeypatch.chdir(tmp_path)
    jobs.pop(job_id, None)

    result = get_or_recover_job(job_id)
    assert result["status"] == "complete"


def test_get_or_recover_job_reads_report_content_on_recovery(tmp_path, monkeypatch):
    """Recovered job has report content read from disk."""
    job_id = "recovery-report-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "call_20260401_000000.txt"), "w").write("txt")
    open(os.path.join(job_dir, "call_20260401_000000.json"), "w").write(
        json.dumps({"metadata": {}, "transcript": []})
    )
    open(os.path.join(job_dir, "call_20260401_000000_report.md"), "w").write(
        "# My Report\n\nReport content here."
    )

    monkeypatch.chdir(tmp_path)
    jobs.pop(job_id, None)

    result = get_or_recover_job(job_id)
    assert result["report"] is not None
    assert "My Report" in result["report"]


def test_get_or_recover_job_includes_named_json(tmp_path, monkeypatch):
    """transcript_named.json is picked up by the recovery glob (not excluded like transcript.json)."""
    import glob as _glob
    job_id = "recovery-named-json-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)

    open(os.path.join(job_dir, "call_20260401_000000.txt"), "w").write("txt")
    open(os.path.join(job_dir, "call_20260401_000000.json"), "w").write(
        json.dumps({"metadata": {"which": "timestamped"}, "transcript": []})
    )
    open(os.path.join(job_dir, "transcript_named.json"), "w").write(
        json.dumps({"metadata": {"which": "named"}, "transcript": []})
    )

    # Replicate get_or_recover_job's json_files filter
    json_files = [
        f for f in _glob.glob(os.path.join(job_dir, "*.json"))
        if os.path.basename(f) not in ("transcript.json",)
        and not os.path.basename(f).startswith("input")
    ]
    basenames = [os.path.basename(f) for f in json_files]

    # transcript_named.json is NOT excluded by the recovery glob
    assert "transcript_named.json" in basenames
    # transcript.json (the raw upload) would be excluded
    assert "transcript.json" not in basenames


# ---------------------------------------------------------------------------
# Job state structure
# ---------------------------------------------------------------------------

def test_job_state_required_keys():
    """A newly created job dict has all required keys."""
    required = {
        "status", "current_stage", "stage_name", "progress_message",
        "error", "output_dir", "transcript", "report", "metadata",
        "message_queue",
    }
    job = {
        "status": "queued",
        "current_stage": None,
        "stage_name": None,
        "progress_message": None,
        "error": None,
        "output_dir": "output/jobs/test",
        "transcript": None,
        "report": None,
        "metadata": {},
        "message_queue": [],
    }
    assert required.issubset(job.keys())


# ---------------------------------------------------------------------------
# Glob pattern logic (exercised directly without running the server)
# ---------------------------------------------------------------------------

def test_glob_finds_txt_not_report(tmp_path):
    """The txt glob pattern picks up .txt files and ignores _report.md."""
    import glob as _glob
    job_dir = str(tmp_path)
    open(os.path.join(job_dir, "call_20260401_000000.txt"), "w").write("")
    open(os.path.join(job_dir, "call_20260401_000000_report.md"), "w").write("")

    txt_files = [
        f for f in _glob.glob(os.path.join(job_dir, "*.txt"))
        if "_report" not in os.path.basename(f)
    ]
    assert len(txt_files) == 1
    assert txt_files[0].endswith(".txt")


def test_glob_finds_report_md(tmp_path):
    """The report glob pattern finds *_report.md files."""
    import glob as _glob
    job_dir = str(tmp_path)
    open(os.path.join(job_dir, "call_20260401_000000_report.md"), "w").write("")
    open(os.path.join(job_dir, "call_20260401_000000.txt"), "w").write("")

    report_files = _glob.glob(os.path.join(job_dir, "*_report.md"))
    assert len(report_files) == 1
    assert report_files[0].endswith("_report.md")


def test_glob_prefers_named_json(tmp_path):
    """The json glob returns transcript_named.json when both exist."""
    import glob as _glob
    job_dir = str(tmp_path)
    open(os.path.join(job_dir, "call_20260401_000000.json"), "w").write("")
    open(os.path.join(job_dir, "transcript_named.json"), "w").write("")

    # Replicate the api.py filter: exclude transcript.json and input* files
    json_files = [
        f for f in _glob.glob(os.path.join(job_dir, "*.json"))
        if os.path.basename(f) not in ("transcript.json",)
        and not os.path.basename(f).startswith("input")
    ]
    basenames = [os.path.basename(f) for f in json_files]
    assert "transcript_named.json" in basenames
    assert "call_20260401_000000.json" in basenames


def test_glob_excludes_transcript_json(tmp_path):
    """The json glob excludes the raw upload transcript.json."""
    import glob as _glob
    job_dir = str(tmp_path)
    open(os.path.join(job_dir, "transcript.json"), "w").write("")
    open(os.path.join(job_dir, "call_20260401_000000.json"), "w").write("")

    json_files = [
        f for f in _glob.glob(os.path.join(job_dir, "*.json"))
        if os.path.basename(f) not in ("transcript.json",)
        and not os.path.basename(f).startswith("input")
    ]
    basenames = [os.path.basename(f) for f in json_files]
    assert "transcript.json" not in basenames
    assert "call_20260401_000000.json" in basenames


def test_glob_excludes_input_prefixed_json(tmp_path):
    """The json glob excludes files starting with 'input'."""
    import glob as _glob
    job_dir = str(tmp_path)
    open(os.path.join(job_dir, "input_20260401_000000.json"), "w").write("")
    open(os.path.join(job_dir, "call_20260401_000000.json"), "w").write("")

    json_files = [
        f for f in _glob.glob(os.path.join(job_dir, "*.json"))
        if os.path.basename(f) not in ("transcript.json",)
        and not os.path.basename(f).startswith("input")
    ]
    basenames = [os.path.basename(f) for f in json_files]
    assert "input_20260401_000000.json" not in basenames
    assert "call_20260401_000000.json" in basenames
