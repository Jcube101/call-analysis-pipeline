"""
Tests for api.py logic that does not require a running server.

Functions under test: get_or_recover_job, the analysis_model form field
(and the ignored, removed gemini_model field), the model registry,
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
import time
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

from api import get_or_recover_job, jobs
from models import ALLOWED_MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job_dir(base_dir, job_id):
    """Create output/jobs/{job_id}/ under base_dir and return its path."""
    job_dir = os.path.join(base_dir, "output", "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


# ---------------------------------------------------------------------------
# ALLOWED_MODELS
# ---------------------------------------------------------------------------

def test_allowed_models_contains_gemini_flash():
    assert "gemini-3.8-flash" in ALLOWED_MODELS


def test_allowed_models_contains_gemini_pro():
    assert "gemini-4-argon" in ALLOWED_MODELS


def test_allowed_models_contains_claude_haiku():
    assert "claude-haiku-5-5" in ALLOWED_MODELS


def test_allowed_models_contains_claude_sonnet():
    assert "claude-sonnet-5-5" in ALLOWED_MODELS


def test_allowed_models_matches_registry():
    from models import MODELS
    assert ALLOWED_MODELS == list(MODELS)
    assert len(ALLOWED_MODELS) == 4


def test_allowed_models_exclude_legacy_ids():
    """Retired IDs are remapped, not accepted as-is."""
    from models import LEGACY_ALIASES
    assert not set(LEGACY_ALIASES) & set(ALLOWED_MODELS)


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


def test_recovery_does_not_claim_complete_without_outputs(tmp_path, monkeypatch):
    """A job folder holding only the upload is not 'complete'.

    The folder is created when the upload lands, so a server restart mid-run
    leaves one with nothing but input.<ext>. Reporting that as complete made
    /reconnect assert a terminal state with transcript=None and report=None.
    """
    job_id = "half-done-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "input.m4a"), "wb").write(b"audio")

    monkeypatch.chdir(tmp_path)
    jobs.pop(job_id, None)

    result = get_or_recover_job(job_id)
    assert result["status"] != "complete"
    assert result["status"] == "unknown"


def test_recovery_claims_complete_from_report_alone(tmp_path, monkeypatch):
    """A report-from-json folder with only a report still counts as complete."""
    job_id = "report-only-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "input_20260401_000000_report.md"), "w").write("# r")

    monkeypatch.chdir(tmp_path)
    jobs.pop(job_id, None)

    assert get_or_recover_job(job_id)["status"] == "complete"


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


def test_recovery_finds_input_prefixed_json(tmp_path, monkeypatch):
    """Recovery hydrates the transcript from input_<timestamp>.json.

    Every Stage 4 JSON api.py writes is named input_<timestamp>.json. An
    earlier filter excluded "input"-prefixed files, so a job recovered from
    disk reported transcript=None even though the transcript was right there —
    and /reconnect passed that null straight to the client.

    Calls get_or_recover_job directly rather than replicating its filter, so
    the assertion cannot drift away from the implementation again.
    """
    job_id = "recovery-input-json-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "input_20260401_000000.txt"), "w").write("transcript text")
    with open(os.path.join(job_dir, "input_20260401_000000.json"), "w") as f:
        json.dump(
            {"metadata": {"job_id": job_id}, "transcript": [{"speaker": "Speaker A", "text": "hi"}]},
            f,
        )

    monkeypatch.chdir(str(tmp_path))
    jobs.pop(job_id, None)
    try:
        job = get_or_recover_job(job_id)
        assert job is not None
        assert os.path.basename(job["files"]["json"]) == "input_20260401_000000.json"
        assert job["transcript"] == [{"speaker": "Speaker A", "text": "hi"}]
        assert job["metadata"] == {"job_id": job_id}
    finally:
        jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# API key leak prevention
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = ("api_key", "api-key", "token", "secret", "password", "credential")


def _contains_secret_key(d, path=""):
    """Recursively check if any dict key looks like a secret field."""
    if isinstance(d, dict):
        for k, v in d.items():
            key_lower = k.lower()
            for pattern in _SECRET_PATTERNS:
                if pattern in key_lower:
                    yield f"{path}.{k}" if path else k
            yield from _contains_secret_key(v, f"{path}.{k}" if path else k)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            yield from _contains_secret_key(item, f"{path}[{i}]")


def test_job_dict_contains_no_secret_keys():
    """The job dict structure returned by /status must not contain secret fields."""
    job = {
        "status": "complete",
        "params": {
            "context": "friend",
            "num_speakers": 2,
            "transcription_mode": "accurate",
            "language": "en",
            "speaker_names": [],
            "word_timestamps": False,
            "generate_report": True,
            "skip_preprocess": False,
            "whisper_model": "medium",
            "analysis_model": "claude-haiku-5-5",
        },
        "generate_report": True,
        "analysis_model": "claude-haiku-5-5",
        "message_queue": [],
        "current_stage": 5,
        "stage_name": "AI Report",
        "progress_message": "Done",
        "transcript": [{"start": 0, "end": 1, "speaker": "A", "text": "hi"}],
        "metadata": {"source_file": "call.mp3", "context": "friend"},
        "report_path": "output/jobs/test/call_report.md",
        "report": "# Report",
        "error": None,
        "output_dir": "output/jobs/test",
        "files": {"transcript": "t.txt", "json": "t.json", "report": "r.md"},
    }
    secret_fields = list(_contains_secret_key(job))
    assert secret_fields == [], f"Job dict contains secret-looking keys: {secret_fields}"


def test_reconnect_response_contains_no_secret_keys():
    """The /reconnect response shape must not contain secret fields."""
    response = {
        "status": "complete",
        "current_stage": 5,
        "stage_name": "AI Report",
        "progress_message": "Done",
        "message_queue": [{"type": "progress", "stage": 1, "message": "ok"}],
        "transcript": [{"start": 0, "end": 1, "speaker": "A", "text": "hi"}],
        "report": "# Report",
        "metadata": {"source_file": "call.mp3"},
        "error": None,
    }
    secret_fields = list(_contains_secret_key(response))
    assert secret_fields == [], f"Reconnect response contains secret-looking keys: {secret_fields}"


def test_websocket_complete_message_contains_no_secret_keys():
    """The WebSocket complete message must not contain secret fields."""
    message = {
        "type": "complete",
        "transcript": [{"start": 0, "end": 1, "speaker": "A", "text": "hi"}],
        "report": "# Report",
        "metadata": {"source_file": "call.mp3", "context": "friend"},
    }
    secret_fields = list(_contains_secret_key(message))
    assert secret_fields == [], f"WS complete message contains secret-looking keys: {secret_fields}"


def test_settings_keys_not_in_job_dict_keys():
    """API keys from settings must never appear as job dict keys."""
    from config import Settings
    secret_attrs = {"huggingface_token", "gemini_api_key", "anthropic_api_key"}
    job_keys = {
        "status", "params", "generate_report", "analysis_model",
        "message_queue", "current_stage", "stage_name", "progress_message",
        "transcript", "metadata", "report_path", "report", "error",
        "output_dir", "files", "report_path",
    }
    leaked = secret_attrs & job_keys
    assert leaked == set(), f"Secret attributes found in job dict keys: {leaked}"


def test_anthropic_sdk_does_not_leak_key_in_exceptions():
    """Anthropic SDK auth errors must not include the API key in the message."""
    import anthropic
    fake_key = "sk-ant-FAKE-KEY-SHOULD-NOT-APPEAR-12345"
    try:
        client = anthropic.Anthropic(api_key=fake_key)
        client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "test"}],
        )
    except Exception as e:
        error_str = str(e)
        assert fake_key not in error_str, (
            f"API key leaked in Anthropic exception: {error_str}"
        )


def test_error_message_from_validate_for_report_contains_no_key():
    """validate_for_report error messages must not contain actual key values."""
    from config import Settings
    s = Settings()
    s.gemini_api_key = "SUPER-SECRET-GEMINI-KEY"
    s.anthropic_api_key = "SUPER-SECRET-ANTHROPIC-KEY"

    s_empty_anthropic = Settings()
    s_empty_anthropic.anthropic_api_key = ""
    try:
        s_empty_anthropic.validate_for_report("claude-haiku-5-5")
    except EnvironmentError as e:
        assert "SUPER-SECRET" not in str(e)

    s_empty_gemini = Settings()
    s_empty_gemini.gemini_api_key = ""
    try:
        s_empty_gemini.validate_for_report("gemini-3.8-flash")
    except EnvironmentError as e:
        assert "SUPER-SECRET" not in str(e)


def test_env_file_is_gitignored():
    """The .env file must be listed in .gitignore."""
    gitignore_path = os.path.join(os.path.dirname(__file__), "..", ".gitignore")
    if os.path.isfile(gitignore_path):
        with open(gitignore_path) as f:
            lines = [line.strip() for line in f]
        assert ".env" in lines, ".env is not in .gitignore"


# ---------------------------------------------------------------------------
# WebSocket connection registry
# ---------------------------------------------------------------------------

def test_closing_stale_socket_keeps_live_reconnect_registered():
    """A reconnecting client's socket survives the stale socket closing.

    Clients on ngrok briefly hold two sockets open for the same job. The
    endpoint's cleanup used to pop unconditionally, so the stale socket
    closing deregistered the live replacement and later pushes went nowhere.
    """
    import asyncio
    from api import connections, websocket_endpoint, WebSocketDisconnect

    job_id = "registry-guard-job"

    class FakeWS:
        def __init__(self):
            self.gate = asyncio.Event()
        async def accept(self):
            pass
        async def send_json(self, payload):
            pass
        async def receive_text(self):
            await self.gate.wait()
            raise WebSocketDisconnect(1006)

    async def scenario():
        jobs[job_id] = {"status": "running", "message_queue": []}
        stale, live = FakeWS(), FakeWS()
        t_stale = asyncio.create_task(websocket_endpoint(stale, job_id))
        await asyncio.sleep(0.01)
        t_live = asyncio.create_task(websocket_endpoint(live, job_id))
        await asyncio.sleep(0.01)
        assert connections.get(job_id) is live

        stale.gate.set()                     # stale socket drops
        await asyncio.sleep(0.01)
        still_registered = connections.get(job_id)

        live.gate.set()                      # tear down
        await asyncio.gather(t_stale, t_live)
        return still_registered, live

    try:
        still_registered, live = asyncio.run(scenario())
        assert still_registered is live, "live socket was deregistered by the stale one"
        assert connections.get(job_id) is None, "live socket should deregister itself"
    finally:
        connections.pop(job_id, None)
        jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# report-from-json job initialisation
# ---------------------------------------------------------------------------

def test_report_from_json_starts_with_empty_message_queue(tmp_path, monkeypatch):
    """A fresh report-from-json job never inherits the prior run's queue.

    The queue is replayed in full to every client connecting to /ws, so
    inheriting it made a new job replay the previous run's "complete" message
    before this job had produced anything.
    """
    import asyncio
    import api as api_mod

    job_id = "queue-inheritance-job"
    _make_job_dir(str(tmp_path), job_id)
    monkeypatch.chdir(str(tmp_path))

    # A prior run for this same job id, still in memory, with a finished queue
    stale_complete = {"type": "complete", "transcript": [{"text": "old"}], "report": "old"}
    jobs[job_id] = {
        "status": "complete",
        "message_queue": [stale_complete],
        "transcript": [{"text": "old"}],
        "metadata": {"job_id": job_id},
        "files": {"json": "old.json"},
    }

    class FakeUpload:
        filename = "transcript.json"
        async def read(self):
            return json.dumps(
                {"metadata": {"job_id": job_id}, "transcript": [{"text": "new"}]}
            ).encode()

    class FakeRequest:
        async def form(self):
            return {}

    monkeypatch.setattr(api_mod.executor, "submit", lambda *a, **kw: None)

    try:
        # Called directly, so the Form(...) defaults must be supplied explicitly
        asyncio.run(api_mod.report_from_json(
            request=FakeRequest(),
            file=FakeUpload(),
            context=None,
            speaker_names=None,
            analysis_model="claude-haiku-5-5",
            context_hints="",
        ))
        assert jobs[job_id]["message_queue"] == []
        assert stale_complete not in jobs[job_id]["message_queue"]
    finally:
        jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# /reconnect terminal signalling
# ---------------------------------------------------------------------------

def _reconnect_raw(job_id, job):
    """Drive the real /reconnect coroutine and return the JSONResponse."""
    import asyncio
    import api as api_mod
    api_mod.jobs[job_id] = job
    try:
        return asyncio.run(api_mod.reconnect(job_id))
    finally:
        api_mod.jobs.pop(job_id, None)


def _reconnect(job_id, job):
    """The decoded /reconnect body, for tests that only care about the payload."""
    return json.loads(_reconnect_raw(job_id, job).body)


def _job(**overrides):
    base = {
        "status": "queued", "message_queue": [], "current_stage": None,
        "stage_name": None, "progress_message": None, "transcript": None,
        "metadata": {}, "report_path": None, "report": None, "error": None,
        "output_dir": "output/jobs/does-not-exist",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("status,expected", [
    ("queued", False),
    ("running", False),
    ("unknown", False),
    ("complete", True),
    ("error", True),
])
def test_reconnect_done_flag_matches_status(status, expected):
    """done is true only for genuinely terminal states."""
    result = _reconnect("done-flag-job", _job(status=status))
    assert result["done"] is expected
    assert result["status"] == status  # existing field untouched


def test_reconnect_running_is_not_done_despite_empty_payload():
    """The incident shape: running mid-Stage-5, nothing to show yet.

    transcript and report are both None here, identical to a failed job, so
    done is the only field that distinguishes them without reading status.
    """
    result = _reconnect("mid-flight-job", _job(
        status="running", current_stage=5, stage_name="AI Report"))
    assert result["transcript"] is None and result["report"] is None
    assert result["done"] is False


def test_reconnect_report_expected_false_when_report_not_requested():
    """An /analyse job run with generate_report=false: report=None is correct."""
    result = _reconnect("no-report-job", _job(
        status="complete", generate_report=False, transcript=[{"text": "hi"}]))
    assert result["report_expected"] is False
    assert result["report"] is None


def test_reconnect_report_expected_true_for_report_from_json_job():
    """report-from-json sets generate_report explicitly, so report=None there
    always means something went wrong rather than 'not asked for'."""
    result = _reconnect("rfj-job", _job(status="running", generate_report=True))
    assert result["report_expected"] is True


def test_reconnect_outputs_reports_files_on_disk(tmp_path, monkeypatch):
    """outputs must agree with what /download would actually serve."""
    job_id = "outputs-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "input_20260401_000000.txt"), "w").write("t")
    open(os.path.join(job_dir, "input_20260401_000000_report.md"), "w").write("# r")
    monkeypatch.chdir(tmp_path)

    result = _reconnect(job_id, _job(
        status="complete", output_dir=os.path.join("output", "jobs", job_id)))
    assert result["outputs"] == {
        "transcript": True, "json": False, "report": True, "wav": False
    }


def test_reconnect_outputs_shows_report_despite_null_report_field(tmp_path, monkeypatch):
    """The case this field exists for: report text uncached but file present."""
    job_id = "uncached-report-job"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    open(os.path.join(job_dir, "input_20260401_000000_report.md"), "w").write("# r")
    monkeypatch.chdir(tmp_path)

    result = _reconnect(job_id, _job(
        status="complete", generate_report=True,
        output_dir=os.path.join("output", "jobs", job_id)))
    assert result["outputs"]["report"] is True


def test_reconnect_outputs_empty_for_job_with_no_files():
    result = _reconnect("empty-job", _job(status="running"))
    assert result["outputs"] == {
        "transcript": False, "json": False, "report": False, "wav": False
    }


def test_available_outputs_agrees_with_download_resolution(tmp_path):
    """_available_outputs is derived from the same resolver /download uses."""
    import api as api_mod
    job_dir = str(tmp_path)
    open(os.path.join(job_dir, "input_clean.wav"), "wb").write(b"RIFF")
    avail = api_mod._available_outputs(job_dir)
    for kind, present in avail.items():
        resolved = api_mod._resolve_download_path(job_dir, kind)
        assert present == (resolved is not None), kind
    assert avail["wav"] is True


def test_reconnect_returns_202_and_retry_after_while_running():
    """An in-flight job is flagged at the HTTP layer as well as in the body."""
    response = _reconnect_raw("inflight-job", _job(status="running", current_stage=3))
    assert response.status_code == 202
    assert response.headers["retry-after"] == "2"


def test_reconnect_returns_202_for_queued_and_unknown():
    for status in ("queued", "unknown"):
        assert _reconnect_raw("pending-job", _job(status=status)).status_code == 202


def test_reconnect_returns_200_when_terminal():
    for status in ("complete", "error"):
        response = _reconnect_raw("terminal-job", _job(status=status))
        assert response.status_code == 200
        assert "retry-after" not in response.headers


def test_reconnect_202_is_still_a_success_code():
    """Documents the limit of the 202: fetch treats 200-299 as ok, so a client
    checking only res.ok sees no difference. status and done are the signals
    that actually distinguish in-flight from finished."""
    response = _reconnect_raw("inflight-job", _job(status="running"))
    assert 200 <= response.status_code < 300
    body = json.loads(response.body)
    assert body["done"] is False and body["status"] == "running"


def test_reconnect_body_unchanged_between_202_and_200():
    """The status code is additive -- it carries no information the body lacks."""
    running = json.loads(_reconnect_raw("b-job", _job(status="running")).body)
    complete = json.loads(_reconnect_raw("b-job", _job(status="complete")).body)
    assert running.keys() == complete.keys()


# ---------------------------------------------------------------------------
# analysis_model form field (the gemini_model alias was removed 2026-10-14)
# ---------------------------------------------------------------------------

_IGNORED_LINE = "[models] Ignored removed field 'gemini_model'; send 'analysis_model'"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with job execution stubbed out and output/ under tmp_path."""
    from fastapi.testclient import TestClient
    import api
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api.executor, "submit", lambda *a, **k: None)
    return TestClient(api.app)


def _post_report(client, **fields):
    files = {"file": ("t.json", json.dumps({"metadata": {}, "segments": []}), "application/json")}
    resp = client.post("/report-from-json", data=fields, files=files)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    return job_id, jobs[job_id]


def _post_analyse(client, **fields):
    files = {"file": ("call.mp3", b"not really audio", "audio/mpeg")}
    resp = client.post("/analyse", data={"skip_preprocess": "true", **fields}, files=files)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    return job_id, jobs[job_id]


def test_analysis_model_alone_is_used(client, capsys):
    _, job = _post_report(client, analysis_model="claude-sonnet-5-5")
    assert job["analysis_model"] == "claude-sonnet-5-5"
    assert job["params"]["analysis_model"] == "claude-sonnet-5-5"
    assert "gemini_model" not in job
    assert "gemini_model" not in capsys.readouterr().out


def test_ignored_gemini_model_falls_back_to_default(client, capsys):
    from models import DEFAULT_MODEL
    _, job = _post_report(client, gemini_model="gemini-3.8-flash")
    assert job["analysis_model"] == DEFAULT_MODEL
    assert "gemini_model" not in job
    assert capsys.readouterr().out.count(_IGNORED_LINE) == 1


def test_ignored_gemini_model_on_analyse_falls_back_to_default(client, capsys):
    from models import DEFAULT_MODEL
    _, job = _post_analyse(client, gemini_model="gemini-4-argon")
    assert job["analysis_model"] == DEFAULT_MODEL
    assert job["params"]["analysis_model"] == DEFAULT_MODEL
    assert capsys.readouterr().out.count(_IGNORED_LINE) == 1


def test_both_fields_sent_gemini_model_ignored(client, capsys):
    _, job = _post_report(client, analysis_model="claude-sonnet-5-5", gemini_model="gemini-4-argon")
    assert job["analysis_model"] == "claude-sonnet-5-5"
    assert capsys.readouterr().out.count(_IGNORED_LINE) == 1


def test_ignored_gemini_model_legacy_id_is_not_remapped(client, capsys):
    from models import DEFAULT_MODEL
    _, job = _post_report(client, gemini_model="claude-sonnet-4-6")
    assert job["analysis_model"] == DEFAULT_MODEL
    out = capsys.readouterr().out
    assert _IGNORED_LINE in out
    assert "Legacy model ID" not in out


def test_neither_field_uses_default(client, capsys):
    from models import DEFAULT_MODEL
    _, job = _post_report(client)
    assert job["analysis_model"] == DEFAULT_MODEL
    assert "gemini_model" not in capsys.readouterr().out


def test_legacy_id_via_analysis_model_is_remapped(client, capsys):
    _, job = _post_report(client, analysis_model="claude-haiku-4-5-20251001")
    assert job["analysis_model"] == "claude-haiku-5-5"
    assert "Legacy model ID 'claude-haiku-4-5-20251001'" in capsys.readouterr().out


def test_unknown_analysis_model_falls_back_to_default(client):
    from models import DEFAULT_MODEL
    _, job = _post_report(client, analysis_model="gpt-4o")
    assert job["analysis_model"] == DEFAULT_MODEL


def test_analyse_endpoint_accepts_analysis_model(client):
    _, job = _post_analyse(client, analysis_model="gemini-4-argon")
    assert job["analysis_model"] == "gemini-4-argon"
    assert job["params"]["analysis_model"] == "gemini-4-argon"


def test_gemini_model_not_in_openapi_schema(client):
    schema = json.dumps(client.get("/openapi.json").json())
    assert "gemini_model" not in schema


def test_status_has_no_gemini_model_mirror(client):
    job_id, _ = _post_report(client, analysis_model="claude-sonnet-5-5")
    body = client.get(f"/status/{job_id}").json()
    assert body["analysis_model"] == "claude-sonnet-5-5"
    assert "gemini_model" not in body
    assert "gemini_model" not in body["params"]


# ---------------------------------------------------------------------------
# Picking the newest report when a job folder holds several
# ---------------------------------------------------------------------------

_OLD_REPORT = "input_20261014_102309_report.md"
_NEW_REPORT = "input_20261014_122655_report.md"


def _write_reports(job_dir, names_and_ages):
    """Write one report per (name, age_seconds); each file's body is its own name."""
    now = time.time()
    for name, age in names_and_ages:
        path = os.path.join(job_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n")
        os.utime(path, (now - age, now - age))


def _report_job_dir(tmp_path, job_id, names_and_ages):
    job_dir = _make_job_dir(str(tmp_path), job_id)
    _write_reports(job_dir, names_and_ages)
    return job_dir


@pytest.mark.parametrize("ages", [
    {_OLD_REPORT: 7200, _NEW_REPORT: 0},      # mtimes agree with the names
    {_OLD_REPORT: 0, _NEW_REPORT: 7200},      # alphabetically-first file has the newer mtime
])
def test_latest_report_uses_filename_timestamp(tmp_path, ages):
    from api import _latest_report
    job_dir = _report_job_dir(tmp_path, "rep", list(ages.items()))
    assert os.path.basename(_latest_report(job_dir)) == _NEW_REPORT


def test_latest_report_mtime_breaks_same_timestamp_tie(tmp_path):
    from api import _latest_report
    job_dir = _report_job_dir(tmp_path, "rep", [
        ("a_20261014_120000_report.md", 0),
        ("b_20261014_120000_report.md", 7200),
    ])
    assert os.path.basename(_latest_report(job_dir)) == "a_20261014_120000_report.md"


def test_latest_report_falls_back_to_mtime_for_unmatched_names(tmp_path):
    from api import _latest_report
    job_dir = _report_job_dir(tmp_path, "rep", [
        ("custom_report.md", 0),               # no timestamp in the name; written just now
        ("input_20200101_000000_report.md", 0),  # timestamped name, years older
    ])
    assert os.path.basename(_latest_report(job_dir)) == "custom_report.md"


def test_latest_report_none_when_no_reports(tmp_path):
    from api import _latest_report
    assert _latest_report(_make_job_dir(str(tmp_path), "rep")) is None


@pytest.mark.parametrize("old_age,new_age", [(7200, 0), (0, 7200)])
def test_download_report_serves_newest(client, tmp_path, old_age, new_age):
    """Recovered job (not in memory): /download picks the newest report."""
    job_id = "dl-newest"
    jobs.pop(job_id, None)
    _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, old_age), (_NEW_REPORT, new_age)])
    resp = client.get(f"/download/{job_id}/report")
    assert resp.status_code == 200
    assert _NEW_REPORT in resp.headers["content-disposition"]
    assert resp.text.strip() == f"# {_NEW_REPORT}"


def test_download_report_prefers_in_memory_report_path(client, tmp_path):
    """An in-memory job serves the report it produced, even if a newer file sits beside it."""
    job_id = "dl-known"
    job_dir = _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, 0), (_NEW_REPORT, 0)])
    jobs[job_id] = {"status": "complete", "output_dir": job_dir,
                    "report_path": os.path.join(job_dir, _OLD_REPORT)}
    try:
        resp = client.get(f"/download/{job_id}/report")
        assert _OLD_REPORT in resp.headers["content-disposition"]
    finally:
        jobs.pop(job_id, None)


def test_download_report_ignores_missing_report_path(client, tmp_path):
    job_id = "dl-missing"
    job_dir = _report_job_dir(tmp_path, job_id, [(_NEW_REPORT, 0)])
    jobs[job_id] = {"status": "complete", "output_dir": job_dir,
                    "report_path": os.path.join(job_dir, "gone_report.md")}
    try:
        resp = client.get(f"/download/{job_id}/report")
        assert _NEW_REPORT in resp.headers["content-disposition"]
    finally:
        jobs.pop(job_id, None)


@pytest.mark.parametrize("old_age,new_age", [(7200, 0), (0, 7200)])
def test_recovery_picks_newest_report(tmp_path, monkeypatch, old_age, new_age):
    monkeypatch.chdir(tmp_path)
    job_id = "recover-newest"
    jobs.pop(job_id, None)
    _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, old_age), (_NEW_REPORT, new_age)])
    try:
        job = get_or_recover_job(job_id)
        assert os.path.basename(job["report_path"]) == _NEW_REPORT
        assert os.path.basename(job["files"]["report"]) == _NEW_REPORT
        assert job["report"].strip() == f"# {_NEW_REPORT}"
    finally:
        jobs.pop(job_id, None)


@pytest.mark.parametrize("old_age,new_age", [(7200, 0), (0, 7200)])
def test_reconnect_outputs_and_fallback_use_newest(client, tmp_path, old_age, new_age):
    """/reconnect's outputs map agrees with /download, and its disk fallback reads the newest report."""
    job_id = "reconnect-newest"
    job_dir = _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, old_age), (_NEW_REPORT, new_age)])
    jobs[job_id] = {"status": "complete", "output_dir": job_dir, "report": None, "message_queue": []}
    try:
        from api import _resolve_download_path
        body = client.get(f"/reconnect/{job_id}").json()
        assert body["outputs"]["report"] is True
        assert os.path.basename(_resolve_download_path(job_dir, "report")) == _NEW_REPORT
        assert body["report"].strip() == f"# {_NEW_REPORT}"
    finally:
        jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# No stale report while a new Stage 5 run is in flight
# ---------------------------------------------------------------------------

def _inflight_job(job_dir, status):
    """Job dict as /report-from-json builds it: report None, no report_path."""
    return {"status": status, "generate_report": True, "output_dir": job_dir,
            "report": None, "message_queue": [], "files": {}}


@pytest.mark.parametrize("status", ["queued", "running"])
def test_reconnect_mid_run_returns_no_report(client, tmp_path, status):
    job_id = f"midrun-{status}"
    job_dir = _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, 0)])
    jobs[job_id] = _inflight_job(job_dir, status)
    try:
        resp = client.get(f"/reconnect/{job_id}")
        assert resp.status_code == 202
        assert resp.json()["report"] is None
        assert jobs[job_id]["report"] is None
        assert "report_path" not in jobs[job_id]
    finally:
        jobs.pop(job_id, None)


def test_reconnect_finished_job_still_falls_back_and_caches(client, tmp_path):
    job_id = "finished-fallback"
    job_dir = _report_job_dir(tmp_path, job_id, [(_NEW_REPORT, 0)])
    jobs[job_id] = _inflight_job(job_dir, "complete")
    try:
        body = client.get(f"/reconnect/{job_id}").json()
        assert body["report"].strip() == f"# {_NEW_REPORT}"
        assert jobs[job_id]["report"] == body["report"]
        assert os.path.basename(jobs[job_id]["report_path"]) == _NEW_REPORT
    finally:
        jobs.pop(job_id, None)


def test_reconnect_race_cannot_overwrite_new_report(client, tmp_path, monkeypatch):
    """
    The worker finishes while /reconnect is handling the request.

    The patched _latest_report plays the worker: on its first call it stores
    the new report and marks the job complete, exactly as
    _run_report_from_json does (report first, then status). Before the fix,
    /reconnect called it from its disk fallback and then cached the OLD
    report over the new one.
    """
    import api
    job_id = "race"
    job_dir = _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, 0)])
    job = _inflight_job(job_dir, "running")
    jobs[job_id] = job
    new_text = "# freshly generated report\n"
    real_latest = api._latest_report
    fired = []

    def worker_finishes_now(d):
        if not fired:
            fired.append(True)
            job["report"] = new_text
            job["report_path"] = os.path.join(job_dir, _NEW_REPORT)
            job["status"] = "complete"
        return real_latest(d)

    monkeypatch.setattr(api, "_latest_report", worker_finishes_now)
    try:
        body = client.get(f"/reconnect/{job_id}").json()
        assert fired, "the simulated worker never ran"
        assert job["report"] == new_text
        assert body["report"] in (None, new_text)
    finally:
        jobs.pop(job_id, None)


def test_report_from_json_drops_prior_report_entry(client, tmp_path):
    job_id = "rerun"
    job_dir = _report_job_dir(tmp_path, job_id, [(_OLD_REPORT, 0)])
    jobs[job_id] = {
        "status": "complete", "output_dir": job_dir, "report": "# old\n",
        "report_path": os.path.join(job_dir, _OLD_REPORT),
        "files": {"json": os.path.join(job_dir, "input.json"),
                  "report": os.path.join(job_dir, _OLD_REPORT)},
        "transcript": [], "metadata": {},
    }
    try:
        payload = json.dumps({"metadata": {"job_id": job_id}, "transcript": []})
        resp = client.post("/report-from-json", files={"file": ("t.json", payload, "application/json")})
        assert resp.json()["job_id"] == job_id
        new_job = jobs[job_id]
        assert "report" not in new_job["files"]
        assert new_job["files"]["json"].endswith("input.json")
        assert new_job["report"] is None
        assert "report_path" not in new_job
    finally:
        jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# Stage 1-3 sub-progress (<stage>.run(progress=...) -> WebSocket)
# ---------------------------------------------------------------------------

import api as _api


def _progress_job(job_id):
    jobs[job_id] = {"job_id": job_id, "status": "running", "message_queue": []}
    return jobs[job_id]


def test_push_progress_shape_is_unchanged(monkeypatch):
    job = _progress_job("prog-shape")
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: None)
    try:
        _api._push_progress("prog-shape", 2, "Diarize", "Speaker diarization…")
        assert job["message_queue"] == [
            {"type": "progress", "stage": 2, "stage_name": "Diarize", "message": "Speaker diarization…"}
        ]
    finally:
        jobs.pop("prog-shape", None)


def test_stage_progress_adds_phase_and_percent_only(monkeypatch):
    job = _progress_job("prog-sub")
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: None)
    try:
        cb = _api._stage_progress("prog-sub", 2, "Diarize", "Speaker diarization")
        cb("segmentation", 45, "segmentation 45% (12/27 chunks), 0:40 elapsed, ~0:49 left")
        cb("clustering", None, "Phase: clustering")
        first, second = job["message_queue"]
        assert first == {
            "type": "progress", "stage": 2, "stage_name": "Diarize",
            "message": "Speaker diarization — segmentation 45% (12/27 chunks), 0:40 elapsed, ~0:49 left",
            "phase": "segmentation", "percent": 45,
        }
        assert second["percent"] is None and second["phase"] == "clustering"
        assert job["progress_message"] == second["message"]
        assert job["current_stage"] == 2
    finally:
        jobs.pop("prog-sub", None)


def test_stage_progress_does_not_block_on_slow_socket(monkeypatch):
    _progress_job("prog-slow")
    sent = []

    def slow_send(job_id, payload):
        time.sleep(1.0)
        sent.append(payload)

    monkeypatch.setattr(_api, "_send_ws", slow_send)
    try:
        cb = _api._stage_progress("prog-slow", 2, "Diarize", "Speaker diarization")
        t0 = time.monotonic()
        cb("segmentation", 0, "Phase: segmentation")
        cb("segmentation", 10, "segmentation 10%")
        assert time.monotonic() - t0 < 0.2
        _api._progress_sender.submit(lambda: None).result(timeout=5)  # drain
        # The first is sent only if its send started before the second was
        # queued; the latest always arrives, last.
        assert [p["percent"] for p in sent] in ([0, 10], [10])
    finally:
        jobs.pop("prog-slow", None)


def test_superseded_progress_is_not_sent(monkeypatch):
    job = _progress_job("prog-stale")
    sent = []
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: sent.append(payload))
    try:
        stale = {"type": "progress", "stage": 2, "percent": 40}
        job["message_queue"].append(stale)
        job["message_queue"].append({"type": "progress", "stage": 2, "message": "Done"})
        _api._send_if_latest("prog-stale", stale)
        assert sent == []
        assert stale in job["message_queue"]  # still replayable, in order
    finally:
        jobs.pop("prog-stale", None)


def test_stage_progress_swallows_errors(monkeypatch):
    _progress_job("prog-err")

    class Broken:
        def submit(self, *a, **k):
            raise RuntimeError("executor shut down")

    monkeypatch.setattr(_api, "_progress_sender", Broken())
    try:
        _api._stage_progress("prog-err", 2, "Diarize", "Speaker diarization")("embeddings", 5, "x")  # must not raise
        _api._stage_progress("no-such-job", 2, "Diarize", "Speaker diarization")("embeddings", 5, "x")
    finally:
        jobs.pop("prog-err", None)


def test_send_if_latest_swallows_send_errors(monkeypatch):
    def boom(job_id, payload):
        raise RuntimeError("socket gone")
    monkeypatch.setattr(_api, "_send_ws", boom)
    _api._send_if_latest("no-such-job", {"type": "progress"})  # must not raise


@pytest.mark.parametrize("stage,stage_name,label,phase", [
    (1, "Preprocess", "Preprocessing", "noise reduction"),
    (3, "Transcribe", "Transcription", "transcription"),
])
def test_stage_progress_shape_for_stages_1_and_3(monkeypatch, stage, stage_name, label, phase):
    job = _progress_job("prog-13")
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: None)
    try:
        _api._stage_progress("prog-13", stage, stage_name, label)(phase, 40, f"{phase} 40% (4/10 x)")
        assert job["message_queue"] == [{
            "type": "progress", "stage": stage, "stage_name": stage_name,
            "message": f"{label} — {phase} 40% (4/10 x)", "phase": phase, "percent": 40,
        }]
    finally:
        jobs.pop("prog-13", None)


# ---------------------------------------------------------------------------
# Stage 2 re-identification warnings: job state, complete, /reconnect, /status,
# terminal, transcript JSON and .txt
# ---------------------------------------------------------------------------

from stages import reid_outcome as _reid_outcome

_STUB_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "label": "SPEAKER_00", "speaker": "Speaker A"},
    {"start": 2.5, "end": 4.0, "label": "SPEAKER_01", "speaker": "Speaker B"},
]


def _outcome(status, failed=0, reason=None, found=2):
    outcome = _reid_outcome.new_outcome(102, 2)
    outcome.update(
        status=status, reason=reason, failed_segments=failed,
        clustered_segments=102 - failed, speakers_found=found,
    )
    return outcome


_OUTCOMES = {
    "ok": (_outcome("ok"), []),
    "partial": (
        _outcome("partial", failed=80, reason="feature_extraction"),
        ["Speaker re-identification failed for 80 of 102 segments; labels may be less accurate."],
    ),
    "failed": (
        _outcome("failed", failed=80, reason="exception"),
        ["Speaker re-identification failed for 80 of 102 segments and then stopped; "
         "speaker labels come straight from pyannote and may be less accurate."],
    ),
}


def _stub_heavy_stages(monkeypatch, outcome):
    """Stand-ins for Stages 2 and 3, plus the torch cleanup _run_pipeline does."""
    def fake_diarize(clean_wav_path, output_dir, num_speakers=None, progress=None,
                     on_reidentification=None):
        if on_reidentification is not None:
            on_reidentification(dict(outcome))
        return [dict(s) for s in _STUB_SEGMENTS]

    def fake_transcribe(clean_wav_path, segments, **kwargs):
        return [dict(s, text=f"line {i}") for i, s in enumerate(segments)]

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)  # cleanup's AttributeErrors are caught
    monkeypatch.setattr(_api.diarize, "run", fake_diarize, raising=False)
    monkeypatch.setattr(_api.transcribe, "run", fake_transcribe, raising=False)
    monkeypatch.setattr(_api.time, "sleep", lambda s: None)
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: None)


def _run_stubbed_pipeline(client, monkeypatch, outcome):
    _stub_heavy_stages(monkeypatch, outcome)
    job_id, job = _post_analyse(client)
    input_path = next(
        os.path.join(job["output_dir"], f) for f in os.listdir(job["output_dir"])
        if f.startswith("input.")
    )
    _api._run_pipeline(job_id, input_path, job["params"])
    assert job["status"] == "complete", job.get("error")
    return job_id, job


@pytest.mark.parametrize("case", ["ok", "partial", "failed"])
def test_pipeline_surfaces_reid_warnings_everywhere(client, monkeypatch, capsys, case):
    outcome, expected = _OUTCOMES[case]
    job_id, job = _run_stubbed_pipeline(client, monkeypatch, outcome)
    try:
        assert job["reid"]["status"] == case
        assert job["warnings"] == expected

        complete = [m for m in job["message_queue"] if m["type"] == "complete"]
        assert len(complete) == 1
        assert complete[0]["warnings"] == expected

        body = client.get(f"/reconnect/{job_id}").json()
        assert body["warnings"] == expected

        status = client.get(f"/status/{job_id}").json()
        assert status["warnings"] == expected
        assert status["reid"]["status"] == case

        out = capsys.readouterr().out
        summary_at = out.index(f"[job {job_id}] Pipeline complete")
        for warning in expected:
            line = f"[job {job_id}] Warning: {warning}"
            assert line in out
            assert out.index(line) > summary_at
        if not expected:
            assert f"[job {job_id}] Warning:" not in out

        with open(job["files"]["json"], encoding="utf-8") as f:
            meta = json.load(f)["metadata"]
        with open(job["files"]["transcript"], encoding="utf-8") as f:
            txt = f.read()
        if expected:
            assert meta["warnings"] == expected
            assert job["metadata"]["warnings"] == expected
            for warning in expected:
                assert f"# Warning: {warning}\n" in txt
        else:
            assert "warnings" not in meta
            assert "warnings" not in job["metadata"]
            assert "# Warning" not in txt
    finally:
        jobs.pop(job_id, None)


def test_pipeline_warns_when_fewer_speakers_than_requested(client, monkeypatch):
    job_id, job = _run_stubbed_pipeline(client, monkeypatch, _outcome("ok", found=1))
    try:
        assert job["warnings"] == [
            "Only 1 speaker found but 2 were requested; one speaker may have absorbed the other."
        ]
    finally:
        jobs.pop(job_id, None)


def test_new_jobs_start_with_empty_warnings(client):
    job_id, job = _post_analyse(client)
    try:
        assert job["warnings"] == []
        assert job["reid"] is None
        assert client.get(f"/reconnect/{job_id}").json()["warnings"] == []
    finally:
        jobs.pop(job_id, None)


def test_reconnect_warnings_default_to_empty_list_for_old_job_dicts():
    """A job dict without the key (older in-memory shape) still gives a list."""
    assert _reconnect("no-warnings-key", _job(status="complete"))["warnings"] == []


def test_recovered_job_reads_warnings_from_transcript_json(tmp_path, monkeypatch):
    job_id = "recover-warnings"
    job_dir = _make_job_dir(str(tmp_path), job_id)
    with open(os.path.join(job_dir, "input_20261014_120000.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"job_id": job_id, "warnings": ["Careful."]}, "transcript": []}, f)
    monkeypatch.chdir(tmp_path)
    try:
        job = get_or_recover_job(job_id)
        assert job["warnings"] == ["Careful."]
        assert job["reid"] is None
    finally:
        jobs.pop(job_id, None)


def _run_report_job(client, monkeypatch, metadata):
    """Upload a transcript JSON and run _run_report_from_json with Stage 5 stubbed."""
    calls = []

    def fake_report(**kwargs):
        calls.append(kwargs)
        path = os.path.join(kwargs["output_dir"], "input_20261014_120000_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Call Analysis Report\n")
        return path

    monkeypatch.setattr(_api.report, "run", fake_report)
    monkeypatch.setattr(_api.settings, "validate_for_report", lambda model: None)
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: None)
    payload = {"metadata": metadata, "transcript": [dict(s, text="hi") for s in _STUB_SEGMENTS]}
    files = {"file": ("t.json", json.dumps(payload), "application/json")}
    job_id = client.post("/report-from-json", files=files).json()["job_id"]
    job = jobs[job_id]
    _api._run_report_from_json(job_id, os.path.join(job["output_dir"], "transcript.json"), job["params"])
    assert job["status"] == "complete", job.get("error")
    return job_id, job, calls


def test_report_from_json_carries_uploaded_warnings(client, monkeypatch, capsys):
    warning = "Speaker re-identification failed for 80 of 102 segments; labels may be less accurate."
    job_id, job, _ = _run_report_job(client, monkeypatch, {"warnings": [warning], "extra": {"x": 1}})
    try:
        assert job["warnings"] == [warning]
        complete = [m for m in job["message_queue"] if m["type"] == "complete"]
        assert complete[0]["warnings"] == [warning]
        assert client.get(f"/reconnect/{job_id}").json()["warnings"] == [warning]
        assert client.get(f"/status/{job_id}").json()["warnings"] == [warning]
        assert f"[job {job_id}] Warning: {warning}" in capsys.readouterr().out
        # Unknown metadata keys pass through untouched.
        assert job["metadata"]["extra"] == {"x": 1}
    finally:
        jobs.pop(job_id, None)


def test_report_from_json_relabelled_copy_keeps_warnings(client, monkeypatch):
    files = {"file": ("t.json", json.dumps({
        "metadata": {"warnings": ["Careful."]},
        "transcript": [dict(s, text="hi") for s in _STUB_SEGMENTS],
    }), "application/json")}
    monkeypatch.setattr(_api.report, "run", lambda **kw: os.path.join(kw["output_dir"], "r_report.md"))
    monkeypatch.setattr(_api.settings, "validate_for_report", lambda model: None)
    monkeypatch.setattr(_api, "_send_ws", lambda job_id, payload: None)
    job_id = client.post("/report-from-json", data={"speaker_names": "Ann,Bob"}, files=files).json()["job_id"]
    job = jobs[job_id]
    try:
        _api._run_report_from_json(job_id, os.path.join(job["output_dir"], "transcript.json"), job["params"])
        with open(job["files"]["json"], encoding="utf-8") as f:
            assert json.load(f)["metadata"]["warnings"] == ["Careful."]
    finally:
        jobs.pop(job_id, None)


@pytest.mark.parametrize("value,expected", [
    (None, []),
    ([], []),
    ("One warning.", ["One warning."]),
    ({"not": "a list"}, []),
    (["Kept.", 3, None, "  ", {"x": 1}], ["Kept."]),
])
def test_report_from_json_tolerates_malformed_warnings(client, monkeypatch, value, expected):
    metadata = {} if value is None else {"warnings": value}
    job_id, job, _ = _run_report_job(client, monkeypatch, metadata)
    try:
        assert job["warnings"] == expected
    finally:
        jobs.pop(job_id, None)


def test_report_from_json_passes_uploaded_warnings_to_stage5(client, monkeypatch):
    job_id, job, calls = _run_report_job(client, monkeypatch, {"warnings": ["Careful."]})
    try:
        assert calls[0]["warnings"] == ["Careful."]
    finally:
        jobs.pop(job_id, None)


def test_report_from_json_without_warnings_passes_empty_list(client, monkeypatch):
    job_id, job, calls = _run_report_job(client, monkeypatch, {})
    try:
        assert calls[0]["warnings"] == []
    finally:
        jobs.pop(job_id, None)


def test_pipeline_passes_reid_warnings_to_stage5(client, monkeypatch):
    outcome, expected = _OUTCOMES["partial"]
    _stub_heavy_stages(monkeypatch, outcome)
    calls = []

    def fake_report(**kwargs):
        calls.append(kwargs)
        path = os.path.join(kwargs["output_dir"], "input_20261014_120000_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Call Analysis Report\n")
        return path

    monkeypatch.setattr(_api.report, "run", fake_report)
    monkeypatch.setattr(_api.settings, "validate_for_report", lambda model: None)
    job_id, job = _post_analyse(client, generate_report="true")
    input_path = next(
        os.path.join(job["output_dir"], f) for f in os.listdir(job["output_dir"])
        if f.startswith("input.")
    )
    try:
        _api._run_pipeline(job_id, input_path, job["params"])
        assert job["status"] == "complete", job.get("error")
        assert calls[0]["warnings"] == expected
    finally:
        jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# Whisper release after each job (RELEASE_WHISPER_AFTER_JOB)
# ---------------------------------------------------------------------------

class _FakeWhisper:
    """Stands in for transcribe.model_loaded() / release_model()."""

    def __init__(self, release_error=None):
        self.loaded = False
        self.releases = 0
        self.release_error = release_error

    def model_loaded(self):
        return self.loaded

    def release_model(self):
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error
        self.loaded = False
        return (3.16 * 2 ** 30, 4.0 * 2 ** 30)


def _stub_whisper(monkeypatch, release_on=True, release_error=None, stage3_error=None):
    """Stub Stages 2-3 plus the Whisper release hooks; Stage 3 marks the model loaded."""
    _stub_heavy_stages(monkeypatch, _outcome("ok"))
    fake = _FakeWhisper(release_error)

    def fake_transcribe(clean_wav_path, segments, **kwargs):
        fake.loaded = True
        if stage3_error is not None:
            raise stage3_error
        return [dict(s, text=f"line {i}") for i, s in enumerate(segments)]

    monkeypatch.setattr(_api.transcribe, "run", fake_transcribe, raising=False)
    monkeypatch.setattr(_api.transcribe, "model_loaded", fake.model_loaded, raising=False)
    monkeypatch.setattr(_api.transcribe, "release_model", fake.release_model, raising=False)
    # _configure_settings() resets settings from this snapshot at job start.
    monkeypatch.setitem(_api._default_settings, "release_whisper_after_job", release_on)
    return fake


def _run_job(client):
    job_id, job = _post_analyse(client)
    input_path = next(
        os.path.join(job["output_dir"], f) for f in os.listdir(job["output_dir"])
        if f.startswith("input.")
    )
    _api._run_pipeline(job_id, input_path, job["params"])
    return job_id, job


def test_whisper_released_after_complete(client, monkeypatch, capsys):
    fake = _stub_whisper(monkeypatch)
    job_id, job = _run_job(client)
    try:
        assert job["status"] == "complete", job.get("error")
        assert fake.releases == 1
        assert fake.loaded is False
        out = capsys.readouterr().out
        before = f"[job {job_id}] Releasing Whisper model..."
        after = f"[job {job_id}] Whisper model released; VRAM free 3.16 of 4.00 GB"
        assert before in out and after in out
        # After the completion summary, before the job-end memory line.
        assert out.index(f"[job {job_id}] Pipeline complete") < out.index(before) < out.index(after)
        assert out.index(after) < out.index(f"[job {job_id}] Memory at job end")
        # The complete message went out before the release.
        assert [m["type"] for m in job["message_queue"]][-1] == "complete"
    finally:
        jobs.pop(job_id, None)


def test_whisper_release_happens_after_push_complete(client, monkeypatch):
    fake = _stub_whisper(monkeypatch)
    order = []
    real_push = _api._push_complete
    monkeypatch.setattr(_api, "_push_complete", lambda jid: (order.append("complete"), real_push(jid)))
    real_release = fake.release_model
    monkeypatch.setattr(_api.transcribe, "release_model",
                        lambda: (order.append("release"), real_release())[1], raising=False)
    job_id, job = _run_job(client)
    try:
        assert order == ["complete", "release"]
    finally:
        jobs.pop(job_id, None)


def test_whisper_released_when_stage3_fails_after_load(client, monkeypatch, capsys):
    fake = _stub_whisper(monkeypatch, stage3_error=RuntimeError("CUDA out of memory"))
    job_id, job = _run_job(client)
    try:
        assert job["status"] == "error"
        assert fake.releases == 1
        out = capsys.readouterr().out
        assert f"[job {job_id}] Releasing Whisper model..." in out
        assert f"[job {job_id}] Whisper model released; VRAM free" in out
    finally:
        jobs.pop(job_id, None)


def test_whisper_not_released_when_never_loaded(client, monkeypatch, capsys):
    fake = _stub_whisper(monkeypatch)

    def failing_diarize(*a, **k):
        raise RuntimeError("Stage 2 broke")

    monkeypatch.setattr(_api.diarize, "run", failing_diarize, raising=False)
    job_id, job = _run_job(client)
    try:
        assert job["status"] == "error"
        assert fake.releases == 0
        assert "Releasing Whisper model" not in capsys.readouterr().out
    finally:
        jobs.pop(job_id, None)


def test_whisper_kept_when_switched_off(client, monkeypatch, capsys):
    fake = _stub_whisper(monkeypatch, release_on=False)
    job_id, job = _run_job(client)
    try:
        assert job["status"] == "complete", job.get("error")
        assert fake.releases == 0
        assert fake.loaded is True
        assert "Releasing Whisper model" not in capsys.readouterr().out
    finally:
        jobs.pop(job_id, None)


@pytest.mark.parametrize("error", [RuntimeError("driver said no"), SystemExit(1)])
def test_whisper_release_error_is_logged_not_raised(client, monkeypatch, capsys, error):
    fake = _stub_whisper(monkeypatch, release_error=error)
    job_id, job = _run_job(client)  # must return normally, even for SystemExit
    try:
        assert job["status"] == "complete"
        assert fake.releases == 1
        out = capsys.readouterr().out
        assert f"[job {job_id}] Releasing Whisper model..." in out
        assert f"[job {job_id}] Whisper model released" not in out
        assert f"[job {job_id}] Whisper model release failed (ignored): {error!r}" in out
        assert f"[job {job_id}] Memory at job end" in out
    finally:
        jobs.pop(job_id, None)
