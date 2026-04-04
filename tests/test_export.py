"""Tests for stages/export.py — the most unit-testable stage."""

import json
import os
import time

import pytest

from stages.export import _format_timestamp, run, write_relabelled


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------

def test_format_timestamp_zero():
    """0.0 formats as 00:00:00."""
    assert _format_timestamp(0.0) == "00:00:00"


def test_format_timestamp_seconds_only():
    """Seconds under 60 format as 00:00:SS."""
    assert _format_timestamp(34.5) == "00:00:34"


def test_format_timestamp_minutes():
    """60+ seconds format as 00:MM:SS."""
    assert _format_timestamp(94.0) == "00:01:34"


def test_format_timestamp_hours():
    """3600+ seconds format as HH:MM:SS."""
    assert _format_timestamp(3661.0) == "01:01:01"


def test_format_timestamp_exact_hour():
    """Exactly 3600 seconds formats as 01:00:00."""
    assert _format_timestamp(3600.0) == "01:00:00"


def test_format_timestamp_large():
    """2h40m formats correctly."""
    assert _format_timestamp(9600.0) == "02:40:00"


# ---------------------------------------------------------------------------
# run() — file creation
# ---------------------------------------------------------------------------

def test_export_creates_txt_file(tmp_output_dir, sample_segments):
    """run() creates a .txt file in output dir."""
    txt_path, _ = run(sample_segments, "call.mp3", tmp_output_dir)
    assert os.path.isfile(txt_path)
    assert txt_path.endswith(".txt")


def test_export_creates_json_file(tmp_output_dir, sample_segments):
    """run() creates a .json file in output dir."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir)
    assert os.path.isfile(json_path)
    assert json_path.endswith(".json")


# ---------------------------------------------------------------------------
# run() — JSON structure
# ---------------------------------------------------------------------------

def test_json_has_metadata_key(tmp_output_dir, sample_segments):
    """Exported JSON has a 'metadata' key."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data


def test_json_has_transcript_key(tmp_output_dir, sample_segments):
    """Exported JSON has a 'transcript' key."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "transcript" in data


def test_json_metadata_contains_job_id(tmp_output_dir, sample_segments):
    """Exported JSON metadata contains job_id when provided."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir, job_id="abc-123")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["job_id"] == "abc-123"


def test_json_metadata_contains_context(tmp_output_dir, sample_segments):
    """Exported JSON metadata contains context."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir, context="work")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["context"] == "work"


def test_json_metadata_num_speakers_inferred(tmp_output_dir, sample_segments):
    """num_speakers is inferred from segment data when not provided."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["num_speakers"] == 2


def test_json_segments_have_required_keys(tmp_output_dir, sample_segments):
    """Each transcript segment has start, end, speaker, text, confidence."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    for seg in data["transcript"]:
        assert "start" in seg
        assert "end" in seg
        assert "speaker" in seg
        assert "text" in seg
        assert "confidence" in seg


def test_json_segment_count_matches(tmp_output_dir, sample_segments):
    """Number of exported segments matches input."""
    _, json_path = run(sample_segments, "call.mp3", tmp_output_dir)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["transcript"]) == len(sample_segments)


# ---------------------------------------------------------------------------
# run() — TXT content
# ---------------------------------------------------------------------------

def test_txt_contains_speaker_labels(tmp_output_dir, sample_segments):
    """Exported TXT contains speaker labels."""
    txt_path, _ = run(sample_segments, "call.mp3", tmp_output_dir)
    content = open(txt_path, encoding="utf-8").read()
    assert "Speaker A" in content
    assert "Speaker B" in content


def test_txt_contains_timestamps(tmp_output_dir, sample_segments):
    """Exported TXT contains timestamps in [HH:MM:SS] format."""
    txt_path, _ = run(sample_segments, "call.mp3", tmp_output_dir)
    content = open(txt_path, encoding="utf-8").read()
    assert "[00:00:00]" in content


def test_txt_contains_segment_text(tmp_output_dir, sample_segments):
    """Exported TXT contains the actual transcript text."""
    txt_path, _ = run(sample_segments, "call.mp3", tmp_output_dir)
    content = open(txt_path, encoding="utf-8").read()
    assert "Hello, how are you doing today?" in content


def test_txt_header_contains_source_file(tmp_output_dir, sample_segments):
    """Exported TXT header contains the source filename."""
    txt_path, _ = run(sample_segments, "my_call.mp3", tmp_output_dir)
    content = open(txt_path, encoding="utf-8").read()
    assert "my_call.mp3" in content


# ---------------------------------------------------------------------------
# run() — speaker names
# ---------------------------------------------------------------------------

def test_export_with_speaker_names_in_metadata(tmp_output_dir, sample_segments):
    """speaker_names are written into JSON metadata when provided."""
    _, json_path = run(
        sample_segments, "call.mp3", tmp_output_dir,
        speaker_names=["Alice", "Bob"]
    )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["speaker_names"] == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# run() — unique filenames
# ---------------------------------------------------------------------------

def test_export_unique_filenames(tmp_output_dir, sample_segments):
    """Two consecutive exports produce different filenames (timestamp-based)."""
    txt1, json1 = run(sample_segments, "call.mp3", tmp_output_dir)
    time.sleep(1.1)  # ensure the second-precision timestamp changes
    txt2, json2 = run(sample_segments, "call.mp3", tmp_output_dir)
    assert os.path.basename(txt1) != os.path.basename(txt2)
    assert os.path.basename(json1) != os.path.basename(json2)


# ---------------------------------------------------------------------------
# write_relabelled()
# ---------------------------------------------------------------------------

def test_write_relabelled_creates_named_json(
    tmp_output_dir, sample_json_file, sample_segments, sample_metadata
):
    """write_relabelled() creates a transcript_named.json file."""
    out_path = write_relabelled(
        source_json_path=sample_json_file,
        segments=sample_segments,
        original_metadata=sample_metadata,
        speaker_names=["Alice", "Bob"],
        output_dir=tmp_output_dir,
    )
    assert os.path.isfile(out_path)
    assert out_path.endswith("_named.json")


def test_write_relabelled_json_has_correct_speakers(
    tmp_output_dir, sample_json_file, sample_segments, sample_metadata
):
    """transcript_named.json has speaker_names in metadata."""
    relabelled_segments = [dict(s) for s in sample_segments]
    relabelled_segments[0]["speaker"] = "Alice"
    relabelled_segments[2]["speaker"] = "Alice"
    relabelled_segments[1]["speaker"] = "Bob"

    out_path = write_relabelled(
        source_json_path=sample_json_file,
        segments=relabelled_segments,
        original_metadata=sample_metadata,
        speaker_names=["Alice", "Bob"],
        output_dir=tmp_output_dir,
    )
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["speaker_names"] == ["Alice", "Bob"]
    assert data["transcript"][0]["speaker"] == "Alice"
    assert data["transcript"][1]["speaker"] == "Bob"


def test_write_relabelled_preserves_original_metadata(
    tmp_output_dir, sample_json_file, sample_segments, sample_metadata
):
    """write_relabelled() preserves fields from original_metadata."""
    out_path = write_relabelled(
        source_json_path=sample_json_file,
        segments=sample_segments,
        original_metadata=sample_metadata,
        speaker_names=["Alice", "Bob"],
        output_dir=tmp_output_dir,
    )
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["job_id"] == "test-job-123"
    assert data["metadata"]["context"] == "friend"


def test_write_relabelled_adds_relabelled_at(
    tmp_output_dir, sample_json_file, sample_segments, sample_metadata
):
    """write_relabelled() adds a relabelled_at timestamp to metadata."""
    out_path = write_relabelled(
        source_json_path=sample_json_file,
        segments=sample_segments,
        original_metadata=sample_metadata,
        speaker_names=["Alice", "Bob"],
        output_dir=tmp_output_dir,
    )
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "relabelled_at" in data["metadata"]
