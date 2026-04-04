import pytest
import json


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for each test."""
    output = tmp_path / "output"
    output.mkdir()
    return str(output)


@pytest.fixture
def sample_segments():
    """Realistic transcript segments for testing."""
    return [
        {
            "start": 0.0,
            "end": 4.5,
            "speaker": "Speaker A",
            "label": "SPEAKER_00",
            "text": "Hello, how are you doing today?",
            "confidence": 0.85,
        },
        {
            "start": 5.0,
            "end": 9.2,
            "speaker": "Speaker B",
            "label": "SPEAKER_01",
            "text": "I'm doing well, thanks for asking.",
            "confidence": 0.91,
        },
        {
            "start": 10.0,
            "end": 15.8,
            "speaker": "Speaker A",
            "label": "SPEAKER_00",
            "text": "Great to hear. Shall we get started?",
            "confidence": 0.78,
        },
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata dict for testing."""
    return {
        "job_id": "test-job-123",
        "source_file": "test_call.m4a",
        "context": "friend",
        "num_speakers": 2,
        "processed_at": "2026-04-01T00:00:00",
    }


@pytest.fixture
def sample_json_file(tmp_path, sample_segments, sample_metadata):
    """Write a sample transcript JSON to a temp file."""
    data = {
        "metadata": sample_metadata,
        "transcript": sample_segments,
    }
    json_file = tmp_path / "transcript.json"
    json_file.write_text(json.dumps(data), encoding="utf-8")
    return str(json_file)
