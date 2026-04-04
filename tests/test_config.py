"""Tests for config.py — Settings dataclass and validation."""

import pytest
from config import Settings, VALID_CONTEXTS


def test_default_values():
    """Settings have sensible defaults when env vars not set."""
    s = Settings()
    assert s.context in VALID_CONTEXTS
    assert s.whisper_model == "medium"
    assert s.transcription_mode in ("accurate", "fast")
    assert isinstance(s.speaker_names, list)
    assert isinstance(s.word_timestamps, bool)


def test_context_default_is_friend(monkeypatch):
    """Default context is friend when env var is not set."""
    monkeypatch.delenv("CONVERSATION_CONTEXT", raising=False)
    s = Settings()
    assert s.context == "friend"


def test_num_speakers_default_is_none(monkeypatch):
    """Default num_speakers is None (auto-detect) when env var is not set."""
    monkeypatch.delenv("NUM_SPEAKERS", raising=False)
    s = Settings()
    assert s.num_speakers is None


def test_whisper_model_default_is_medium(monkeypatch):
    """Default whisper model is medium when env var is not set."""
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    s = Settings()
    assert s.whisper_model == "medium"


def test_transcription_mode_default_is_accurate(monkeypatch):
    """Default transcription mode is accurate when env var is not set."""
    monkeypatch.delenv("TRANSCRIPTION_MODE", raising=False)
    s = Settings()
    assert s.transcription_mode == "accurate"


def test_override_context():
    """settings.override(context='work') changes context."""
    s = Settings()
    s.override(context="work")
    assert s.context == "work"


def test_override_num_speakers():
    """settings.override(num_speakers=3) changes num_speakers."""
    s = Settings()
    s.override(num_speakers=3)
    assert s.num_speakers == 3


def test_override_speaker_names():
    """settings.override(speaker_names=['Alice', 'Bob']) sets names."""
    s = Settings()
    s.override(speaker_names=["Alice", "Bob"])
    assert s.speaker_names == ["Alice", "Bob"]


def test_override_invalid_context_raises():
    """override() raises ValueError for an invalid context."""
    s = Settings()
    with pytest.raises(ValueError, match="--context must be one of"):
        s.override(context="invalid_context")


def test_override_invalid_num_speakers_raises():
    """override() raises ValueError for num_speakers < 1."""
    s = Settings()
    with pytest.raises(ValueError, match="--num-speakers must be a positive integer"):
        s.override(num_speakers=0)


def test_validate_for_report_raises_without_api_key(monkeypatch):
    """validate_for_report() raises EnvironmentError if GEMINI_API_KEY not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings()
    s.gemini_api_key = ""
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        s.validate_for_report()


def test_validate_for_report_passes_with_api_key(monkeypatch):
    """validate_for_report() passes when GEMINI_API_KEY is set."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    s = Settings()
    s.validate_for_report()  # should not raise


def test_invalid_context_env_var_defaults_to_friend(monkeypatch):
    """Invalid CONVERSATION_CONTEXT env var falls back to 'friend'."""
    monkeypatch.setenv("CONVERSATION_CONTEXT", "nonsense")
    s = Settings()
    assert s.context == "friend"


def test_num_speakers_parsed_from_env(monkeypatch):
    """NUM_SPEAKERS env var is parsed as int."""
    monkeypatch.setenv("NUM_SPEAKERS", "3")
    s = Settings()
    assert s.num_speakers == 3


def test_speaker_names_parsed_from_env(monkeypatch):
    """SPEAKER_NAMES env var is parsed as a list."""
    monkeypatch.setenv("SPEAKER_NAMES", "Alice,Bob")
    s = Settings()
    assert s.speaker_names == ["Alice", "Bob"]


def test_word_timestamps_false_by_default(monkeypatch):
    """WORD_TIMESTAMPS is False when env var is not set."""
    monkeypatch.delenv("WORD_TIMESTAMPS", raising=False)
    s = Settings()
    assert s.word_timestamps is False


def test_word_timestamps_true_when_set(monkeypatch):
    """WORD_TIMESTAMPS is True when env var is 'true'."""
    monkeypatch.setenv("WORD_TIMESTAMPS", "true")
    s = Settings()
    assert s.word_timestamps is True
