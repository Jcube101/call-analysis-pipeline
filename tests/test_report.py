"""Tests for stages/report.py — prompt loading, chunking, fallback, and file output."""

import os
from unittest.mock import MagicMock, patch, call

import pytest

from stages.report import (
    FALLBACK_MODEL,
    _DEFAULT_PROMPTS,
    _call_gemini,
    _call_gemini_model,
    _load_prompt,
    _split_transcript,
    _CHUNK_CHARS,
)


# ---------------------------------------------------------------------------
# _load_prompt
# ---------------------------------------------------------------------------

def test_prompt_loads_from_file(tmp_path):
    """Prompt is loaded from prompts/{context}.md when the file exists."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "friend.md").write_text("Custom friend prompt", encoding="utf-8")

    result = _load_prompt("friend", str(prompts_dir))
    assert result == "Custom friend prompt"


def test_prompt_falls_back_to_default_if_missing(tmp_path):
    """Built-in default prompt used when prompts/{context}.md is missing."""
    empty_dir = tmp_path / "prompts"
    empty_dir.mkdir()

    result = _load_prompt("friend", str(empty_dir))
    assert result == _DEFAULT_PROMPTS["friend"]


def test_prompt_falls_back_for_unknown_context(tmp_path):
    """Unknown context falls back to the 'friend' default prompt."""
    empty_dir = tmp_path / "prompts"
    empty_dir.mkdir()

    result = _load_prompt("unknown_context", str(empty_dir))
    assert result == _DEFAULT_PROMPTS["friend"]


def test_prompt_whitespace_stripped(tmp_path):
    """Prompt loaded from file has leading/trailing whitespace stripped."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "work.md").write_text("  Work prompt  \n\n", encoding="utf-8")

    result = _load_prompt("work", str(prompts_dir))
    assert result == "Work prompt"


def test_all_default_contexts_have_prompts():
    """Every valid context has a built-in default prompt."""
    for context in ("friend", "work", "interview", "date"):
        assert context in _DEFAULT_PROMPTS
        assert len(_DEFAULT_PROMPTS[context]) > 20


# ---------------------------------------------------------------------------
# _split_transcript
# ---------------------------------------------------------------------------

def test_transcript_chunking_short():
    """Short transcripts (< 500k chars) are returned as a single chunk."""
    text = "line\n" * 100
    chunks = _split_transcript(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_transcript_chunking_long():
    """Long transcripts (> _CHUNK_CHARS) are split into at least 2 chunks."""
    line = "a" * 1000 + "\n"
    # Build a transcript slightly over the threshold so we get at least 2 chunks
    text = line * (_CHUNK_CHARS // len(line) + 2)
    chunks = _split_transcript(text)
    assert len(chunks) >= 2


def test_transcript_chunking_preserves_all_content():
    """Content is not lost when chunking (overlap may duplicate some lines)."""
    line = "word " * 200 + "\n"
    text = line * (_CHUNK_CHARS // len(line) + 2)
    chunks = _split_transcript(text)
    # Every line in the original must appear in at least one chunk
    total_combined = "".join(chunks)
    for original_line in text.splitlines():
        assert original_line in total_combined


def test_transcript_chunking_exact_size():
    """A transcript exactly at _CHUNK_CHARS is returned as one chunk."""
    text = "x" * _CHUNK_CHARS
    chunks = _split_transcript(text)
    # Exactly at limit is not split
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _call_gemini — 503 fallback
# ---------------------------------------------------------------------------

def test_model_fallback_on_503():
    """Falls back to FALLBACK_MODEL when the requested model returns a 503."""
    mock_client = MagicMock()
    fallback_response = MagicMock()
    fallback_response.text = "Fallback report content"

    # First call raises 503, second call (fallback) succeeds
    with patch("stages.report._call_gemini_model") as mock_call:
        mock_call.side_effect = [
            Exception("503 UNAVAILABLE: service overloaded"),
            fallback_response.text,
        ]
        text, actual_model = _call_gemini(
            mock_client, "system", "user", "gemini-3.1-pro-preview"
        )

    assert actual_model == FALLBACK_MODEL
    assert text == "Fallback report content"
    assert mock_call.call_count == 2
    # Second call used the fallback model
    assert mock_call.call_args_list[1][0][3] == FALLBACK_MODEL


def test_model_fallback_not_triggered_on_other_errors():
    """Non-503 errors are not caught by fallback logic and propagate."""
    mock_client = MagicMock()

    with patch("stages.report._call_gemini_model") as mock_call:
        mock_call.side_effect = Exception("400 BAD REQUEST: invalid api key")
        with pytest.raises(Exception, match="400 BAD REQUEST"):
            _call_gemini(mock_client, "system", "user", "gemini-3.1-pro-preview")

    assert mock_call.call_count == 1


def test_model_fallback_not_triggered_when_already_flash():
    """503 on Flash itself is not retried (would infinite-loop)."""
    mock_client = MagicMock()

    with patch("stages.report._call_gemini_model") as mock_call:
        mock_call.side_effect = Exception("503 UNAVAILABLE")
        with pytest.raises(Exception, match="503"):
            _call_gemini(mock_client, "system", "user", FALLBACK_MODEL)

    assert mock_call.call_count == 1


def test_successful_call_returns_text_and_model():
    """Successful _call_gemini returns (response_text, requested_model)."""
    mock_client = MagicMock()

    with patch("stages.report._call_gemini_model", return_value="Great report"):
        text, actual_model = _call_gemini(
            mock_client, "system", "user", "gemini-3-flash-preview"
        )

    assert text == "Great report"
    assert actual_model == "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# run() — integration with mocked Gemini
# ---------------------------------------------------------------------------

def test_report_saved_to_correct_path(tmp_path, sample_segments):
    """Report file is saved to output_dir with _report.md suffix."""
    with patch("stages.report._call_gemini", return_value=("# Report content", "gemini-3-flash-preview")):
        with patch("stages.report._genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from stages import report as report_module
            report_module._genai = mock_genai

            out_path = report_module.run(
                segments=sample_segments,
                source_file="test_call.mp3",
                output_dir=str(tmp_path),
                context="friend",
                num_speakers=2,
                audio_duration=15.8,
                speaker_counts={"Speaker A": 2, "Speaker B": 1},
                api_key="fake-key",
                prompts_dir=str(tmp_path / "prompts"),
            )

    assert os.path.isfile(out_path)
    assert out_path.endswith("_report.md")
    assert "test_call" in os.path.basename(out_path)


def test_report_file_contains_report_body(tmp_path, sample_segments):
    """Report file contains the text returned by _call_gemini."""
    with patch("stages.report._call_gemini", return_value=("Unique report body text", "gemini-3-flash-preview")):
        with patch("stages.report._genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            from stages import report as report_module
            report_module._genai = mock_genai

            out_path = report_module.run(
                segments=sample_segments,
                source_file="call.mp3",
                output_dir=str(tmp_path),
                context="friend",
                num_speakers=2,
                audio_duration=None,
                speaker_counts={"Speaker A": 2, "Speaker B": 1},
                api_key="fake-key",
                prompts_dir=str(tmp_path / "prompts"),
            )

    content = open(out_path, encoding="utf-8").read()
    assert "Unique report body text" in content


def test_run_requires_gemini_api_key(tmp_path, sample_segments):
    """run() raises EnvironmentError via settings.validate_for_report() when key is missing."""
    # Directly patching the genai import path so we reach validate logic;
    # but validate_for_report is called by the caller, not inside report.run().
    # Verify that an empty api_key causes the Gemini client init to fail or
    # that an explicit pre-check raises appropriately.
    from config import Settings
    s = Settings()
    s.gemini_api_key = ""
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        s.validate_for_report()
