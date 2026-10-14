"""Tests for stages/report.py — prompt loading, chunking, fallback, and file output."""

import os
from unittest.mock import MagicMock, patch, call

import pytest

from stages import report
from stages.report import (
    GEMINI_FALLBACK_MODEL,
    CLAUDE_FALLBACK_MODEL,
    _DEFAULT_PROMPTS,
    _call_gemini,
    _call_gemini_model,
    _call_claude,
    _call_claude_model,
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
    """Falls back to GEMINI_FALLBACK_MODEL when the requested model returns a 503."""
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
            mock_client, "system", "user", "gemini-4-argon"
        )

    assert actual_model == GEMINI_FALLBACK_MODEL
    assert text == "Fallback report content"
    assert mock_call.call_count == 2
    # Second call used the fallback model
    assert mock_call.call_args_list[1][0][3] == GEMINI_FALLBACK_MODEL


def test_model_fallback_not_triggered_on_other_errors():
    """Non-503 errors are not caught by fallback logic and propagate."""
    mock_client = MagicMock()

    with patch("stages.report._call_gemini_model") as mock_call:
        mock_call.side_effect = Exception("400 BAD REQUEST: invalid api key")
        with pytest.raises(Exception, match="400 BAD REQUEST"):
            _call_gemini(mock_client, "system", "user", "gemini-4-argon")

    assert mock_call.call_count == 1


def test_model_fallback_not_triggered_when_already_flash():
    """503 on Flash itself is not retried (would infinite-loop)."""
    mock_client = MagicMock()

    with patch("stages.report._call_gemini_model") as mock_call:
        mock_call.side_effect = Exception("503 UNAVAILABLE")
        with pytest.raises(Exception, match="503"):
            _call_gemini(mock_client, "system", "user", GEMINI_FALLBACK_MODEL)

    assert mock_call.call_count == 1


def test_successful_call_returns_text_and_model():
    """Successful _call_gemini returns (response_text, requested_model)."""
    mock_client = MagicMock()

    with patch("stages.report._call_gemini_model", return_value="Great report"):
        text, actual_model = _call_gemini(
            mock_client, "system", "user", "gemini-3.8-flash"
        )

    assert text == "Great report"
    assert actual_model == "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# Response shapes and truncation guards
# ---------------------------------------------------------------------------

def _claude_message(blocks, stop_reason="end_turn"):
    message = MagicMock()
    message.content = blocks
    message.stop_reason = stop_reason
    return message


def _block(kind, text=""):
    b = MagicMock()
    b.type = kind
    b.text = text
    return b


def test_claude_skips_thinking_blocks():
    """Claude 5.5 can return a thinking block before the text; only text is kept."""
    msg = _claude_message([_block("thinking"), _block("text", "Report "), _block("text", "body")])
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = msg
        text = _call_claude_model("key", "system", "user", "claude-haiku-5-5")
    assert text == "Report body"


def test_claude_max_tokens_stop_raises():
    """A report cut off at max_tokens raises instead of being saved truncated."""
    msg = _claude_message([_block("text", "Half a rep")], stop_reason="max_tokens")
    with patch("anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = msg
        with pytest.raises(RuntimeError, match="truncated"):
            _call_claude_model("key", "system", "user", "claude-sonnet-5-5")


def test_gemini_max_tokens_finish_raises():
    """A Gemini response that stopped on its output limit raises."""
    response = MagicMock()
    response.text = "Half a rep"
    response.candidates = [MagicMock(finish_reason="FinishReason.MAX_TOKENS")]
    client = MagicMock()
    client.models.generate_content.return_value = response
    with pytest.raises(RuntimeError, match="truncated"):
        _call_gemini_model(client, "system", "user", "gemini-4-argon")


def test_gemini_normal_finish_returns_text():
    response = MagicMock()
    response.text = "Full report"
    response.candidates = [MagicMock(finish_reason="FinishReason.STOP")]
    client = MagicMock()
    client.models.generate_content.return_value = response
    assert _call_gemini_model(client, "system", "user", "gemini-3.8-flash") == "Full report"


# ---------------------------------------------------------------------------
# run() — integration with mocked Gemini
# ---------------------------------------------------------------------------

def test_report_saved_to_correct_path(tmp_path, sample_segments):
    """Report file is saved to output_dir with _report.md suffix."""
    with patch("stages.report._call_gemini", return_value=("# Report content", "gemini-3.8-flash")):
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
                prompts_dir=str(tmp_path / "prompts"),
                analysis_model="gemini-3.8-flash",
                gemini_api_key="fake-key",
            )

    assert os.path.isfile(out_path)
    assert out_path.endswith("_report.md")
    assert "test_call" in os.path.basename(out_path)


def test_report_saved_to_correct_path_claude(tmp_path, sample_segments):
    """Report file is saved correctly when using a Claude model."""
    with patch("stages.report._call_claude", return_value=("# Claude report", "claude-haiku-5-5")):
        out_path = report.run(
            segments=sample_segments,
            source_file="test_call.mp3",
            output_dir=str(tmp_path),
            context="friend",
            num_speakers=2,
            audio_duration=15.8,
            speaker_counts={"Speaker A": 2, "Speaker B": 1},
            prompts_dir=str(tmp_path / "prompts"),
            analysis_model="claude-haiku-5-5",
            anthropic_api_key="fake-key",
        )

    assert os.path.isfile(out_path)
    assert out_path.endswith("_report.md")


def test_report_file_contains_report_body(tmp_path, sample_segments):
    """Report file contains the text returned by _call_gemini."""
    with patch("stages.report._call_gemini", return_value=("Unique report body text", "gemini-3.8-flash")):
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
                prompts_dir=str(tmp_path / "prompts"),
                analysis_model="gemini-3.8-flash",
                gemini_api_key="fake-key",
            )

    content = open(out_path, encoding="utf-8").read()
    assert "Unique report body text" in content


def test_run_requires_gemini_api_key(tmp_path, sample_segments):
    """validate_for_report() raises EnvironmentError when gemini key is missing for gemini model."""
    from config import Settings
    s = Settings()
    s.gemini_api_key = ""
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        s.validate_for_report("gemini-3.8-flash")


def test_run_requires_anthropic_api_key(tmp_path, sample_segments):
    """validate_for_report() raises EnvironmentError when anthropic key is missing for claude model."""
    from config import Settings
    s = Settings()
    s.anthropic_api_key = ""
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        s.validate_for_report("claude-haiku-5-5")


# ---------------------------------------------------------------------------
# API key leak prevention — report output files
# ---------------------------------------------------------------------------

def test_report_output_does_not_contain_api_key(tmp_path, sample_segments):
    """Report file content must not contain any API key values."""
    fake_gemini_key = "FAKE-GEMINI-KEY-SHOULD-NOT-APPEAR"
    fake_anthropic_key = "FAKE-ANTHROPIC-KEY-SHOULD-NOT-APPEAR"

    with patch("stages.report._call_gemini", return_value=("Report body", "gemini-3.8-flash")):
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
                prompts_dir=str(tmp_path / "prompts"),
                analysis_model="gemini-3.8-flash",
                gemini_api_key=fake_gemini_key,
                anthropic_api_key=fake_anthropic_key,
            )

    content = open(out_path, encoding="utf-8").read()
    assert fake_gemini_key not in content, "Gemini API key leaked into report file"
    assert fake_anthropic_key not in content, "Anthropic API key leaked into report file"


# ---------------------------------------------------------------------------
# warnings= -- header and prompt metadata
# ---------------------------------------------------------------------------

_WARNING = "Speaker re-identification failed for 80 of 102 segments; labels may be less accurate."


def _run_with_warnings(tmp_path, sample_segments, warnings):
    with patch("stages.report._call_claude", return_value=("# Body", "claude-haiku-5-5")) as mock:
        out_path = report.run(
            segments=sample_segments,
            source_file="test_call.mp3",
            output_dir=str(tmp_path),
            context="friend",
            num_speakers=2,
            audio_duration=15.8,
            speaker_counts={"Speaker A": 2, "Speaker B": 1},
            prompts_dir=str(tmp_path / "prompts"),
            analysis_model="claude-haiku-5-5",
            anthropic_api_key="fake-key",
            warnings=warnings,
        )
    with open(out_path, encoding="utf-8") as f:
        content = f.read()
    user_message = mock.call_args.args[2]
    return content, user_message


@pytest.mark.parametrize("warnings", [None, []])
def test_report_without_warnings_has_the_original_header(tmp_path, sample_segments, warnings):
    content, user_message = _run_with_warnings(tmp_path, sample_segments, warnings)
    lines = content.splitlines()
    assert lines[0] == "# Call Analysis Report"
    assert lines[2:7] == [
        "**Source:** test_call.mp3  ",
        "**Context:** friend  ",
        "**Speakers:** 2  ",
        "**Model:** claude-haiku-5-5  ",
        lines[6],
    ]
    assert lines[6].startswith("**Generated:** ")
    assert lines[7:10] == ["", "---", ""]
    assert "Warning" not in content
    assert "Warning" not in user_message


def test_metadata_block_unchanged_without_warnings():
    args = ("call.mp3", "friend", 2, 15.8, {"Speaker A": 2})
    assert report._build_metadata_block(*args) == report._build_metadata_block(*args, None)
    assert report._build_metadata_block(*args) == report._build_metadata_block(*args, [])
    assert report._build_metadata_block(*args) == (
        "Source file:  call.mp3\nContext:      friend\nSpeakers:     2\n"
        "Duration:     0:15\n  Speaker A: 2 segment(s)"
    )


def test_report_header_and_prompt_show_warnings(tmp_path, sample_segments):
    second = "Only 1 speaker found but 2 were requested; one speaker may have absorbed the other."
    content, user_message = _run_with_warnings(tmp_path, sample_segments, [_WARNING, second])
    lines = content.splitlines()
    assert lines[6].startswith("**Generated:** ")
    assert lines[7:9] == [f"**Warning:** {_WARNING}  ", f"**Warning:** {second}  "]
    assert lines[9:12] == ["", "---", ""]
    assert f"Warning:      {_WARNING}" in user_message
    assert f"Warning:      {second}" in user_message
