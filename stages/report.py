"""
Stage 5 — Analysis Report (Gemini / Claude API)

Sends the transcript to an LLM and writes a context-aware Markdown report.
Supports both Google Gemini (gemini-* models) and Anthropic Claude (claude-* models).

The prompt is loaded from prompts/<context>.md — edit those files to customise
the analysis focus for each conversation type.

For transcripts that exceed the context window the transcript is split into
overlapping chunks, each chunk is analysed independently, and a final synthesis
pass combines the partial reports.

Output
------
  output/<name>_<timestamp>_report.md   — full Markdown report
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

# Loaded lazily so the import doesn't fail when --report is not used
_genai = None

# Default model for report generation
_MODEL = "claude-haiku-4-5-20251001"

# Max characters per chunk sent to Gemini (~1M token context, cap at 500k chars ~125k tokens)
_CHUNK_CHARS = 500_000

# Number of characters of overlap between adjacent chunks
_OVERLAP_CHARS = 2_000

# Default prompts used when the prompts/<context>.md file is missing
_DEFAULT_PROMPTS: dict[str, str] = {
    "friend": (
        "Analyse this conversation between friends. Cover: emotional tone, recurring "
        "themes, engagement level, any concerns worth following up on, and conversation "
        "balance. Write a concise Markdown report with clear section headings."
    ),
    "work": (
        "Analyse this work conversation. Cover: action items with owners, decisions made, "
        "open questions, key topics, risks or blockers, and overall meeting effectiveness. "
        "Write a concise Markdown report. Use bullet points for action items."
    ),
    "interview": (
        "Analyse this job interview transcript. Cover: candidate strengths, weaknesses/gaps, "
        "communication style, notable answers, cultural fit, suggested follow-up questions, "
        "and an overall recommendation (proceed / hold / pass). Write a concise Markdown "
        "report backed by specific examples."
    ),
    "work_interview": (
        "Analyse this job interview transcript. Cover: candidate strengths, weaknesses/gaps, "
        "communication style, notable answers, cultural fit, suggested follow-up questions, "
        "and an overall recommendation (proceed / hold / pass). Write a concise Markdown "
        "report backed by specific examples."
    ),
    "user_interview": (
        "Analyse this user interview transcript. Cover: the core problem the participant described, "
        "their current behaviour and workarounds, trigger moments for the pain, their definition of "
        "done, notable quotes, surprises or contradictions, and the 2–3 strongest product signals. "
        "Write a concise Markdown report backed by specific transcript references."
    ),
    "date": (
        "Analyse this conversation. Cover: compatibility signals, conversation balance, "
        "shared interests, moments of connection or awkwardness, communication style, "
        "and overall vibe. Write a concise Markdown report."
    ),
}


def _load_prompt(context: str, prompts_dir: str) -> str:
    """Return the instruction prompt for `context`, falling back to built-in defaults."""
    prompt_path = os.path.join(prompts_dir, f"{context}.md")
    if os.path.isfile(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return _DEFAULT_PROMPTS.get(context, _DEFAULT_PROMPTS["friend"])


def _build_transcript_text(segments: list[dict]) -> str:
    """Render segments as plain text for inclusion in the prompt."""
    lines = []
    for seg in segments:
        total = int(seg["start"])
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        ts = f"{h:02d}:{m:02d}:{s:02d}"
        lines.append(f"[{ts}] {seg['speaker']}: {seg['text']}")
    return "\n".join(lines)


def _build_metadata_block(
    source_file: str,
    context: str,
    num_speakers: int,
    audio_duration: Optional[float],
    speaker_counts: dict[str, int],
) -> str:
    """Build a human-readable metadata header for the prompt."""
    lines = [
        f"Source file:  {os.path.basename(source_file)}",
        f"Context:      {context}",
        f"Speakers:     {num_speakers}",
    ]
    if audio_duration is not None:
        total = int(audio_duration)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        duration_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        lines.append(f"Duration:     {duration_str}")
    for spk, cnt in sorted(speaker_counts.items()):
        lines.append(f"  {spk}: {cnt} segment(s)")
    return "\n".join(lines)


GEMINI_FALLBACK_MODEL = "gemini-3-flash-preview"
CLAUDE_FALLBACK_MODEL = "claude-haiku-4-5-20251001"


def _call_gemini_model(client, system_prompt: str, user_message: str, model: str) -> str:
    """Send one request to Gemini and return the response text."""
    full_prompt = f"{system_prompt}\n\n{user_message}"
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
    )
    return response.text


def _call_gemini(client, system_prompt: str, user_message: str, model: str) -> tuple[str, str]:
    """Call Gemini, falling back to Flash on 503. Returns (response_text, actual_model)."""
    try:
        return _call_gemini_model(client, system_prompt, user_message, model), model
    except Exception as e:
        if "503" in str(e) and model != GEMINI_FALLBACK_MODEL:
            print(f"[Stage 5] {model} unavailable, falling back to {GEMINI_FALLBACK_MODEL}...")
            return _call_gemini_model(client, system_prompt, user_message, GEMINI_FALLBACK_MODEL), GEMINI_FALLBACK_MODEL
        raise


def _call_claude_model(api_key: str, system_prompt: str, user_message: str, model: str) -> str:
    """Send one request to Claude and return the response text."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def _call_claude(api_key: str, system_prompt: str, user_message: str, model: str) -> tuple[str, str]:
    """Call Claude, falling back to Haiku on failure. Returns (response_text, actual_model)."""
    try:
        return _call_claude_model(api_key, system_prompt, user_message, model), model
    except Exception as e:
        if model != CLAUDE_FALLBACK_MODEL:
            print(f"[Stage 5] {model} failed ({e}), falling back to {CLAUDE_FALLBACK_MODEL}...")
            return _call_claude_model(api_key, system_prompt, user_message, CLAUDE_FALLBACK_MODEL), CLAUDE_FALLBACK_MODEL
        raise


def _split_transcript(transcript_text: str) -> list[str]:
    """Split a long transcript into overlapping chunks by line boundaries."""
    if len(transcript_text) <= _CHUNK_CHARS:
        return [transcript_text]

    chunks: list[str] = []
    lines = transcript_text.splitlines(keepends=True)
    current: list[str] = []
    current_len = 0

    for line in lines:
        current.append(line)
        current_len += len(line)

        if current_len >= _CHUNK_CHARS:
            chunks.append("".join(current))
            # Keep the last few lines as overlap for the next chunk
            overlap_chars = 0
            overlap_lines: list[str] = []
            for ol in reversed(current):
                overlap_chars += len(ol)
                overlap_lines.insert(0, ol)
                if overlap_chars >= _OVERLAP_CHARS:
                    break
            current = overlap_lines
            current_len = sum(len(l) for l in current)

    if current:
        chunks.append("".join(current))

    return chunks


def run(
    segments: list[dict],
    source_file: str,
    output_dir: str,
    context: str,
    num_speakers: int,
    audio_duration: Optional[float],
    speaker_counts: dict[str, int],
    api_key: str = "",
    prompts_dir: str = "prompts",
    gemini_model: str = _MODEL,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    context_hints: str = "",
) -> str:
    """
    Run Stage 5.

    Routes to Claude or Gemini based on the gemini_model prefix.
    For backwards compatibility, `api_key` is still accepted and used
    as the Gemini key when `gemini_api_key` is not provided.

    Returns:
        Path to the written report file.
    """
    use_claude = gemini_model.startswith("claude-")

    if use_claude:
        _api_key = anthropic_api_key or ""
    else:
        _api_key = gemini_api_key or api_key or ""
        global _genai
        if _genai is None:
            from google import genai as _genai_module
            _genai = _genai_module

    print(f"\n[Stage 5] Generating analysis report (context: {context}, model: {gemini_model})...")

    instruction_prompt = _load_prompt(context, prompts_dir)
    if context_hints.strip():
        instruction_prompt = (
            "CONTEXT HINTS (use these for correct spelling "
            "of names, places, companies, and technical terms "
            "— do not infer alternatives):\n"
            f"{context_hints.strip()}\n\n"
            + instruction_prompt
        )
    metadata_block = _build_metadata_block(
        source_file, context, num_speakers, audio_duration, speaker_counts
    )
    transcript_text = _build_transcript_text(segments)

    chunks = _split_transcript(transcript_text)
    n_chunks = len(chunks)

    def _call(system_prompt: str, user_msg: str, model: str) -> tuple[str, str]:
        if use_claude:
            return _call_claude(_api_key, system_prompt, user_msg, model)
        client = _genai.Client(api_key=_api_key)
        return _call_gemini(client, system_prompt, user_msg, model)

    actual_model = gemini_model
    provider = "Claude" if use_claude else "Gemini"

    if n_chunks == 1:
        print(f"[Stage 5] Sending transcript to {provider} ({len(transcript_text):,} chars)...")
        user_message = (
            f"## Conversation metadata\n\n{metadata_block}\n\n"
            f"## Transcript\n\n{transcript_text}"
        )
        report_body, actual_model = _call(instruction_prompt, user_message, gemini_model)
    else:
        print(f"[Stage 5] Transcript is large — splitting into {n_chunks} chunks...")
        partial_reports: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            print(f"[Stage 5] Analysing chunk {i}/{n_chunks} ({len(chunk):,} chars)...")
            user_message = (
                f"## Conversation metadata\n\n{metadata_block}\n\n"
                f"## Transcript (part {i} of {n_chunks})\n\n{chunk}"
            )
            partial, actual_model = _call(instruction_prompt, user_message, gemini_model)
            partial_reports.append(f"### Part {i} of {n_chunks}\n\n{partial}")

        print(f"[Stage 5] Synthesising {n_chunks} partial reports...")
        synthesis_prompt = (
            "You have received partial analysis reports for different sections of the same "
            "conversation. Combine them into one coherent, de-duplicated final report. "
            "Use clear Markdown section headings. Remove any repeated observations. "
            "Maintain the analytical focus from the original instruction."
        )
        combined_partials = "\n\n---\n\n".join(partial_reports)
        report_body, actual_model = _call(synthesis_prompt, combined_partials, gemini_model)

    print(f"[Stage 5] Report generated using {actual_model}")

    # Build the full report with a header
    processed_at = datetime.now().isoformat(timespec="seconds")
    report_lines = [
        "# Call Analysis Report",
        "",
        f"**Source:** {os.path.basename(source_file)}  ",
        f"**Context:** {context}  ",
        f"**Speakers:** {num_speakers}  ",
        f"**Model:** {actual_model}  ",
        f"**Generated:** {processed_at}  ",
        "",
        "---",
        "",
        report_body,
    ]
    report_content = "\n".join(report_lines)

    # Write to file
    source_stem = os.path.splitext(os.path.basename(source_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{source_stem}_{timestamp}_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Stage 5] Report saved: {report_path}")
    return report_path
