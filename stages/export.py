"""
Stage 4 — Export

Produces two output files from the enriched transcript:

  transcript.txt  — human-readable, one line per segment
  transcript.json — structured JSON with a metadata header

JSON schema
-----------
{
  "metadata": {
    "source_file": "test_call.mp3",
    "context": "friend",
    "num_speakers": 2,
    "processed_at": "2024-01-15T14:30:00"
  },
  "transcript": [
    {
      "start": 4.2,
      "end": 9.8,
      "speaker": "Speaker A",
      "text": "Hey, how are you doing..."
    }
  ]
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional


def _format_timestamp(seconds: float) -> str:
    """Convert fractional seconds → [HH:MM:SS] display string."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(
    segments: list[dict],
    source_file: str,
    output_dir: str,
    context: str = "friend",
    num_speakers: Optional[int] = None,
) -> tuple[str, str]:
    """
    Run Stage 4.

    Args:
        segments:     Enriched segments from Stage 3 (each has start/end/speaker/text).
        source_file:  Original input filename (used in metadata only).
        output_dir:   Directory where outputs are written.
        context:      Conversation context tag.
        num_speakers: Speaker count (may be None if auto-detected).

    Returns:
        (txt_path, json_path) — absolute paths to the two output files.
    """
    print(f"\n[Stage 4] Writing outputs to: {output_dir}")

    # --- Resolve actual speaker count from data if not provided ---
    if num_speakers is None:
        num_speakers = len(set(s["speaker"] for s in segments))

    metadata = {
        "source_file": os.path.basename(source_file),
        "context": context,
        "num_speakers": num_speakers,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    # --- TXT ---
    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Call Transcript\n")
        f.write(f"# Source:  {metadata['source_file']}\n")
        f.write(f"# Context: {metadata['context']}\n")
        f.write(f"# Speakers: {metadata['num_speakers']}\n")
        f.write(f"# Processed: {metadata['processed_at']}\n\n")

        for seg in segments:
            ts = _format_timestamp(seg["start"])
            f.write(f"[{ts}] {seg['speaker']}: \"{seg['text']}\"\n")

    print(f"[Stage 4] Text transcript saved:  {txt_path}")

    # --- JSON ---
    json_payload = {
        "metadata": metadata,
        "transcript": [
            {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "speaker": seg["speaker"],
                "text": seg["text"],
            }
            for seg in segments
        ],
    }

    json_path = os.path.join(output_dir, "transcript.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"[Stage 4] JSON transcript saved:  {json_path}")

    return txt_path, json_path
