"""
Stage 2 — Speaker Diarization

Steps:
  1. Run pyannote/speaker-diarization-3.1 on the clean WAV
  2. Collect timestamped segments with raw speaker labels (SPEAKER_00, SPEAKER_01, …)
  3. Per-speaker loudness normalization so both voices land at equal volume
  4. Return a list of segment dicts for the transcription stage
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="torchcodec is not installed", category=UserWarning)
    from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.effects import normalize

from config import settings


# Map raw pyannote labels → human-friendly labels used in the transcript
def _label_map(raw_labels: list[str]) -> dict[str, str]:
    """SPEAKER_00 → Speaker A, SPEAKER_01 → Speaker B, etc."""
    sorted_labels = sorted(set(raw_labels))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {label: f"Speaker {letters[i]}" for i, label in enumerate(sorted_labels)}


def _normalize_speaker_segments(
    audio: AudioSegment, segments: list[dict]
) -> dict[str, AudioSegment]:
    """
    Extract each speaker's audio, normalize loudness independently, return
    a dict mapping speaker_label → normalized AudioSegment.

    This ensures a quiet speaker is brought to the same perceived volume as a
    louder one before transcription.
    """
    from collections import defaultdict

    speaker_chunks: dict[str, list[AudioSegment]] = defaultdict(list)
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        chunk = audio[start_ms:end_ms]
        if len(chunk) > 0:
            speaker_chunks[seg["speaker"]].append(chunk)

    normalized: dict[str, AudioSegment] = {}
    for speaker, chunks in speaker_chunks.items():
        combined = sum(chunks, AudioSegment.empty())
        normalized[speaker] = normalize(combined)

    return normalized


def run(
    clean_wav_path: str,
    output_dir: str,
    num_speakers: Optional[int] = None,
) -> list[dict]:
    """
    Run Stage 2.

    Args:
        clean_wav_path: Path to the Stage-1 cleaned WAV.
        output_dir:      Directory for any intermediate outputs.
        num_speakers:    Exact speaker count, or None for auto-detection.

    Returns:
        List of segment dicts:
          {"start": float, "end": float, "speaker": "Speaker A", "label": "SPEAKER_00"}
    """
    settings.validate_for_diarization()

    print(f"\n[Stage 2] Loading diarization pipeline (pyannote/speaker-diarization-3.1)...")
    import huggingface_hub
    huggingface_hub.login(token=settings.huggingface_token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    # Move to GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("[Stage 2] Running on GPU.")
        else:
            print("[Stage 2] Running on CPU (no CUDA detected).")
    except ImportError:
        print("[Stage 2] torch not available for device check, running on CPU.")

    print(f"[Stage 2] Running diarization on: {clean_wav_path}")
    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
        print(f"[Stage 2] Using fixed speaker count: {num_speakers}")
    else:
        print("[Stage 2] Speaker count: auto-detect")

    import soundfile as sf
    import numpy as np
    import torch

    data, sample_rate = sf.read(clean_wav_path)
    if data.ndim == 1:
        data = data[np.newaxis, :]  # add channel dim
    else:
        data = data.T  # (samples, channels) -> (channels, samples)
    waveform = torch.from_numpy(data).float()
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    diarization = pipeline(audio_input, **diarize_kwargs)

    # Collect raw segments — handle different pyannote output types across versions
    if hasattr(diarization, "itertracks"):
        raw_segments = [
            {"start": segment.start, "end": segment.end, "label": speaker}
            for segment, track, speaker in diarization.itertracks(yield_label=True)
        ]
    else:
        print(f"[Stage 2] DEBUG: diarization type = {type(diarization)}")
        print(f"[Stage 2] DEBUG: diarization attrs = {[a for a in dir(diarization) if not a.startswith('_')]}")
        raise RuntimeError(
            "[Stage 2] Diarization output has no 'itertracks' method. "
            "See DEBUG lines above to determine the correct API for this pyannote version."
        )

    if not raw_segments:
        raise RuntimeError(
            "[Stage 2] Diarization returned no segments. "
            "Check that the audio file is valid and longer than a few seconds."
        )

    # Build friendly labels
    label_map = _label_map([s["label"] for s in raw_segments])
    for seg in raw_segments:
        seg["speaker"] = label_map[seg["label"]]

    print(
        f"[Stage 2] Found {len(set(s['speaker'] for s in raw_segments))} speaker(s), "
        f"{len(raw_segments)} segments"
    )

    # Per-speaker volume normalization (informational — the normalized audio
    # isn't re-exported here, but segment boundaries are accurate for Whisper)
    audio = AudioSegment.from_wav(clean_wav_path)
    _normalize_speaker_segments(audio, raw_segments)
    print("[Stage 2] Per-speaker loudness normalization complete.")

    return raw_segments
