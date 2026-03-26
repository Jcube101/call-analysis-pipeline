"""
Stage 3 — Whisper Transcription (faster-whisper)

Steps:
  1. Load the faster-whisper model (downloads on first run if not cached)
  2. For each diarization segment, slice the clean WAV and transcribe
  3. Return a list of enriched segment dicts including the transcribed text

faster-whisper uses CTranslate2 under the hood and is significantly faster
than openai-whisper, especially with CUDA. On GPU it runs with float16;
on CPU it falls back to int8 quantization.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm

from config import settings

# Whisper expects 16 kHz mono float32
_WHISPER_SR = 16_000


def _audio_segment_to_numpy(segment: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment slice to a float32 numpy array for Whisper."""
    seg = segment.set_channels(1).set_frame_rate(_WHISPER_SR)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (seg.sample_width * 8 - 1))
    return samples


def run(
    clean_wav_path: str,
    segments: list[dict],
    model_size: Optional[str] = None,
) -> list[dict]:
    """
    Run Stage 3.

    Args:
        clean_wav_path: Path to the Stage-1 cleaned WAV.
        segments:        Diarization output from Stage 2.
        model_size:      Whisper model size override; falls back to settings.whisper_model.

    Returns:
        The same segment list, each dict extended with a "text" key.
    """
    import torch

    model_name = model_size or settings.whisper_model

    if torch.cuda.is_available():
        device, compute_type = "cuda", "int8_float16"
    else:
        device, compute_type = "cpu", "int8"

    print(f"\n[Stage 3] Loading faster-whisper '{model_name}' on {device} (downloads on first run)...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print(f"[Stage 3] Model loaded. Transcribing {len(segments)} segment(s)...")

    audio = AudioSegment.from_wav(clean_wav_path)
    transcribed: list[dict] = []

    for seg in tqdm(segments, desc="Transcribing", unit="seg"):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        chunk = audio[start_ms:end_ms]

        # Skip very short segments (< 0.5 s) — Whisper may hallucinate on them
        if len(chunk) < 500:
            continue

        audio_array = _audio_segment_to_numpy(chunk)

        # faster-whisper returns (segments_generator, info) — consume the generator
        fw_segments, _ = model.transcribe(audio_array, language="en")
        text = " ".join(s.text.strip() for s in fw_segments).strip()

        transcribed.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "label": seg["label"],
                "text": text,
            }
        )

    print(f"[Stage 3] Transcription complete. {len(transcribed)} segment(s) produced.")
    return transcribed
