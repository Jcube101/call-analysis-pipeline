"""
Stage 3 — Whisper Transcription (faster-whisper)

Two modes, selectable via TRANSCRIPTION_MODE in .env or --transcription-mode CLI flag:

  fast (default):
    Transcribes the full WAV in a single faster-whisper call, then assigns each
    Whisper segment to a diarization speaker using max-overlap logic. Significantly
    faster than accurate mode — one GPU call instead of one per segment.

  accurate:
    One model.transcribe() call per diarization segment. Perfect speaker boundary
    accuracy at the cost of per-segment overhead (~3 s/seg on GPU).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm

from config import settings

# Module-level reference keeps the model alive until process exit.
# ctranslate2's CUDA cleanup calls exit() when triggered mid-process on
# Windows — holding the reference here defers cleanup to process shutdown
# where it is handled safely.
_active_model = None

# Whisper expects 16 kHz mono float32
_WHISPER_SR = 16_000

def _audio_segment_to_numpy(segment: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a float32 numpy array for Whisper."""
    seg = segment.set_channels(1).set_frame_rate(_WHISPER_SR)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (seg.sample_width * 8 - 1))
    return samples


def _assign_speakers(fw_segs: list, diar_segs: list[dict]) -> list[dict]:
    """
    Assign each faster-whisper segment to a diarization speaker using
    max-overlap logic, then merge consecutive segments from the same speaker.

    Args:
        fw_segs:   List of faster-whisper Segment objects (have .start, .end, .text).
        diar_segs: Diarization segments with keys start, end, speaker, label.

    Returns:
        List of segment dicts with keys: start, end, speaker, label, text.
    """
    assigned: list[dict] = []

    for fw in fw_segs:
        best_diar = None
        best_overlap = 0.0

        for diar in diar_segs:
            overlap = min(fw.end, diar["end"]) - max(fw.start, diar["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_diar = diar

        if best_diar is None or best_overlap <= 0:
            continue  # segment falls entirely outside all diarization windows — skip

        assigned.append({
            "start":   fw.start,
            "end":     fw.end,
            "speaker": best_diar["speaker"],
            "label":   best_diar["label"],
            "text":    fw.text.strip(),
        })

    # Merge consecutive segments from the same speaker
    merged: list[dict] = []
    for item in assigned:
        if merged and merged[-1]["speaker"] == item["speaker"]:
            merged[-1]["end"] = item["end"]
            merged[-1]["text"] += " " + item["text"]
        else:
            merged.append(dict(item))

    return merged


def _transcribe_accurate(
    model: WhisperModel,
    audio: AudioSegment,
    segments: list[dict],
) -> list[dict]:
    """
    One model.transcribe() call per diarization segment.
    Perfect speaker boundary accuracy; slower due to per-call GPU overhead.
    """
    transcribed: list[dict] = []

    for seg in tqdm(segments, desc="Transcribing (accurate)", unit="seg"):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        chunk = audio[start_ms:end_ms]

        if len(chunk) < 500:
            continue

        audio_array = _audio_segment_to_numpy(chunk)
        fw_segments, _ = model.transcribe(audio_array, language=settings.whisper_language)
        text = " ".join(s.text.strip() for s in fw_segments).strip()

        transcribed.append({
            "start":   seg["start"],
            "end":     seg["end"],
            "speaker": seg["speaker"],
            "label":   seg["label"],
            "text":    text,
        })

    return transcribed


def _transcribe_fast(
    model: WhisperModel,
    audio: AudioSegment,
    segments: list[dict],
) -> list[dict]:
    """
    Transcribe the full WAV in one faster-whisper call, then assign each
    resulting segment to a speaker via max-overlap with diarization windows.
    Much faster than accurate mode for typical recordings.
    """
    audio_array = _audio_segment_to_numpy(audio)
    fw_segs_gen, _ = model.transcribe(audio_array, language=settings.whisper_language, vad_filter=True)
    fw_segs = list(fw_segs_gen)  # consume generator before alignment
    return _assign_speakers(fw_segs, segments)


def run(
    clean_wav_path: str,
    segments: list[dict],
    model_size: Optional[str] = None,
    mode: str = "fast",
) -> list[dict]:
    """
    Run Stage 3.

    Args:
        clean_wav_path: Path to the Stage-1 cleaned WAV.
        segments:       Diarization output from Stage 2.
        model_size:     Whisper model size override; falls back to settings.whisper_model.
        mode:           "fast" or "accurate" (default: "fast").

    Returns:
        The segment list, each dict extended with a "text" key.
    """
    import torch

    model_name = model_size or settings.whisper_model

    if torch.cuda.is_available():
        device, compute_type = "cuda", "int8_float16"
    else:
        device, compute_type = "cpu", "int8"

    print(f"\n[Stage 3] Loading faster-whisper '{model_name}' on {device} (downloads on first run)...")
    global _active_model
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _active_model = model  # prevent GC / CUDA teardown until process exit
    print(f"[Stage 3] Model loaded. Running in '{mode}' mode on {len(segments)} segment(s)...")

    audio = AudioSegment.from_wav(clean_wav_path)

    if mode == "accurate":
        transcribed = _transcribe_accurate(model, audio, segments)
    else:
        transcribed = _transcribe_fast(model, audio, segments)

    print(f"[Stage 3] Transcription complete. {len(transcribed)} segment(s) produced.")
    return transcribed
