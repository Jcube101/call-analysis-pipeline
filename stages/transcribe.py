"""
Stage 3 — Whisper Transcription (faster-whisper)

Two modes, selectable via TRANSCRIPTION_MODE in .env or --transcription-mode CLI flag:

  accurate (default):
    One model.transcribe() call per diarization segment. Perfect speaker boundary
    accuracy. Each segment maps exactly to a diarization window.

  fast:
    Merges consecutive same-speaker diarization segments into "turns" (gap ≤ 1s),
    then makes one model.transcribe() call per turn. Produces one output line per
    speaker turn rather than per diarization segment — coarser than accurate but
    10-20x fewer Whisper calls on long recordings. Speaker accuracy is preserved
    because turns are built from diarization, not from Whisper's internal chunking.

Both modes attach a segment-level confidence score (derived from Whisper's
avg_logprob) to every output segment.  When word_timestamps=True is passed,
each segment also includes a "words" list with per-word start/end times and
probability scores.
"""

from __future__ import annotations

import math
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


def _confidence_from_segments(fw_segs: list) -> float:
    """
    Compute a 0–1 confidence score from a list of faster-whisper segments.

    Uses duration-weighted average of exp(avg_logprob) across all segments.
    Clamped to [0, 1] and rounded to 3 decimal places.
    """
    if not fw_segs:
        return 0.0
    total_dur = sum(max(0.0, s.end - s.start) for s in fw_segs)
    if total_dur > 0:
        weighted = sum((s.end - s.start) * s.avg_logprob for s in fw_segs)
        raw = math.exp(weighted / total_dur)
    else:
        raw = math.exp(sum(s.avg_logprob for s in fw_segs) / len(fw_segs))
    return round(min(1.0, max(0.0, raw)), 3)


def _words_from_segments(fw_segs: list, time_offset: float = 0.0) -> list[dict]:
    """
    Extract per-word timestamps from a list of faster-whisper segments.

    Args:
        fw_segs:     faster-whisper Segment objects (must have been transcribed
                     with word_timestamps=True).
        time_offset: seconds to add to each word's start/end (used in accurate
                     mode where the audio chunk starts at seg["start"]).

    Returns:
        List of dicts: {word, start, end, probability}
    """
    words = []
    for s in fw_segs:
        if not s.words:
            continue
        for w in s.words:
            words.append({
                "word": w.word,
                "start": round(w.start + time_offset, 3),
                "end":   round(w.end   + time_offset, 3),
                "probability": round(w.probability, 3),
            })
    return words


def _merge_turns(segments: list[dict], gap_s: float = 1.0) -> list[dict]:
    """
    Merge consecutive same-speaker diarization segments into speaker turns.

    Two segments from the same speaker are merged into one turn when the gap
    between them is ≤ gap_s seconds.  The merged turn keeps the start of the
    first segment and the end of the last.

    Used by fast mode to reduce the number of Whisper calls from one-per-
    segment to one-per-turn while preserving speaker-accurate boundaries.
    """
    if not segments:
        return []
    turns = [dict(segments[0])]
    for seg in segments[1:]:
        last = turns[-1]
        if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] <= gap_s:
            last["end"] = seg["end"]
        else:
            turns.append(dict(seg))
    return turns


def _transcribe_accurate(
    model: WhisperModel,
    audio: AudioSegment,
    segments: list[dict],
    word_timestamps: bool = False,
) -> list[dict]:
    """
    One model.transcribe() call per diarization segment.
    Perfect speaker boundary accuracy; slower due to per-call GPU overhead.
    """
    transcribed: list[dict] = []

    for seg in tqdm(segments, desc="Transcribing (accurate)", unit="seg"):
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        chunk = audio[start_ms:end_ms]

        if len(chunk) < 500:
            continue

        audio_array = _audio_segment_to_numpy(chunk)
        fw_segs_gen, _ = model.transcribe(
            audio_array,
            language=settings.whisper_language,
            word_timestamps=word_timestamps,
        )
        fw_segs = list(fw_segs_gen)
        text = " ".join(s.text.strip() for s in fw_segs).strip()

        out: dict = {
            "start":      seg["start"],
            "end":        seg["end"],
            "speaker":    seg["speaker"],
            "label":      seg["label"],
            "text":       text,
            "confidence": _confidence_from_segments(fw_segs),
        }
        if word_timestamps:
            out["words"] = _words_from_segments(fw_segs, time_offset=seg["start"])

        transcribed.append(out)

    return transcribed


def _transcribe_fast(
    model: WhisperModel,
    audio: AudioSegment,
    segments: list[dict],
    word_timestamps: bool = False,
) -> list[dict]:
    """
    Merge same-speaker diarization segments into turns, then transcribe
    one turn at a time.

    One Whisper call per speaker turn (≈ 10-20x fewer calls than accurate
    mode on long recordings) while preserving speaker-accurate boundaries.
    Output is one segment per speaker turn rather than per diarization segment.
    """
    turns = _merge_turns(segments, gap_s=1.0)
    transcribed: list[dict] = []

    for turn in tqdm(turns, desc="Transcribing (fast)", unit="turn"):
        start_ms = int(turn["start"] * 1000)
        end_ms   = int(turn["end"]   * 1000)
        chunk = audio[start_ms:end_ms]

        if len(chunk) < 500:
            continue

        audio_array = _audio_segment_to_numpy(chunk)
        fw_segs_gen, _ = model.transcribe(
            audio_array,
            language=settings.whisper_language,
            word_timestamps=word_timestamps,
        )
        fw_segs = list(fw_segs_gen)
        text = " ".join(s.text.strip() for s in fw_segs).strip()

        out: dict = {
            "start":      turn["start"],
            "end":        turn["end"],
            "speaker":    turn["speaker"],
            "label":      turn["label"],
            "text":       text,
            "confidence": _confidence_from_segments(fw_segs),
        }
        if word_timestamps:
            out["words"] = _words_from_segments(fw_segs, time_offset=turn["start"])

        transcribed.append(out)

    return transcribed


def run(
    clean_wav_path: str,
    segments: list[dict],
    model_size: Optional[str] = None,
    mode: str = "accurate",
    word_timestamps: bool = False,
) -> list[dict]:
    """
    Run Stage 3.

    Args:
        clean_wav_path:  Path to the Stage-1 cleaned WAV.
        segments:        Diarization output from Stage 2.
        model_size:      Whisper model size override; falls back to settings.whisper_model.
        mode:            "accurate" (default) or "fast".
        word_timestamps: When True, include per-word timestamps in each segment.

    Returns:
        The segment list with "text", "confidence", and optionally "words" added.
    """
    import torch

    model_name = model_size or settings.whisper_model

    if torch.cuda.is_available():
        device, compute_type = "cuda", "int8_float16"
    else:
        device, compute_type = "cpu", "int8"

    wt_note = " + word timestamps" if word_timestamps else ""
    print(f"\n[Stage 3] Loading faster-whisper '{model_name}' on {device} (downloads on first run)...")
    global _active_model
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _active_model = model  # prevent GC / CUDA teardown until process exit
    print(f"[Stage 3] Model loaded. Running in '{mode}' mode{wt_note} on {len(segments)} segment(s)...")

    audio = AudioSegment.from_wav(clean_wav_path)

    if mode == "accurate":
        transcribed = _transcribe_accurate(model, audio, segments, word_timestamps=word_timestamps)
    else:
        transcribed = _transcribe_fast(model, audio, segments, word_timestamps=word_timestamps)

    print(f"[Stage 3] Transcription complete. {len(transcribed)} segment(s) produced.")
    return transcribed
