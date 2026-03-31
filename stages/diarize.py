"""
Stage 2 — Speaker Diarization

Steps:
  1. Run pyannote/speaker-diarization-3.1 on the clean WAV
  2. Collect timestamped segments with raw speaker labels (SPEAKER_00, SPEAKER_01, …)
  3. Re-identify speakers globally via voice embeddings + clustering (fixes label
     flipping on long recordings — pyannote processes audio in chunks and can
     inconsistently assign the same physical voice to different labels across chunks)
  4. Return a list of segment dicts for the transcription stage
"""

from __future__ import annotations

import gc
import os
import warnings
from typing import Optional

import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="torchcodec is not installed", category=UserWarning)
    from pyannote.audio import Pipeline

from config import settings


# Map raw pyannote labels → human-friendly labels used in the transcript
def _label_map(raw_labels: list[str]) -> dict[str, str]:
    """SPEAKER_00 → Speaker A, SPEAKER_01 → Speaker B, etc."""
    sorted_labels = sorted(set(raw_labels))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {label: f"Speaker {letters[i]}" for i, label in enumerate(sorted_labels)}


def _reidentify_speakers(
    segments: list[dict],
    waveform: "torch.Tensor",
    sample_rate: int,
    num_speakers: int,
) -> list[dict]:
    """
    Re-assign speaker labels using MFCC features + KMeans clustering.

    pyannote processes long audio in chunks and can flip speaker labels between
    chunks (e.g. the same physical voice becomes SPEAKER_00 in the first half
    and SPEAKER_01 in the second half). This function computes MFCC + delta
    features per segment, clusters them globally, then reassigns labels in
    first-appearance order so the same voice always maps to the same label.

    MFCC features (via librosa, already a pipeline dependency) are used instead
    of pyannote's internal embedding model because pyannote's SpeakerEmbedding
    pipeline calls .to(device) on the AudioFile dict, which fails for plain
    Python dicts.  MFCC features are sufficient to distinguish speakers
    reliably on two-speaker call recordings.

    Segments shorter than MIN_SEG_SECONDS are skipped during feature extraction
    and inherit their label from the nearest longer segment.

    Returns the modified segment list, or the original list unchanged if
    re-identification fails for any reason.
    """
    import librosa
    from tqdm import tqdm

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        print("[Stage 2] scikit-learn not installed — skipping speaker re-identification")
        return segments

    MIN_SEG_SECONDS = 1.0
    N_MFCC = 20

    # Convert waveform to a mono float32 numpy array for librosa.
    # waveform shape is (channels, samples); take channel 0.
    audio_np = waveform[0].numpy().astype(np.float32)

    embeddings: list[np.ndarray] = []
    long_indices: list[int] = []
    n_short = 0
    n_failed = 0
    first_error: str = ""

    print(f"[Stage 2] Extracting MFCC features ({len(segments)} segments, ≥{MIN_SEG_SECONDS}s only)...")

    for i, seg in enumerate(tqdm(segments, desc="  Features", unit="seg")):
        duration = seg["end"] - seg["start"]
        if duration < MIN_SEG_SECONDS:
            n_short += 1
            continue

        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        seg_audio = audio_np[start_sample:end_sample]

        try:
            mfcc = librosa.feature.mfcc(y=seg_audio, sr=sample_rate, n_mfcc=N_MFCC)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            # Mean + std pooling over time → fixed-size feature vector (5 * N_MFCC,)
            feat = np.concatenate([
                mfcc.mean(axis=1),
                delta.mean(axis=1),
                delta2.mean(axis=1),
                mfcc.std(axis=1),
                delta.std(axis=1),
            ])
            embeddings.append(feat)
            long_indices.append(i)
        except Exception as exc:
            n_failed += 1
            if not first_error:
                first_error = str(exc)
            continue

    if n_failed > 0:
        print(f"[Stage 2] {n_failed} segment(s) failed feature extraction (first error: {first_error})")

    if len(embeddings) < num_speakers:
        print(
            f"[Stage 2] Only {len(embeddings)} usable features for {num_speakers} speakers "
            f"(filtered short: {n_short}, failed: {n_failed}) — skipping re-identification"
        )
        return segments

    def _smooth_labels(segments: list[dict], labels: np.ndarray, window: int = 5) -> np.ndarray:
        """Majority-vote smoothing over a sliding window to fix isolated label flips."""
        smoothed = labels.copy()
        for i in range(len(labels)):
            start = max(0, i - window // 2)
            end = min(len(labels), i + window // 2 + 1)
            window_labels = labels[start:end]
            smoothed[i] = max(set(window_labels), key=list(window_labels).count)
        return smoothed

    # Cluster globally
    emb_matrix = normalize(np.array(embeddings))
    kmeans = KMeans(
        n_clusters=num_speakers,
        n_init=20,
        max_iter=500,
        random_state=42,
        tol=1e-6,
    )
    cluster_ids = kmeans.fit_predict(emb_matrix)
    cluster_ids = _smooth_labels(segments, cluster_ids)

    # Map cluster IDs → SPEAKER_XX in first-appearance order
    cluster_to_label: dict[int, str] = {}
    counter = 0
    for cid in cluster_ids:
        if cid not in cluster_to_label:
            cluster_to_label[cid] = f"SPEAKER_{counter:02d}"
            counter += 1

    # Write new labels back to long segments
    long_set = set(long_indices)
    for seg_idx, cid in zip(long_indices, cluster_ids):
        segments[seg_idx]["label"] = cluster_to_label[cid]

    # Short segments inherit the label of their nearest long segment (by midpoint)
    for i, seg in enumerate(segments):
        if i in long_set:
            continue
        seg_mid = (seg["start"] + seg["end"]) / 2
        nearest = min(
            long_indices,
            key=lambda li: abs((segments[li]["start"] + segments[li]["end"]) / 2 - seg_mid),
        )
        segments[i]["label"] = segments[nearest]["label"]

    n_distinct = len(set(s["label"] for s in segments))
    print(f"[Stage 2] Re-identification complete — {n_distinct} distinct speaker(s) after clustering")
    return segments


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
        annotation = diarization
    elif hasattr(diarization, "exclusive_speaker_diarization"):
        annotation = diarization.exclusive_speaker_diarization
    elif hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    else:
        raise RuntimeError(
            f"[Stage 2] Cannot parse diarization output of type {type(diarization)}. "
            f"Available attrs: {[a for a in dir(diarization) if not a.startswith('_')]}"
        )

    raw_segments = [
        {"start": segment.start, "end": segment.end, "label": speaker}
        for segment, track, speaker in annotation.itertracks(yield_label=True)
    ]

    if not raw_segments:
        raise RuntimeError(
            "[Stage 2] Diarization returned no segments. "
            "Check that the audio file is valid and longer than a few seconds."
        )

    # Re-identify speakers globally — must happen before we free the waveform
    num_spk = num_speakers or len(set(s["label"] for s in raw_segments))
    try:
        raw_segments = _reidentify_speakers(raw_segments, waveform, sample_rate, num_spk)
    except Exception as e:
        print(f"[Stage 2] Speaker re-identification failed ({e}) — using pyannote labels as-is")

    # Free the waveform tensors now that re-identification is done
    del data, waveform, audio_input
    gc.collect()

    # Build friendly labels
    label_map = _label_map([s["label"] for s in raw_segments])
    for seg in raw_segments:
        seg["speaker"] = label_map[seg["label"]]

    print(
        f"[Stage 2] Found {len(set(s['speaker'] for s in raw_segments))} speaker(s), "
        f"{len(raw_segments)} segments"
    )

    # Release the pyannote model from VRAM before Stage 3 loads Whisper.
    # Without this, both models sit in VRAM simultaneously (~3.7/4.0 GB on GTX 1650).
    # Moving to CPU + clearing the cache drops Stage 3 VRAM usage by ~1-1.5 GB.
    try:
        pipeline.to(torch.device("cpu"))
        del pipeline, diarization, annotation
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass  # non-critical — pipeline will be GC'd eventually regardless

    return raw_segments
