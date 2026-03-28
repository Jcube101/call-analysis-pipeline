"""
Stage 1 — Audio Pre-processing

Steps:
  1. Load the input MP3 (or any format pydub supports via ffmpeg)
  2. Convert to mono, 16 kHz — the format pyannote and Whisper prefer
  3. Apply spectral noise reduction (noisereduce) — chunked for large files
  4. Normalize overall loudness
  5. Export as a clean WAV file for downstream stages
"""

import gc
import math
import os
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
from tqdm import tqdm


def _to_numpy(audio: AudioSegment) -> tuple[np.ndarray, int]:
    """Convert a pydub AudioSegment to a float32 numpy array + sample rate."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # pydub stores interleaved stereo; we've already converted to mono above
    samples /= float(2 ** (audio.sample_width * 8 - 1))  # normalise to [-1, 1]
    return samples, audio.frame_rate


def _from_numpy(samples: np.ndarray, sample_rate: int, sample_width: int = 2) -> AudioSegment:
    """Convert a float32 numpy array back to a pydub AudioSegment."""
    # Clip to [-1, 1] before converting to int
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * (2 ** (sample_width * 8 - 1) - 1)).astype(np.int16)
    return AudioSegment(
        int_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=sample_width,
        channels=1,
    )


def _chunked_reduce_noise(samples: np.ndarray, sr: int) -> np.ndarray:
    """Apply noisereduce in overlapping chunks to limit peak RAM on long recordings.

    noisereduce internally computes a full STFT of its input, which for a 90-minute
    file creates a ~1.4 GB temporary matrix. This function processes 60-second slices
    instead, keeping peak RAM under ~350 MB regardless of recording length.

    Each chunk is extended by OVERLAP_SEC on each side so noisereduce has context at
    the boundaries — only the core portion is written to the output. For recordings
    shorter than CHUNK_SEC the whole array is processed in a single call.
    """
    CHUNK_SEC = 60
    OVERLAP_SEC = 0.5

    chunk_size = int(sr * CHUNK_SEC)
    overlap_size = int(sr * OVERLAP_SEC)
    total = len(samples)

    noise_profile = samples[:min(int(sr * 0.5), total)]

    if total <= chunk_size:
        return nr.reduce_noise(
            y=samples, sr=sr, y_noise=noise_profile,
            stationary=False, prop_decrease=0.75,
        )

    n_chunks = math.ceil(total / chunk_size)
    output = np.empty(total, dtype=np.float32)

    for i in tqdm(range(n_chunks), desc="  Noise reduction", unit="chunk"):
        pos = i * chunk_size
        read_start = max(0, pos - overlap_size)
        read_end = min(total, pos + chunk_size + overlap_size)
        chunk = samples[read_start:read_end]

        reduced = nr.reduce_noise(
            y=chunk, sr=sr, y_noise=noise_profile,
            stationary=False, prop_decrease=0.75,
        )

        write_end = min(total, pos + chunk_size)
        offset = pos - read_start
        output[pos:write_end] = reduced[offset : offset + (write_end - pos)]

    return output


def run(input_path: str, output_dir: str) -> str:
    """
    Run Stage 1.

    Args:
        input_path: Path to the source audio file (e.g. input/call.mp3).
        output_dir:  Directory where the clean WAV will be written.

    Returns:
        Path to the cleaned WAV file.
    """
    print(f"\n[Stage 1] Loading audio: {input_path}")
    audio = AudioSegment.from_file(input_path)

    # Standardise: mono, 16 kHz
    audio = audio.set_channels(1).set_frame_rate(16_000)
    print(f"[Stage 1] Duration: {len(audio) / 1000:.1f}s  |  Sample rate: {audio.frame_rate} Hz")

    # --- Noise reduction (chunked for large files) ---
    print("[Stage 1] Applying noise reduction...")
    samples, sr = _to_numpy(audio)
    del audio  # free pydub copy before allocating numpy output
    gc.collect()

    reduced = _chunked_reduce_noise(samples, sr)
    del samples
    gc.collect()

    audio = _from_numpy(reduced, sr)
    del reduced
    gc.collect()

    # --- Loudness normalization ---
    print("[Stage 1] Normalizing loudness...")
    audio = normalize(audio)

    # --- Export ---
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_clean.wav")
    audio.export(output_path, format="wav")
    print(f"[Stage 1] Clean audio saved to: {output_path}")

    return output_path
