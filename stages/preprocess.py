"""
Stage 1 — Audio Pre-processing

Steps:
  1. Decode the input (MP3/M4A/WAV or any ffmpeg-supported format) straight to a
     mono, 16 kHz, 16-bit PCM WAV via a single streaming ffmpeg pass
  2. Read that WAV with soundfile — already at target rate, so the read is cheap
  3. Apply spectral noise reduction (noisereduce) — chunked for large files
  4. Normalize overall loudness
  5. Export as a clean WAV file for downstream stages
"""

import gc
import math
import os
import shutil
import subprocess
import numpy as np
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
from tqdm import tqdm

# Mono 16 kHz is what pyannote (Stage 2) and Whisper (Stage 3) both expect.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


def _decode_to_wav(input_path: str, output_path: str) -> None:
    """Decode any ffmpeg-supported source to a mono, 16 kHz, 16-bit PCM WAV on disk.

    Runs as a single streaming ffmpeg pass: the audio never exists as a Python
    object, and the downsample to 16 kHz happens inside ffmpeg rather than after
    the fact. This replaces AudioSegment.from_file(), which buffered the whole
    decoded stream in memory and copied it several times before any downsampling
    could occur - peaking at ~2.7 GB on a 2h43m recording and raising MemoryError
    on machines without that much headroom.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on your PATH.\n"
            "  macOS:          brew install ffmpeg\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  Windows:        https://ffmpeg.org/download.html (add to PATH)"
        )

    command = [
        "ffmpeg",
        "-nostdin",             # never block waiting on stdin
        "-hide_banner",
        "-loglevel", "error",
        "-y",                   # overwrite a stale temp file from a crashed run
        "-i", input_path,
        "-vn",                  # drop cover art / video streams
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        "-f", "wav",
        output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {input_path} "
            f"(exit code {result.returncode}):\n{result.stderr.strip()}"
        )


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
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_clean.wav")
    # Intermediate decode target. The "_decoded.tmp" suffix keeps it clear of the
    # "*_clean.wav" glob that api.py uses to serve the download endpoint.
    decoded_path = os.path.join(output_dir, f"{base_name}_decoded.tmp.wav")

    try:
        _decode_to_wav(input_path, decoded_path)

        # Safe to read whole: already mono at 16 kHz, so float32 costs ~115 MB
        # per hour of audio regardless of the source rate or channel count.
        samples, sr = sf.read(decoded_path, dtype="float32")
        print(f"[Stage 1] Duration: {len(samples) / sr:.1f}s  |  Sample rate: {sr} Hz")

        # --- Noise reduction (chunked for large files) ---
        print("[Stage 1] Applying noise reduction...")
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
        audio.export(output_path, format="wav")
        print(f"[Stage 1] Clean audio saved to: {output_path}")
    finally:
        if os.path.exists(decoded_path):
            try:
                os.remove(decoded_path)
            except OSError:
                pass  # a leftover temp file is harmless; do not mask a real error

    return output_path
