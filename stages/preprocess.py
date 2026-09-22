"""
Stage 1 — Audio Pre-processing

Steps:
  1. Decode the input (MP3/M4A/WAV or any ffmpeg-supported format) straight to a
     mono, 16 kHz, 16-bit PCM WAV via a single streaming ffmpeg pass
  2. Read that WAV with soundfile — already at target rate, so the read is cheap
  3. Apply spectral noise reduction (noisereduce) in overlapping chunks, writing
     each denoised chunk straight to the output WAV
  4. Normalize overall loudness in a second streaming pass over that WAV
  5. Leave the clean WAV on disk for downstream stages

Audio is never held in memory in full: it moves source file -> decoded WAV ->
output WAV, a chunk at a time. Peak memory is therefore flat with respect to
recording length, and is dominated by library imports rather than by audio.
"""

import gc
import math
import os
import shutil
import subprocess
import numpy as np
import noisereduce as nr
import soundfile as sf
from tqdm import tqdm

# Mono 16 kHz is what pyannote (Stage 2) and Whisper (Stage 3) both expect.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

# Matches the default of pydub.effects.normalize(), which this module used to call.
HEADROOM_DB = 0.1
INT16_PEAK = 2 ** 15 - 1          # 32767 — largest magnitude we ever write
INT16_FULL_SCALE = 2.0 ** 15      # 32768 — pydub's max_possible_amplitude


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


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and scale to int16 — the conversion _from_numpy() used."""
    return (np.clip(samples, -1.0, 1.0) * INT16_PEAK).astype(np.int16)


def _block_peak(block: np.ndarray) -> int:
    """Largest absolute sample in an int16 block, computed without overflowing."""
    if not len(block):
        return 0
    return int(np.abs(block.astype(np.int32)).max())


def _stream_reduce_noise(reader: sf.SoundFile, sr: int, writer: sf.SoundFile) -> int:
    """Apply noisereduce in overlapping chunks, streaming from `reader` to `writer`.

    noisereduce internally computes a full STFT of its input, which for a 90-minute
    file creates a ~1.4 GB temporary matrix. This function processes 60-second slices
    instead, keeping peak RAM under ~350 MB regardless of recording length.

    Each chunk is extended by OVERLAP_SEC on each side so noisereduce has context at
    the boundaries — only the core portion is written to the output. For recordings
    shorter than CHUNK_SEC the whole file is processed in a single call.

    Chunk boundaries, overlap trimming and the noisereduce call itself are
    unchanged from the earlier in-memory version — only where the audio lives
    changed. Chunks are read from disk on demand instead of sliced out of one
    full-length input array, and each denoised chunk is converted to int16 and
    appended to `writer` instead of being accumulated into a full-length
    np.empty(total) array. Those two arrays were Stage 1's largest allocations
    (595 MB each on a 2h43m recording); neither exists now.

    Returns the peak absolute int16 sample written, which the caller needs in
    order to normalize without a second pass over memory.
    """
    CHUNK_SEC = 60
    OVERLAP_SEC = 0.5

    chunk_size = int(sr * CHUNK_SEC)
    overlap_size = int(sr * OVERLAP_SEC)
    total = reader.frames

    reader.seek(0)
    noise_profile = reader.read(min(int(sr * 0.5), total), dtype="float32")

    if total <= chunk_size:
        reader.seek(0)
        block = _to_int16(nr.reduce_noise(
            y=reader.read(total, dtype="float32"), sr=sr, y_noise=noise_profile,
            stationary=False, prop_decrease=0.75,
        ))
        writer.write(block)
        return _block_peak(block)

    n_chunks = math.ceil(total / chunk_size)
    peak = 0

    for i in tqdm(range(n_chunks), desc="  Noise reduction", unit="chunk"):
        pos = i * chunk_size
        read_start = max(0, pos - overlap_size)
        read_end = min(total, pos + chunk_size + overlap_size)
        reader.seek(read_start)
        chunk = reader.read(read_end - read_start, dtype="float32")

        reduced = nr.reduce_noise(
            y=chunk, sr=sr, y_noise=noise_profile,
            stationary=False, prop_decrease=0.75,
        )

        write_end = min(total, pos + chunk_size)
        offset = pos - read_start
        block = _to_int16(reduced[offset : offset + (write_end - pos)])
        writer.write(block)
        peak = max(peak, _block_peak(block))

    return peak


def _normalize_in_place(path: str, peak: int, block_frames: int = 1 << 20) -> None:
    """Scale the WAV at `path` so its loudest sample sits HEADROOM_DB below full scale.

    Reproduces pydub.effects.normalize(), which derived the gain from the whole
    file's peak and applied it to an in-memory AudioSegment. Here the peak is
    accumulated while the file is written, and the gain is applied by rewriting
    the file a block at a time, so memory stays flat.
    """
    if peak <= 0:
        return  # silent track — pydub returns it unchanged rather than dividing by 0

    gain = (INT16_FULL_SCALE * (10 ** (-HEADROOM_DB / 20.0))) / peak
    if abs(gain - 1.0) < 1e-9:
        return

    with sf.SoundFile(path, mode="r+") as fh:
        pos = 0
        while pos < fh.frames:
            fh.seek(pos)
            block = fh.read(block_frames, dtype="int16")
            if not len(block):
                break
            scaled = np.clip(np.rint(block * gain), -INT16_FULL_SCALE, INT16_PEAK)
            fh.seek(pos)
            fh.write(scaled.astype(np.int16))
            pos += len(block)


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

        # --- Noise reduction, streamed from the decoded WAV into the output WAV ---
        with sf.SoundFile(decoded_path, mode="r") as reader:
            sr = reader.samplerate
            print(f"[Stage 1] Duration: {reader.frames / sr:.1f}s  |  Sample rate: {sr} Hz")
            print("[Stage 1] Applying noise reduction...")
            with sf.SoundFile(
                output_path, mode="w", samplerate=sr,
                channels=TARGET_CHANNELS, subtype="PCM_16",
            ) as writer:
                peak = _stream_reduce_noise(reader, sr, writer)
        gc.collect()

        # --- Loudness normalization (second streaming pass over the output) ---
        print("[Stage 1] Normalizing loudness...")
        _normalize_in_place(output_path, peak)
        print(f"[Stage 1] Clean audio saved to: {output_path}")
    finally:
        if os.path.exists(decoded_path):
            try:
                os.remove(decoded_path)
            except OSError:
                pass  # a leftover temp file is harmless; do not mask a real error

    return output_path
