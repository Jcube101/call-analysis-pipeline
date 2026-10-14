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
import random
import sys
import time
import warnings
from typing import Callable, Optional

import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="torchcodec is not installed", category=UserWarning)
    from pyannote.audio import Pipeline

from tqdm import tqdm

from config import settings

# Upper bound on embeddings handed to agglomerative clustering. See the comment
# at the assignment in run() for why this is set and why 1000.
MAX_CLUSTERING_EMBEDDINGS = 1000

# Seed for that subsample. Above the cap, pyannote picks which embeddings to
# cluster with an unseeded random.shuffle(), so four runs of a 10-minute
# excerpt gave 272, 272, 283 and 270 segments. The value is arbitrary; fixing
# it makes Stage 2 repeatable.
CLUSTERING_SUBSAMPLE_SEED = 0

# Stage 2's peak commit, fitted from two measured points -- the 2 minute file
# (5404 MB) and the 2h43m file (5670 MB), both with the clustering cap and with
# audio read from disk rather than held resident.
#
# The same fit before the disk-reading change gave 5396 MB + 443 MB/hour, which
# is why these numbers are recalibrated rather than inherited: removing the
# resident waveform and the MFCC pass's copy took the duration-dependent term
# from ~443 to ~99 MB/hour. Peak is dominated by the embedding model and the
# CUDA context, which do not scale with recording length at all.
#
# The check below uses ullAvailPageFile, not ullAvailPhys. A MemoryError on
# Windows is a failed *commit*, and the two are far apart here -- this machine
# routinely sits at ~1 GB AvailPhys with ~8 GB of commit headroom, and the
# three verification runs of the 2h43m file started at 0.81, 1.24 and 1.40 GB
# AvailPhys and all completed. Gating on AvailPhys would reject runs that work.
_STAGE2_BASE_MB = 5400
_STAGE2_PER_HOUR_MB = 100


def _available_commit_mb() -> Optional[float]:
    """Commit headroom in MB, or None where that cannot be queried (non-Windows)."""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPageFile / 2 ** 20
    except Exception:
        return None


def _process_commit_mb() -> Optional[float]:
    """This process's committed private bytes in MB, or None if unavailable."""
    try:
        import ctypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        get_current = ctypes.WinDLL("kernel32").GetCurrentProcess
        get_current.restype = ctypes.c_void_p
        # argtypes are required here: without them ctypes passes the process
        # handle as a 32-bit int and the call fails on 64-bit Python. They are
        # set on a private WinDLL handle rather than ctypes.windll.kernel32,
        # which caches one function object per process -- annotating the shared
        # one breaks any other code that calls it with its own struct type.
        kernel32 = ctypes.WinDLL("kernel32")
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_info.restype = ctypes.c_int
        if not get_info(get_current(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PrivateUsage / 2 ** 20
    except Exception:
        return None


def _check_memory_headroom(clean_wav_path: str) -> None:
    """Fail before Stage 2 starts if this run cannot fit in available memory.

    Without this the shortfall surfaces ten minutes into a multi-stage run as a
    MemoryError from deep inside scipy or pyannote, after Stage 1 has already
    done its work.

    _STAGE2_BASE_MB is a *peak total* for the process, and by the time this
    runs the process has already committed a good part of it (torch, the CUDA
    context, whatever Stage 1 left behind). Only the difference still has to
    come out of system headroom -- comparing the whole peak against headroom
    rejected the 2h43m file that completes three times out of three.
    """
    available = _available_commit_mb()
    if available is None:
        return  # cannot measure; do not block the run on a guess

    try:
        import soundfile as sf
        info = sf.info(clean_wav_path)
        hours = (info.frames / info.samplerate) / 3600.0
    except Exception:
        return  # unreadable file is Stage 2's problem to report, not this check's

    peak = _STAGE2_BASE_MB + _STAGE2_PER_HOUR_MB * hours
    already = _process_commit_mb() or 0.0
    needed = max(0.0, peak - already)
    if available < needed:
        raise RuntimeError(
            "\n".join([
                f"not enough memory for Stage 2. This {hours * 60:.0f} minute "
                f"recording needs roughly {needed / 1024:.1f} GB more commit headroom "
                f"and only {available / 1024:.1f} GB is available.",
                "  - close other applications and retry, or",
                "  - raise the Windows page file size, or",
                "  - split the recording into shorter files.",
                f"Estimated peak {peak / 1024:.1f} GB (a measured "
                f"{_STAGE2_BASE_MB / 1024:.1f} GB base for torch, CUDA and the "
                f"pyannote models, plus {_STAGE2_PER_HOUR_MB} MB per hour of audio), "
                f"of which {already / 1024:.1f} GB is already committed.",
            ])
        )


# Map raw pyannote labels → human-friendly labels used in the transcript
def _label_map(raw_labels: list[str]) -> dict[str, str]:
    """SPEAKER_00 → Speaker A, SPEAKER_01 → Speaker B, etc."""
    sorted_labels = sorted(set(raw_labels))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {label: f"Speaker {letters[i]}" for i, label in enumerate(sorted_labels)}


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def vram_status() -> Optional[str]:
    """One-line VRAM summary, or None without CUDA.

    mem_get_info() is device-wide, so it also sees memory this process holds
    outside torch's allocator -- ctranslate2 (Whisper), the CUDA context -- and
    other processes' use. memory_allocated() is live torch tensors only, and
    memory_reserved() adds blocks torch's caching allocator is holding on to.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    except Exception:
        return None
    gb = 2 ** 30
    return (
        f"free {free / gb:.2f} of {total / gb:.2f} GB, "
        f"torch allocated {allocated / gb:.2f} GB, reserved {reserved / gb:.2f} GB"
    )


class _PhaseMonitor:
    """pyannote hook that makes Stage 2 observable in the server log.

    pyannote's own ProgressHook is no use here. It draws rich progress bars,
    and rich treats a non-TTY stdout as non-interactive: under uvicorn with
    stdout redirected to a file it printed nothing while the pipeline ran,
    then one static snapshot of finished bars when it returned. This prints
    plain lines as it goes -- one per phase change -- and a tqdm bar, like
    Stages 1 and 3, while a phase reports completed/total.

    pyannote 3.4.0 calls the hook in this order: "segmentation" (progress,
    then one call carrying the artifact), "speaker_counting" (a single call,
    made after counting has finished), "embeddings" (progress, then the
    artifact) and "discrete_diarization". Clustering never calls the hook. It
    runs between the embeddings artifact and "discrete_diarization", so that
    gap is reported as a phase of its own.

    Artifacts are never stored -- the hook only reads step names and counts.

    `progress`, if given, is called as progress(phase, percent, message) on
    every phase change and, while a phase reports counts, at most every 5%
    and 10 s. `percent` is the rounded percent within the current phase, 0
    when a counted phase (segmentation, embeddings) begins, and None for
    phases that report no counts. A phase is finished only when the next one begins. Errors the
    callback raises are swallowed; it must not be able to fail diarization.
    """

    _PROGRESS_STEP = 0.05     # a progress callback every 5% ...
    _PROGRESS_MIN_GAP = 10.0  # ... but no more often than every 10 s
    _UNITS = {"segmentation": "chunks", "embeddings": "batches"}
    _BAR_UNITS = {"segmentation": "chunk", "embeddings": "batch"}
    # pyannote reports every batch, and tqdm redraws up to 10x a second. A file
    # can't overwrite a line, so under uvicorn with output piped every redraw is
    # appended to the log; redraw at most every 5 s there.
    _BAR_MININTERVAL_NON_TTY = 5.0

    def __init__(self, progress: Optional[Callable[[str, Optional[int], str], None]] = None) -> None:
        self._callback = progress
        self.phase: Optional[str] = None
        self._phase_start = 0.0
        self._last_line = 0.0
        self._last_fraction = 0.0
        self._pending_start: Optional[float] = None
        self._created = time.monotonic()
        self._bar: Optional[tqdm] = None
        self.durations: list[tuple[str, float]] = []

    def begin(self, phase: str, started: Optional[float] = None, announce: bool = True) -> None:
        """Start a phase, closing the current one. `started` backdates it (monotonic seconds)."""
        now = time.monotonic()
        self._close(now)
        self.phase = phase
        self._phase_start = started if started is not None else now
        self._last_line = now
        self._last_fraction = 0.0
        if announce:
            print(f"[Stage 2] Phase: {phase}", flush=True)
        self._notify(phase, 0 if phase in self._UNITS else None, f"Phase: {phase}")
        vram = vram_status()
        if vram:
            print(f"[Stage 2] VRAM at {phase} start: {vram}", flush=True)

    def end(self) -> None:
        self._close(time.monotonic())

    def _notify(self, phase: str, percent: Optional[int], message: str) -> None:
        if self._callback is None:
            return
        try:
            self._callback(phase, percent, message)
        except Exception as e:
            print(f"[Stage 2] Progress callback failed (ignored): {e}", flush=True)

    def _close(self, now: float) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if self.phase is not None:
            self.durations.append((self.phase, now - self._phase_start))
        self.phase = None

    def summary(self) -> str:
        """Wall-clock per phase, e.g. "segmentation 4:12, embeddings 18:40, ..."."""
        self.end()
        parts = [f"{phase} {_fmt_elapsed(secs)}" for phase, secs in self.durations]
        parts.append(f"total {_fmt_elapsed(time.monotonic() - self._created)}")
        return ", ".join(parts)

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is not None:
            if step_name == "embeddings" and self.phase != "embeddings":
                # Embedding extraction starts right after speaker counting
                # ends; backdate to then so its setup is counted as embeddings.
                self.begin("embeddings", started=self._pending_start)
            self._progress(step_name, completed, total)
            return

        # A step has finished and handed over its artifact.
        if step_name == "segmentation":
            self.begin("speaker counting")
        elif step_name == "speaker_counting":
            # Embeddings announce themselves with their first progress call.
            # With a single speaker they are skipped, along with clustering.
            self.end()
            self._pending_start = time.monotonic()
        elif step_name == "embeddings":
            self.begin("clustering")
        elif step_name == "discrete_diarization":
            self.end()

    def _update_bar(self, step_name: str, completed: int, total: int) -> None:
        if self._bar is None:
            tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
            self._bar = tqdm(
                total=total,
                desc=f"  {step_name.capitalize()}",
                unit=self._BAR_UNITS.get(step_name, "step"),
                mininterval=0.1 if tty else self._BAR_MININTERVAL_NON_TTY,
            )
        self._bar.update(completed - self._bar.n)

    def _progress(self, step_name: str, completed: int, total: Optional[int]) -> None:
        if not total:
            return
        completed = min(completed, total)  # Inference overshoots on its last batch
        try:
            self._update_bar(step_name, completed, total)
        except Exception:
            pass  # the bar is cosmetic
        fraction = completed / total
        now = time.monotonic()
        if (
            fraction >= 1.0  # the next phase line follows immediately
            or fraction - self._last_fraction < self._PROGRESS_STEP
            or now - self._last_line < self._PROGRESS_MIN_GAP
        ):
            return
        elapsed = now - self._phase_start
        left = elapsed * (1 - fraction) / fraction if fraction > 0 else 0.0
        unit = self._UNITS.get(step_name, "steps")
        percent = round(fraction * 100)
        line = (
            f"{step_name} {percent}% ({completed}/{total} {unit}), "
            f"{_fmt_elapsed(elapsed)} elapsed, ~{_fmt_elapsed(left)} left"
        )
        # Not printed: the bar shows this, and a line printed into a bar that
        # is still open would land mid-line in a piped log.
        self._notify(self.phase or step_name, percent, line)
        self._last_line = now
        self._last_fraction = fraction


def _reidentify_speakers(
    segments: list[dict],
    clean_wav_path: str,
    sample_rate: int,
    num_speakers: int,
) -> list[dict]:
    """
    Re-assign speaker labels using MFCC features + KMeans clustering.

    pyannote processes long audio in chunks and can flip speaker labels between
    chunks (e.g. the same physical voice becomes SPEAKER_00 in the first half
    and SPEAKER_01 in the second half). This function computes MFCC features
    per segment, clusters them globally, then reassigns labels in
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

    import soundfile as sf

    MIN_SEG_SECONDS = 0.5
    N_MFCC = 20

    # Each segment is read from disk on demand rather than sliced out of a
    # resident copy of the whole recording. This used to be
    # `waveform[0].numpy().astype(np.float32)`, which copied unconditionally
    # even though the array was already float32 -- 624 MB on the 2h43m file, on
    # top of the 624 MB waveform it copied from. Same approach as Stage 3's
    # _WavReader; see stages/transcribe.py.
    embeddings: list[np.ndarray] = []
    long_indices: list[int] = []
    n_short = 0
    n_failed = 0
    first_error: str = ""

    print(f"[Stage 2] Extracting MFCC features ({len(segments)} segments, >={MIN_SEG_SECONDS}s only)...")

    with sf.SoundFile(clean_wav_path, mode="r") as handle:
        total_frames = handle.frames
        for i, seg in enumerate(tqdm(segments, desc="  Features", unit="seg")):
            duration = seg["end"] - seg["start"]
            if duration < MIN_SEG_SECONDS:
                n_short += 1
                continue

            # Same bounds as the old array slice: truncate to whole frames and
            # clamp to the file, so a segment running past the end reads short
            # exactly as numpy slicing did rather than raising.
            start_sample = min(int(seg["start"] * sample_rate), total_frames)
            end_sample = min(int(seg["end"] * sample_rate), total_frames)
            n_frames = max(0, end_sample - start_sample)

            try:
                handle.seek(start_sample)
                seg_audio = handle.read(n_frames, dtype="float32")
                if seg_audio.ndim > 1:
                    seg_audio = seg_audio.mean(axis=1)
                mfcc = librosa.feature.mfcc(y=seg_audio, sr=sample_rate, n_mfcc=N_MFCC)
                feat = np.mean(mfcc, axis=1)
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

    # Cluster globally
    emb_matrix = normalize(np.array(embeddings))
    kmeans = KMeans(
        n_clusters=num_speakers,
        n_init=10,
        random_state=42,
    )
    cluster_ids = kmeans.fit_predict(emb_matrix)

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
    progress: Optional[Callable[[str, Optional[int], str], None]] = None,
) -> list[dict]:
    """
    Run Stage 2.

    Args:
        clean_wav_path: Path to the Stage-1 cleaned WAV.
        output_dir:      Directory for any intermediate outputs.
        num_speakers:    Exact speaker count, or None for auto-detection.
        progress:        Optional callback(phase, percent_or_None, message),
                         called on each phase change and throttled progress
                         line. See _PhaseMonitor. Errors in it are ignored.

    Returns:
        List of segment dicts:
          {"start": float, "end": float, "speaker": "Speaker A", "label": "SPEAKER_00"}
    """
    settings.validate_for_diarization()
    _check_memory_headroom(clean_wav_path)

    monitor = _PhaseMonitor(progress)
    monitor.begin("model load", announce=False)

    print(f"\n[Stage 2] Loading diarization pipeline (pyannote/speaker-diarization-3.1)...")
    import huggingface_hub
    huggingface_hub.login(token=settings.huggingface_token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    # Cap the embeddings fed to agglomerative clustering.
    #
    # pyannote's BaseClustering defaults this to 1000 and subsamples to it in
    # filter_embeddings(), clustering that subset and labelling the rest via
    # assign_embeddings(). The pretrained speaker-diarization-3.1 config ships
    # max_num_embeddings=inf, which disables the cap, so scipy's linkage()
    # receives every embedding and allocates an O(N^2) condensed distance
    # matrix -- 16760 embeddings on a 2h43m file, 1.05 GB, roughly 2.1 GB with
    # the working copy centroid linkage needs. That allocation, not the
    # waveform, is what exhausted host RAM on long recordings.
    #
    # 1000 is pyannote's own default. Measured on the 2h43m reference file with
    # num_speakers=2: peak 7656 -> 5671 MB, and the output is better, not just
    # cheaper -- uncapped split the two speakers 1278s/6213s, capped gives
    # 3626s/3864s. Caps of 1000 and 5000 agree on 99.96% of frames, so the
    # smaller one costs nothing.
    if hasattr(pipeline, "clustering") and hasattr(pipeline.clustering, "max_num_embeddings"):
        pipeline.clustering.max_num_embeddings = MAX_CLUSTERING_EMBEDDINGS

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

    sample_rate = sf.info(clean_wav_path).samplerate

    # The file path is handed to pyannote rather than a pre-loaded
    # {"waveform": ..., "sample_rate": ...} dict.
    #
    # The dict form was chosen to avoid a torchcodec dependency, and that
    # rationale no longer holds: pyannote 3.4.0 reads audio through torchaudio,
    # which selects the soundfile backend here (torchaudio.list_audio_backends()
    # returns ['soundfile'] and torchcodec is not installed). Passing a path
    # lets Audio.crop() seek-and-read each window it needs instead of slicing a
    # resident copy of the whole recording -- 300 ten-second crops from the
    # 2h43m file cost +5 MB this way against +597 MB for the dict.
    #
    # Output is unchanged: the full pipeline produced byte-identical segments
    # both ways on the 2 minute reference file (37 segments, boundaries equal to
    # 4 decimal places) and the same 4735 segments on the 2h43m file.
    #
    # Note this does not make Stage 2 streaming. Inference.__call__ still loads
    # the whole signal once for the segmentation pass; only the embedding crops
    # avoid residency.
    #
    # Segmentation is started here rather than at its first hook call: the
    # whole-file audio load happens before that call and belongs to it.
    monitor.begin("segmentation")
    # Seed the module-level `random` that pyannote's clustering subsample
    # uses, then restore its state so nothing else in the process is affected.
    random_state = random.getstate()
    random.seed(CLUSTERING_SUBSAMPLE_SEED)
    try:
        diarization = pipeline(clean_wav_path, hook=monitor, **diarize_kwargs)
    finally:
        random.setstate(random_state)
    monitor.end()

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

    # Re-identify speakers globally. This reads its own audio from the clean WAV
    # a segment at a time, so it no longer has to run before a waveform is freed.
    num_spk = num_speakers or len(set(s["label"] for s in raw_segments))
    monitor.begin("re-identification", announce=False)
    try:
        raw_segments = _reidentify_speakers(raw_segments, clean_wav_path, sample_rate, num_spk)
    except Exception as e:
        print(f"[Stage 2] Speaker re-identification failed ({e}) — using pyannote labels as-is")
    monitor.end()

    gc.collect()

    # Build friendly labels
    label_map = _label_map([s["label"] for s in raw_segments])
    for seg in raw_segments:
        seg["speaker"] = label_map[seg["label"]]

    print(
        f"[Stage 2] Found {len(set(s['speaker'] for s in raw_segments))} speaker(s), "
        f"{len(raw_segments)} segments"
    )
    print(f"[Stage 2] Timing: {monitor.summary()}", flush=True)

    # Release the pyannote model from VRAM before Stage 3 loads Whisper.
    # Without this, both models sit in VRAM simultaneously (~3.7/4.0 GB on GTX 1650).
    # Moving to CPU + clearing the cache drops Stage 3 VRAM usage by ~1-1.5 GB.
    try:
        import torch  # bound locally: the device check above may have skipped it
        pipeline.to(torch.device("cpu"))
        del pipeline, diarization, annotation
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass  # non-critical — pipeline will be GC'd eventually regardless

    return raw_segments
