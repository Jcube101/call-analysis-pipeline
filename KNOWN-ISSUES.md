# Known Issues — Call Analysis Pipeline

Open problems that are understood but deliberately not fixed yet. Each entry
records the root cause and the measurement behind it so a future session does
not have to re-diagnose from scratch.

For solved problems and the reasoning behind existing design choices, see
[LEARNINGS.md](LEARNINGS.md). For planned feature work, see [ROADMAP.md](ROADMAP.md).

---

## 1. `diarize.py` loads the entire clean WAV as float64

**Status:** open — now the largest audio buffer left in the pipeline, and the
binding constraint on recording length.

**Where:** [`stages/diarize.py`](stages/diarize.py) lines 204-208:

```python
data, sample_rate = sf.read(clean_wav_path)      # float64 by default
...
waveform = torch.from_numpy(data).float()        # full-length float32 copy
```

and again inside `_reidentify_speakers()`:

```python
audio_np = waveform[0].numpy().astype(np.float32)  # another full-length copy
```

**Problem:** three full-length arrays are live at once. On the 2h43m reference
recording (156,094,464 frames) that is 1.16 GiB of float64 plus two 595 MB
float32 copies — roughly **2.5 GB**, versus the ~600 MB that Stage 3 used to
cost.

**Measured:** an end-to-end run under 2.1 GB of free RAM clears Stage 1 (116 s,
peak 1748 MB) and then dies here:

```
File "stages/diarize.py", line 204, in run
  data, sample_rate = sf.read(clean_wav_path)
numpy.core._exceptions._ArrayMemoryError:
  Unable to allocate 1.16 GiB for an array with shape (156094464,) and data type float64
```

**Cheap partial fix:** the `float64` is gratuitous — it is just `sf.read`'s
default dtype on a 16-bit file. Passing `dtype="float32"` halves that allocation
to 595 MB *and* removes a full-length copy, because `torch.from_numpy(data).float()`
then has nothing to convert. That alone should take the three buffers down to
two and the total from ~2.5 GB to ~1.2 GB.

**Constraint on a full fix:** unlike Stages 1 and 3, this is not
straightforwardly streamable. The waveform is handed to pyannote as an in-memory
`{"waveform": ..., "sample_rate": ...}` dict *on purpose*, to avoid a
`torchcodec` dependency (see CLAUDE.md, "Audio input to pyannote"). pyannote
needs the whole signal, and so does the MFCC re-identification pass.

---

## 2. Stage 2 exhausts VRAM on multi-hour recordings

**Status:** open — separate from the RAM issue above, and hit first on a 4 GB GPU.

Diarizing the 2h43m reference recording on a GTX 1650 fails with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 312.00 MiB.
GPU 0 has a total capacity of 4.00 GiB of which 2.02 GiB is free.
```

2 GiB reported free but a 312 MiB allocation failing points at fragmentation;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the first thing to try.
Not investigated further — noted here so it is not re-diagnosed as a RAM problem.

---

## 3. `pydub` is now an unused dependency

**Status:** open — harmless, but it is dead weight.

Nothing in the pipeline imports `pydub` any more: Stage 1 stopped using it when
decoding moved to a direct ffmpeg pass, and Stage 3 stopped when the clean WAV
moved behind `_WavReader`. It is still listed in `requirements.txt` and still
named in CLAUDE.md's description of `stages/preprocess.py`. Removing it means
one less native-audio dependency to install; the only reason to keep it is that
`audioop`, which pydub wraps, is removed in Python 3.13 anyway.

---

## Resolved

### Stage 3 loaded the entire clean WAV into memory (fixed)

[`stages/transcribe.py`](stages/transcribe.py) used to do
`audio = AudioSegment.from_wav(clean_wav_path)` and slice that object per
diarization segment, holding the full PCM payload plus a read copy for the whole
stage. Replaced with `_WavReader`, which keeps the file open via
`soundfile.SoundFile` and reads only the frames each segment needs.

Measured on `output/03_Lunch_with_Rachita_clean.wav` (2h43m, 298 MB on disk,
2490 segments read):

| | before | after |
|---|---|---|
| audio buffer alone, peak private | +598 MB | **+25 MB** |
| audio buffer alone, peak working set | 597 MB | **57 MB** |
| whole stage with Whisper loaded, peak private | 3754 MB | **3455 MB** |
| whole stage with Whisper loaded, peak working set | 1553 MB | **1043 MB** |

The +598 MB matches the "roughly 2x file size" estimate this entry originally
carried. Stage 3's audio cost is now flat with respect to recording length
instead of growing at ~230 MB per hour of audio.

Under the memory balloon, that 300 MB decides the run at 3.0 GB free:

| Free RAM | before | after |
|---|---|---|
| 3.5 GB | completes, 309 s, peak 3757 MB | completes, 332 s, peak 3457 MB |
| **3.0 GB** | **fails** — `mkl_malloc: failed to allocate memory` | **completes, 330 s, peak 3457 MB** |
| 2.1 GB | fails at `WhisperModel()` load | fails at `WhisperModel()` load |

**2.1 GB is below Stage 3's floor in either version**, and always was: both fail
inside `ctranslate2.models.Whisper(...)` before any audio is read. The `medium`
model alone needs roughly 3 GB of host memory, so the audio buffer was never the
binding constraint at that level — it only became one between about 3.0 and
3.5 GB, and on recordings longer than this one it would bind sooner.

**Output is unchanged.** Slice arithmetic mirrors pydub's `__getitem__` exactly —
millisecond bounds clamped to the file duration, truncated to whole frames via
`int(ms * rate / 1000.0)`, short tail reads zero-padded — and `sf.read(dtype="float32")`
divides a PCM_16 sample by 2**15 just as `_audio_segment_to_numpy()` did. Verified
bit-identical (frame range, 500 ms gate decision, and sample values) across all
3102 windows of the 2h43m file plus 89 synthetic windows each on four other WAVs,
including zero-length, sub-500 ms, EOF-overrun and awkward-frame-count cases.

**Do not** treat small transcript diffs between runs as a regression here.
faster-whisper with `int8_float16` on CUDA is not deterministic: three
consecutive runs of the *unmodified* old code on the same 1.4 s window produced
three different texts ("Okay. I'm through." / "I'm through." / "I'm through.
I'm through. But.", confidence 0.27-0.36). Only low-confidence segments are
affected.

**One deliberate behaviour difference**, confined to `--skip-preprocess` with a
WAV that did not come from Stage 1: if the input is not 16 kHz, `_WavReader`
resamples with `librosa` where pydub used `audioop.ratecv`. librosa's
band-limited sinc resampler is the better one but is not bit-identical, and
segment lengths can differ by a frame. Anything produced by Stage 1 is already
mono 16 kHz and never takes this path.

### Stage 1 peak memory (fixed — recorded for reference)

Stage 1 used to raise `MemoryError` on 1 hr+ recordings. Two rounds of work
removed every full-length allocation:

1. `AudioSegment.from_file()` buffered the entire decoded PCM stream and copied
   it several times *before* any downsampling. Replaced with a single streaming
   ffmpeg pass that decodes straight to mono/16 kHz/`pcm_s16le` on disk.
2. The peak then relocated to two full-length float32 arrays — the `sf.read()`
   input array and the `np.empty(total)` accumulator in the noise-reduction
   helper — plus `_from_numpy()`'s several full-length temporaries. Replaced by
   streaming chunk-by-chunk from an open `sf.SoundFile` reader to an open
   `sf.SoundFile` writer, with loudness normalization applied as a second
   block-wise pass over the output file.

Measured on `input/03_Lunch_with_Rachita.m4a` (2h43m, 48 kHz mono AAC):

| Free RAM | Original | Decode fix only | Fully streaming |
|---|---|---|---|
| 5.2 GB | completes, 386 s, peak 4077 MB | completes, 210 s, peak 3513 MB | completes, ~124 s, peak 1587 MB |
| 3.0 GB | not measured | `MemoryError` after 187 s | completes |
| 2.1 GB | `MemoryError` after 30 s | `MemoryError` | **completes, peak 1587 MB / 576 MB working set** |
| 1.2 GB | — | — | fails importing torch's CUDA DLLs (WinError 1455), not on audio |

Peak memory is now flat with respect to recording length and is dominated by
library imports (~1390 MB of numpy/scipy/torch/noisereduce) rather than by
audio. Output is unchanged to within 1 LSB of 32767 (≈ −90 dBFS), the residue
of `np.rint` rounding replacing audioop's truncation.

**Do not** reintroduce `pydub` into `stages/preprocess.py`. Stage 1 no longer
imports it; `normalize()` is reproduced in `_normalize_in_place()`, which
matches `pydub.effects.normalize`'s 0.1 dB headroom default exactly.

**Reproducing the pressure test:** allocate and touch 128 MB blocks until
`GlobalMemoryStatusEx().ullAvailPageFile` reaches the target, then run the
pipeline as a subprocess under that balloon. Note this holds pages resident, so
it is a harsher condition than natural memory contention.
