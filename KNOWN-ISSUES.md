# Known Issues — Call Analysis Pipeline

Open problems that are understood but deliberately not fixed yet. Each entry
records the root cause and the measurement behind it so a future session does
not have to re-diagnose from scratch.

For solved problems and the reasoning behind existing design choices, see
[LEARNINGS.md](LEARNINGS.md). For planned feature work, see [ROADMAP.md](ROADMAP.md).

---

## 1. `transcribe.py` loads the entire clean WAV into memory

**Status:** open — now the only remaining full-file buffer in the pipeline.

**Where:** [`stages/transcribe.py`](stages/transcribe.py) line 253:

```python
audio = AudioSegment.from_wav(clean_wav_path)
```

**Problem:** Stage 3 reads the whole Stage-1 output into a single in-memory
`AudioSegment` before slicing it per diarization segment. This is the same
in-memory-buffer pattern that caused the Stage 1 `MemoryError`, just one stage
later and at a smaller constant factor.

**Why it did not bite yet:** Stage 1 used to crash first on long recordings, so
Stage 3 was never reached with a multi-hour file. Now that Stage 1 streams end
to end, Stage 3 is the only remaining ceiling on recording length.

**Cost:** the clean WAV is mono 16 kHz 16-bit, so ~115 MB per hour of audio on
disk. `AudioSegment.from_wav()` takes pydub's WAV fast path — no ffmpeg
subprocess — but still holds the full PCM payload plus a read copy, so budget
roughly 2x file size. For the 2h43m reference recording that is a 298 MB WAV and
~600 MB peak, on top of whatever Whisper has allocated.

**Likely fix:** read slices on demand with `soundfile.read(..., start=, stop=)`
using the segment offsets, instead of materialising the whole file. `soundfile`
is already a dependency, and `preprocess.py` now uses exactly this pattern via
an open `sf.SoundFile` reader with `seek()` / `read(n)`.

**Constraint:** both `_transcribe_accurate()` and `_transcribe_fast()` take an
`AudioSegment` and slice it with `audio[start_ms:end_ms]`, so the helper
signatures change together with the loading strategy.

---

## Resolved

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
`GlobalMemoryStatusEx().ullAvailPhys` reaches the target, then run the pipeline
as a subprocess under that balloon. Note this holds pages resident, so it is a
harsher condition than natural memory contention.
