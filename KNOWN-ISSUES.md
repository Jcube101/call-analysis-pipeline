# Known Issues — Call Analysis Pipeline

Open problems that are understood but deliberately not fixed yet. Each entry
records the root cause and the measurement behind it so a future session does
not have to re-diagnose from scratch.

For solved problems and the reasoning behind existing design choices, see
[LEARNINGS.md](LEARNINGS.md). For planned feature work, see [ROADMAP.md](ROADMAP.md).

---

## 1. `transcribe.py` loads the entire clean WAV into memory

**Status:** open — follow-up, out of scope for the Stage 1 decode fix.

**Where:** [`stages/transcribe.py`](stages/transcribe.py) line 253:

```python
audio = AudioSegment.from_wav(clean_wav_path)
```

**Problem:** Stage 3 reads the whole Stage-1 output into a single in-memory
`AudioSegment` before slicing it per diarization segment. This is the same
in-memory-buffer pattern that caused the Stage 1 `MemoryError`, just one stage
later and at a smaller constant factor.

**Why it did not bite yet:** Stage 1 used to crash first on long recordings, so
Stage 3 was never reached with a multi-hour file. Now that Stage 1 streams
through ffmpeg, Stage 3 is the next ceiling on recording length.

**Cost:** the clean WAV is mono 16 kHz 16-bit, so ~115 MB per hour of audio on
disk. `AudioSegment.from_wav()` takes pydub's WAV fast path — no ffmpeg
subprocess — but still holds the full PCM payload plus a read copy, so budget
roughly 2x file size. For the 2h43m reference recording that is a 298 MB WAV and
~600 MB peak, on top of whatever Whisper has allocated.

**Likely fix:** read slices on demand with `soundfile.read(..., start=, stop=)`
using the segment offsets, instead of materialising the whole file. `soundfile`
is already a dependency and already used this way in `diarize.py`.

**Constraint:** both `_transcribe_accurate()` and `_transcribe_fast()` take an
`AudioSegment` and slice it with `audio[start_ms:end_ms]`, so the helper
signatures change together with the loading strategy.

---

## 2. `preprocess._from_numpy()` is now Stage 1's peak-memory step

**Status:** open — **blocking on low-memory machines.** Discovered while
verifying the Stage 1 decode fix; see the memory-pressure results below.

**Where:** [`stages/preprocess.py`](stages/preprocess.py), `_from_numpy()` and the
`_chunked_reduce_noise()` output buffer.

**Problem:** with the ffmpeg decode fixed, the largest allocation in Stage 1
moved downstream. `_from_numpy()` runs `np.clip()`, a float multiply, and
`.astype(np.int16).tobytes()` over the full-length array, each producing another
full-size temporary.

**Measured** on `input/03_Lunch_with_Rachita.m4a` (2h43m, 48 kHz mono AAC),
private commit at each phase boundary:

| Phase | Private | Peak |
|---|---|---|
| imports loaded (numpy/scipy/noisereduce/pydub) | 1390 MB | 1390 MB |
| ffmpeg decode | 1391 MB | 1391 MB |
| `sf.read` → samples (595 MB array) | 1988 MB | 1988 MB |
| `_chunked_reduce_noise` | 2618 MB | 2775 MB |
| `_from_numpy` → AudioSegment | 2320 MB | **3513 MB** |
| `normalize()` + `export()` | 1723 MB | 3513 MB |

The decode step itself now costs ~1 MB, down from a 2695 MB peak that raised
`MemoryError`. This is not a regression introduced by that fix — these later
phases always cost this much, they were simply never reached on files this long.

**Likely fix:** convert to int16 in place, chunk-wise, reusing one output buffer,
and write the WAV with `soundfile.write()` rather than building an
`AudioSegment` just to call `normalize()` and `export()`. Loudness normalization
would need a two-pass peak scan to stay streaming.

**Note:** the 1390 MB import baseline is private commit, which counts reserved
numpy/scipy arenas; peak working set for the same run was 1929 MB. Use the
deltas between phases, not the absolute numbers, when comparing.

### Behaviour under memory pressure

The Stage 1 decode fix is necessary but **not sufficient** on this dev box
(7.4 GB total RAM). Same 2h43m reference file, varying free RAM:

| Free RAM | Before decode fix | After decode fix |
|---|---|---|
| 5.2 GB | completes, 386 s, peak 4077 MB | completes, **210 s**, peak **3513 MB** |
| 3.0 GB | not measured | `MemoryError` in `_chunked_reduce_noise` after 187 s |
| 2.1 GB | `MemoryError` in `AudioSegment.from_file` after 30 s | `MemoryError` allocating the 595 MB output buffer in `_chunked_reduce_noise` |

So the fix removes the original crash site and makes the healthy path ~45%
faster, but a 2h43m recording still cannot be processed when free RAM is at the
~2 GB level that triggered the original report. The remaining blockers, in the
order they bite:

1. `np.empty(total, dtype=np.float32)` in `_chunked_reduce_noise` — one
   full-length output buffer (595 MB for 2h43m), live at the same time as the
   full-length input array.
2. noisereduce's per-chunk STFT temporaries (float64).
3. `_from_numpy()`'s full-length `np.clip` / multiply / `astype` copies.

Fixing (1) alone — stream each denoised chunk straight to a `soundfile`
writer instead of accumulating into one array — would remove the largest
single allocation and is the cheapest next step.

**Reproducing:** hold free RAM down to a target and run Stage 1 under it. The
harness used for the table above allocates and touches 128 MB blocks until
`GlobalMemoryStatusEx().ullAvailPhys` reaches the target, then runs the
pipeline as a subprocess.
