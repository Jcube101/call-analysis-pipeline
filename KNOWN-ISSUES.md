# Known Issues — Call Analysis Pipeline

Open problems that are understood but deliberately not fixed yet, plus fixed
ones whose original diagnosis was wrong and is worth not repeating. Each entry
records the root cause and the measurement behind it so a future session does
not have to re-diagnose from scratch.

For solved problems and the reasoning behind existing design choices, see
[LEARNINGS.md](LEARNINGS.md). For planned feature work, see [ROADMAP.md](ROADMAP.md).

---

## 1. Stage 2 ran out of host RAM on long recordings — FIXED

**Status:** fixed. The 2h43m reference recording now completes reliably. Kept
here rather than moved to Resolved because the diagnosis below was wrong for a
long time and the correction is the useful part.

### The real cause: an O(N²) clustering matrix, not the waveform

Stage 2 hands its speaker embeddings to `scipy.cluster.hierarchy.linkage`, which
builds a **full condensed pairwise distance matrix**. That is quadratic in the
number of embeddings, and the number of embeddings is linear in recording length
(one 10 s window per second, up to 3 speakers each).

Measured on the 2h43m file: **16,760 embeddings** reach `linkage` (of a 29,241
upper bound; `filter_embeddings` keeps ~57%), at dimension 256 — a **1.05 GB**
condensed matrix, and roughly **2.1 GB** once centroid linkage takes its working
copy. Isolated per-method runs at N=8000, predicted condensed size 244 MB:

| method | peak delta |
|---|---|
| centroid (what 3.1 uses) | 478 MB |
| average | 478 MB |
| ward | 478 MB |
| single | 263 MB |

Scaling, upper bound on embeddings:

| audio | embeddings | condensed matrix |
|---|---|---|
| 8 min | 1,413 | 0.01 GB |
| 60 min | 10,773 | 0.43 GB |
| 120 min | 21,573 | 1.73 GB |
| 162 min | 29,241 | 3.19 GB |
| 240 min | 43,173 | 6.94 GB |

**The earlier diagnosis pointed at the wrong line.** The failure was recorded at
`get_embeddings`, and the waveform was assumed to be the load. Re-running under
pressure put it in `clustering.py:365` → `hierarchy.py:1064` every time, three
times out of three. The 640 KB allocation quoted in the old write-up was the
straw, not the weight — by then the process was already at its ceiling.

### The fix: pyannote's own cap, which the 3.1 config disables

`BaseClustering.__init__` defaults `max_num_embeddings=1000` and subsamples in
`filter_embeddings()`, clustering that subset and labelling everything else
through `assign_embeddings()`. The pretrained `speaker-diarization-3.1` pipeline
instantiates with **`max_num_embeddings = inf`**, so the cap never fires.

`stages/diarize.py` now sets it to 1000. Measured on the 2h43m file with
`num_speakers=2`:

| cap | segments | speaker split | peak |
|---|---|---|---|
| `inf` | 3527 | **1278 s / 6213 s** | 7656 MB |
| 5000 | 3389 | 3630 s / 3861 s | 5689 MB |
| 1000 | 3386 | 3626 s / 3864 s | 5671 MB |

This buys output quality as well as memory: uncapped splits a two-person
conversation 17/83, which is the label-collapse failure. Caps of 1000 and 5000
agree on **99.96%** of frames (best label permutation, 0.1 s resolution), so the
smaller one costs nothing. In auto-detect mode (no `num_speakers`) uncapped found
**11 speakers** on the same two-person recording; capped found 3.

### Second fix: stop holding the audio in memory

pyannote now receives a **file path**, and the MFCC re-identification pass reads
each segment from the clean WAV instead of copying a resident waveform. Together
these removed ~900 MB on the 2h43m file. See CLAUDE.md, "Audio input to
pyannote" — the `torchcodec` rationale for the in-memory dict was mistaken.

### Result

Three runs before the audio change and three after, `num_speakers=2`:

| | peak | outcome |
|---|---|---|
| before any fix | — | MemoryError in `linkage`, 3 of 3 |
| clustering cap only | 6597 MB | completes, 3 of 3 |
| cap + path + no MFCC copy | **5670 MB** | completes, 3 of 3 |

One of those runs started at 0.81 GB `AvailPhys`, inside the range where it
previously failed. Output is equivalent, not merely similar: frame agreement
across the audio change is 99.88-99.94%, against 99.92-99.97% between repeat runs
of the *same* build — the residual is pyannote's own run-to-run nondeterminism.

### Remaining safety net: the memory preflight

`_check_memory_headroom()` runs before Stage 2 and fails immediately with a clear
message rather than letting a shortfall surface ten minutes into a run.

Two things about it are easy to get wrong, and both were, before measurement:

- **It checks `ullAvailPageFile`, not `ullAvailPhys`.** A `MemoryError` on
  Windows is a failed *commit*. This box routinely sits at ~1 GB `AvailPhys` with
  ~8 GB of commit headroom, and the three runs that succeeded started at 0.81,
  1.24 and 1.40 GB `AvailPhys`. Gating on `AvailPhys` rejects runs that work.
- **It subtracts what the process has already committed.** The budget is a peak
  *total*; by the time Stage 2 starts, torch, the CUDA context and Stage 1's
  leftovers are already committed. Comparing the whole peak against headroom
  rejected the 2h43m file that completes 3 out of 3.

Budget is fitted from two measured peaks — 5404 MB at 2 minutes, 5670 MB at
2h43m — giving **5400 MB + 100 MB per hour**, which predicts both within 1 MB.
Before the audio change the same fit gave 5396 MB + 443 MB/hour; recalibrate
these constants if Stage 2's memory profile changes again.

### Still true, and still not fixed

Stage 2 is **not streaming**. `Inference.__call__` loads the whole signal once for
the segmentation pass regardless of input form; only the embedding crops avoid
residency. The ceiling is higher, not removed.

### Do not be misled by CUDA OOM errors here

The same failure sometimes surfaces as a CUDA error instead:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 158.00 MiB.
GPU 0 has a total capacity of 4.00 GiB of which 2.98 GiB is free.
Of the allocated memory 215.13 MiB is allocated by PyTorch, and 12.87 MiB
is reserved by PyTorch but unallocated.
```

**That is not a VRAM capacity or fragmentation problem** — 158 MiB requested
against 2.98 GiB free with only 215 MiB allocated. On Windows WDDM a CUDA
allocation must be backed by host memory, so host RAM pressure makes the driver
fail and PyTorch reports it as a CUDA OOM. Measured directly by allocating
128 MiB CUDA blocks in a loop until failure:

| Host condition | Allocated on GPU before failure | GPU still free | Result |
|---|---|---|---|
| no pressure (`AvailPhys` 1.02 GB) | **3.75 GiB** | 0 MiB | no error — hit the test's own cap |
| under a balloon (`AvailPhys` 0.22 GB) | 1.75 GiB | **1.47 GiB** | `CUDA out of memory` |

Every Stage 2 OOM observed so far happened while `AvailPhys` was 0.22-1.2 GB.
Same family as the `WinError 1455` note in CLAUDE.md.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **unlikely to help but
low-cost to try** — it addresses fragmentation of reserved-but-unallocated
blocks, and at 215 MiB allocated there is nothing to defragment. Untested. Do
not treat it as the leading candidate; the host-RAM work above is.

---

## Tooling caveat — `squeeze.py` cannot pressure-test GPU stages

The memory balloon used for the Stage 1 measurements allocates and touches
128 MB blocks until `GlobalMemoryStatusEx().ullAvailPageFile` reaches a target,
then runs the pipeline as a subprocess under it.

**It constrains `AvailPageFile`, not `AvailPhys`.** Squeezing to 3.0 GB
AvailPageFile leaves only ~0.22 GB of AvailPhys on this machine. That is a valid
— indeed harsh — test for CPU-only work like Stage 1, which is why the Stage 1
numbers below are trustworthy.

**It is not valid for Stage 2 or Stage 3.** At that little physical RAM the CUDA
driver cannot get the host memory it needs to back device allocations, so GPU
stages fail on driver backing rather than on anything the pipeline controls.
A "Stage 2 fails at 3.0 GB" result from this tool says nothing about Stage 2.

Do not re-run that test expecting a meaningful answer. To pressure-test a GPU
stage, constrain `ullAvailPhys` directly and keep enough physical headroom for
the driver, or test on a machine with more RAM.

---

## Resolved

> **What the Stage 1 and Stage 3 entries below actually bought — corrected.**
> Each of them is accurate about its own stage, and each was worth doing: Stage 1
> genuinely could not process long files before, and Stage 3's `_WavReader`
> genuinely decides the run at 3.0 GB. But they were also read at the time as
> progress *towards* fixing Stage 2, on the assumption that audio residency was
> the thing starving it. It was not. Stage 2's ceiling was an O(N²) clustering
> matrix (see issue 1), which no amount of audio streaming would have touched —
> the float64 fix below moved the failure point later and was credited with more
> than it delivered. Treat "peak memory is now flat with respect to recording
> length" as a per-stage claim, never a whole-pipeline one.

### Stage 2 read the clean WAV as float64 (fixed)

[`stages/diarize.py`](stages/diarize.py) called `sf.read(clean_wav_path)` and
took `soundfile`'s float64 default on a 16-bit file. On the 2h43m reference
recording that was a 1.16 GiB array, and because the next line is
`torch.from_numpy(data).float()`, it also forced a second full-length copy just
to narrow back to float32. Together with `_reidentify_speakers()`'s own
`audio_np` copy, three full-length arrays were live at once — roughly 2.5 GB.
An end-to-end run at 2.1 GB free RAM cleared Stage 1 and then died here with
`numpy ... Unable to allocate 1.16 GiB for an array with shape (156094464,) and
data type float64`.

Fixed by passing `dtype="float32"`. Every sample in a 16-bit WAV is
`int16 / 2**15` and exactly representable in float32, so the values are
identical — this is a smaller copy of the same numbers, not a new rounding
step. Verified: the waveform tensor pyannote actually receives is bit-identical
(`torch.equal` true, max abs diff 0.0), and a full Stage 2 run before and after
produced byte-identical segment output (37 segments on `01_Test_File_clean.wav`).

The array halves from 1.16 GiB to 595 MB and one full-length copy disappears.
On the 2h43m file this moved Stage 2's failure point from `sf.read` at the very
start through to `get_embeddings` near the end — a real improvement, though not
enough to complete on a 7.42 GB machine (see open issue 1).

**Superseded.** This entry used to close by saying the waveform is handed to
pyannote in memory *on purpose*, to avoid a `torchcodec` dependency. Both halves
of that turned out to be wrong: pyannote 3.4.0 reads through torchaudio's
soundfile backend and never needed torchcodec, and the code now passes a file
path, with the MFCC pass reading segments from disk. The `float32` change below
still stands on its own — it is a smaller, bit-identical array either way — but
it did **not** move Stage 2's failure point for the reason claimed here. The
binding constraint was the clustering matrix (issue 1), which this never touched.

### `pydub` removed from the dependency list (fixed)

Nothing imported `pydub` any more: Stage 1 stopped using it when decoding moved
to a direct ffmpeg pass, and Stage 3 stopped when the clean WAV moved behind
`_WavReader`. It has been dropped from `requirements.txt`, and CLAUDE.md and
README.md no longer name it. The remaining mentions in `preprocess.py` and
`transcribe.py` are comments recording which pydub behaviour each replacement
reproduces — keep those.

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
pipeline as a subprocess under that balloon.

The balloon touches every page to force residency, so it is harsher than natural
memory contention — which is what makes these Stage 1 numbers meaningful. See
"Tooling caveat" above before using the same approach on Stage 2 or Stage 3: it
does not work for GPU stages.
