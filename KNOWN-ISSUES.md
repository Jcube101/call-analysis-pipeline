# Known Issues — Call Analysis Pipeline

Open problems that are understood but deliberately not fixed yet. Each entry
records the root cause and the measurement behind it so a future session does
not have to re-diagnose from scratch.

For solved problems and the reasoning behind existing design choices, see
[LEARNINGS.md](LEARNINGS.md). For planned feature work, see [ROADMAP.md](ROADMAP.md).

---

## 1. Stage 2 cannot diarize multi-hour recordings on this machine

**Status:** open — untouched. The only open issue left, and the one that stops a
2h43m recording. Stages 1 and 3 both clear it.

**It presents as a GPU error but the evidence says host RAM.** Read the next
section before trying a CUDA fix, because the obvious one is unlikely to help.

**Symptoms.** Two different failures, both inside Stage 2, depending on how much
host memory is free:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 158.00 MiB.
GPU 0 has a total capacity of 4.00 GiB of which 2.98 GiB is free.
Of the allocated memory 215.13 MiB is allocated by PyTorch, and 12.87 MiB
is reserved by PyTorch but unallocated.
```

```
File "pyannote/audio/pipelines/speaker_diarization.py", line 339, in get_embeddings
RuntimeError: [enforce fail at alloc_cpu.cpp:114] DefaultCPUAllocator:
  not enough memory: you tried to allocate 640000 bytes.
```

The second is plainly host RAM — a 640 KB CPU allocation failing. The first
looks like VRAM but is not: **158 MiB requested, 2.98 GiB free, 215 MiB
allocated.** No VRAM capacity limit and no fragmentation pattern produces those
numbers.

**Measured directly.** Allocating 128 MiB CUDA blocks in a loop until failure:

| Host condition | Allocated on GPU before failure | GPU still free | Result |
|---|---|---|---|
| no pressure (`AvailPhys` 1.02 GB) | **3.75 GiB** | 0 MiB | no error — hit the test's own cap |
| under a balloon (`AvailPhys` 0.22 GB) | 1.75 GiB | **1.47 GiB** | `CUDA out of memory` |

On Windows WDDM a CUDA allocation must be backed by host memory, so host RAM
pressure surfaces as a spurious "CUDA out of memory" with gigabytes of VRAM
free. Every Stage 2 OOM observed so far happened while `AvailPhys` was
0.22-1.2 GB. This is the same family as the `WinError 1455` note in CLAUDE.md.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is worth one try but is not
expected to help** — it addresses fragmentation of reserved-but-unallocated
blocks, and at 215 MiB allocated there is nothing to defragment. Untested.

**The real constraint** is that this box has 7.42 GB of RAM with roughly
1.0-1.5 GB genuinely free at rest, and pyannote needs the whole waveform
resident plus its own embedding buffers. The `float32` fix (see Resolved) took
Stage 2's own buffers from ~2.5 GB to ~1.2 GB and moved the failure point
noticeably later — from `sf.read` at the very start, through to
`get_embeddings` near the end — but did not clear it.

**Worth trying, in order:** run Stage 2 on CPU for long files; chunk the
waveform through pyannote; or free host RAM before the run. Note that the
waveform is passed in memory *on purpose*, to avoid a `torchcodec` dependency
(CLAUDE.md, "Audio input to pyannote"), so a file-path-based fix trades one
problem for another.

---

## Resolved

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

**Not fully streamable**, so this is a shrink rather than an elimination: the
waveform is handed to pyannote as an in-memory
`{"waveform": ..., "sample_rate": ...}` dict *on purpose*, to avoid a
`torchcodec` dependency (see CLAUDE.md, "Audio input to pyannote"). pyannote
needs the whole signal, and so does the MFCC re-identification pass.

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

Two caveats when reading the numbers above. The balloon touches every page to
force residency, so it is harsher than natural memory contention. And the
target is **AvailPageFile**, not physical RAM: squeezing to 3.0 GB AvailPageFile
leaves only ~0.22 GB of AvailPhys on this machine. That is fine for the CPU-only
work in Stage 1, but it starves the CUDA driver of the host memory it needs to
back device allocations, so **GPU stages cannot be meaningfully tested under the
balloon at all** — they fail on host backing rather than on anything the
pipeline controls.
