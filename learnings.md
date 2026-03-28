# Learnings — Call Analysis Pipeline

Lessons discovered during development and first real-world testing. Kept here so future sessions don't repeat the same debugging cycles.

---

## 1. pyannote 3.x changed its pipeline return type

**Problem:** `diarization.itertracks(yield_label=True)` raised `AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'`.

**Root cause:** pyannote.audio 3.x no longer returns a `pyannote.core.Annotation` directly from `Pipeline.__call__`. It returns a `DiarizeOutput` dataclass. The actual annotation lives in the `exclusive_speaker_diarization` attribute.

**Fix:** Resolve the annotation object before iterating, with a fallback chain:

```python
if hasattr(diarization, "itertracks"):
    annotation = diarization                                     # old API
elif hasattr(diarization, "exclusive_speaker_diarization"):
    annotation = diarization.exclusive_speaker_diarization       # new API (3.x)
elif hasattr(diarization, "speaker_diarization"):
    annotation = diarization.speaker_diarization
```

**What `DiarizeOutput` actually contains:**
```
['exclusive_speaker_diarization', 'serialize', 'speaker_diarization', 'speaker_embeddings']
```

Use `exclusive_speaker_diarization` (not `speaker_diarization`) — it resolves overlapping speech so each time window maps to exactly one speaker, which is what transcription needs.

**Diagnostic pattern:** When you hit an unknown object type, add temporary debug prints:
```python
print(type(diarization))
print([a for a in dir(diarization) if not a.startswith('_')])
```

---

## 2. torchcodec warning is harmless — avoid installing it

**Problem:** pyannote emits a `UserWarning` on import:
```
torchcodec is not installed correctly so built-in audio decoding will fail.
```

**Root cause:** pyannote 3.x tries to use `torchcodec` for file-based audio loading. We don't use that path.

**Fix:** Pass audio as a pre-loaded in-memory dict — pyannote's *other* supported input format:
```python
data, sample_rate = sf.read(clean_wav_path)
waveform = torch.from_numpy(data).float()
audio_input = {"waveform": waveform, "sample_rate": sample_rate}
diarization = pipeline(audio_input, ...)
```

Suppress the warning at import time since it's irrelevant:
```python
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="torchcodec is not installed", category=UserWarning)
    from pyannote.audio import Pipeline
```

**Do not install torchcodec** — it has strict torch version requirements and adds complexity.

---

## 3. pip dependency resolution installs incompatible numpy when given free rein

**Problem:** `pip install -r requirements.txt` pulled in `torch 2.10.0` + `numpy 2.4.2`, which broke `pyannote.audio` (numpy 2.x API incompatibility).

**Root cause:** When pip sees `torch` in requirements without a pin, it resolves to the latest version. `torch 2.10.0` declares `numpy>=2.0` as acceptable, so pip upgrades numpy. `pyannote.audio` then fails because it uses numpy 1.x APIs.

**Fix — install in this exact order:**
```bash
# Step 1: torch from the CPU wheel index (prevents PyPI from resolving torch 2.10+)
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Step 2: force numpy back before anything else can touch it
pip install "numpy<2.0" --force-reinstall

# Step 3: everything else will link against the already-installed torch/numpy
pip install -r requirements.txt
```

**Key insight:** The install order matters because pip doesn't re-evaluate already-satisfied constraints when processing later packages. If torch is installed first, subsequent packages won't upgrade it.

---

## 4. soundfile is the right way to load audio for pyannote

**Problem:** Loading audio with pydub/librosa and converting to a tensor introduced channel dimension bugs.

**Fix:** Use `soundfile.read()` — it returns a clean numpy array with explicit shape documentation:

```python
import soundfile as sf
data, sample_rate = sf.read(clean_wav_path)
if data.ndim == 1:
    data = data[np.newaxis, :]   # mono: add channel dim → (1, samples)
else:
    data = data.T                 # stereo: (samples, channels) → (channels, samples)
waveform = torch.from_numpy(data).float()
```

pyannote expects waveform shape `(channels, time)` — this is the opposite of numpy's default `(time, channels)`.

---

## 5. M4A files work fine — ffmpeg handles the format transparently

Both `pydub.AudioSegment.from_file()` (Stage 1) and the ffmpeg preflight check in `main.py` handle M4A without any special casing. The pipeline accepts any ffmpeg-supported format; the README and spec previously said "MP3" but this was misleading.

**Tested formats:** MP3, M4A (iPhone voice memos, WhatsApp audio).

---

## 6. Debugging unknown API objects: use `dir()` + `hasattr()`

When working with third-party libraries that change between versions, a reliable pattern is:

1. Add a temporary `print(type(obj))` and `print(dir(obj))` to identify what you're working with
2. Write `hasattr()`-guarded fallback chains rather than assuming a single API
3. Raise a descriptive `RuntimeError` as the last branch so future API changes produce a clear message rather than an `AttributeError`

This pattern was used to diagnose the `DiarizeOutput` issue and is now baked into `diarize.py` as a permanent defensive fallback.

---

## 7. CPU-only torch on Windows is a legitimate and stable setup

The `torch==2.1.0+cpu` build works reliably on Windows 11 for pyannote diarization and faster-whisper transcription. CUDA is not required for the pipeline to function — it just reduces transcription time from ~10x real-time to ~1x real-time. All core logic was developed and validated on CPU before GPU was added.

---

## 8. pyannote's SpeakerEmbedding pipeline calls `.to(device)` on the AudioFile dict

**Problem:** Attempting to extract per-segment voice embeddings by calling `pipeline._embedding({"waveform": tensor, "sample_rate": int})` raised `'dict' object has no attribute 'to'` for every segment. 185 segments processed in <0.002 s (all silently failed).

**Root cause:** `pipeline._embedding` is a `SpeakerEmbedding` pipeline (not a raw `Inference` object). Internally, it calls `file.to(device)` on the AudioFile argument — expecting a pyannote `ProtocolFile` object that has a `.to()` method. Plain Python dicts do not have `.to()`. Navigating to the inner `Inference` object (`pipeline._embedding._embedding`) did not resolve it — the `.to(dict)` call occurs at a different level of the stack.

**Fix:** Abandon pyannote's embedding API entirely for this use case. Use `librosa.feature.mfcc` + delta features (mean-pooled over time) per segment, then cluster with KMeans. MFCC features are sufficient to distinguish two speakers reliably on call recordings and have no dependency on pyannote internals or GPU placement.

**Lesson:** When a third-party internal API (prefixed with `_`) fails in an opaque way across multiple nesting levels, the right move is to replace it with a self-contained implementation rather than keep digging into private internals.

---

## 9. noisereduce STFT causes OOM on long recordings — use chunked processing

**Problem:** `noisereduce.reduce_noise()` on a 90+ minute audio array caused RAM spikes of 3–4 GB and risked OOM crashes.

**Root cause:** noisereduce internally computes a full STFT of the input. For 90 min at 16 kHz the STFT matrix alone is ~1.4 GB (1025 bins × 168k frames × 8 bytes complex64), and multiple temporary arrays coexist during processing.

**Fix:** Process in 60-second slices with a 0.5-second overlap on each side for boundary context. Each slice's STFT is ~57 MB. Only the core (non-overlap) portion of each reduced slice is written to the output array:
```python
for each chunk at pos:
    read_start = max(0, pos - overlap)
    read_end   = min(total, pos + chunk_size + overlap)
    reduced = nr.reduce_noise(y=samples[read_start:read_end], ...)
    offset = pos - read_start
    output[pos : pos + chunk_size] = reduced[offset : offset + chunk_size]
```
Peak RAM stays at ~350 MB regardless of recording length.

**Validated on:** 2h40m M4A file (163 chunks), Stage 1 completed in 2m53s with no OOM errors.

---

## 10. Dead code in Stage 2 was loading full audio for no reason

**Problem:** `diarize.py` called `_normalize_speaker_segments()` but immediately discarded the return value. For a 2h40m file this loaded ~350 MB of audio and created per-speaker audio chunks that were never used.

**Fix:** Removed the dead function call and the function itself. The speaker segment boundaries from pyannote are accurate regardless.

**Lesson:** "Informational" code that produces no side effects and whose return value is discarded should be deleted, not left in place — especially when it carries significant memory cost.

---

## 11. Pipeline scales linearly with recording length at ~1x real-time on GTX 1650

**Observation:** 2h40m recording processed in 2h43m total elapsed:
- Stage 1 (noise reduction): 2:53 (163 chunks × ~1s each)
- Stage 2 (diarization): 11:14 (3102 segments — pyannote batches internally)
- Stage 3 (transcription): 2:29 (3102 segments × 2.88s each)
- Stage 4 (export): <1s

**Conclusion:** Accurate mode scales linearly with segment count. Stage 2 is sublinear (pyannote processes audio in batches that amortise well). There is no hard ceiling for file length — the pipeline will complete in approximately real-time on a GTX 1650 for any length recording.

## 12. google-generativeai is deprecated — use google-genai instead

**Problem:** `import google.generativeai` emitted a `FutureWarning` stating that all support for the `google.generativeai` package has ended and it will no longer receive updates or bug fixes.

**Fix:** Switch to the `google-genai` package (`pip install google-genai`). The new client API:
```python
from google import genai

client = genai.Client(api_key=api_key)
response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
return response.text
```

**Note:** `google-generativeai` and `google-genai` are separate packages with different import paths (`google.generativeai` vs `google.genai`) and different client patterns. Do not mix them.
