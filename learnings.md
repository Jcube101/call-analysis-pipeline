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

## 12. Windows-specific issues

A collection of issues that only manifest on Windows or in Windows-hosted GPU environments.

### ctranslate2 CUDA teardown calls exit() on garbage collection

**Problem:** When `WhisperModel` is garbage-collected mid-process on Windows, ctranslate2's `__del__` calls `exit()`, crashing the server.

**Fix:** Keep a module-level reference in `transcribe.py`:
```python
_active_model = None  # module-level
# inside run():
model = WhisperModel(...)
_active_model = model  # prevents GC
```
Do not add `del model` anywhere inside `transcribe.run()`.

In `api.py` this is compounded because pipeline jobs run in background threads. `BaseException` (not just `Exception`) is caught in pipeline threads because `SystemExit` from ctranslate2 is a `BaseException` subclass. `threading.excepthook` is also installed so any uncaught thread crash prints a traceback without killing uvicorn.

### CUDA memory not released between Stage 2 and Stage 3

**Problem:** pyannote and Whisper together exceed 4 GB VRAM. After `diarize.run()` finishes, enough VRAM remains allocated from pyannote that `WhisperModel` fails to initialise.

**Fix:** At the end of `diarize.run()`, move the pipeline off GPU and explicitly clear the cache:
```python
pipeline.to(torch.device("cpu"))
del pipeline, diarization, annotation
torch.cuda.empty_cache()
gc.collect()
```

In `api.py`, an additional cleanup step is performed in the pipeline thread after `diarize.run()` returns and before `transcribe.run()` is called:
```python
torch.cuda.synchronize()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
time.sleep(2)
```
The `sleep(2)` gives the CUDA driver time to process the deallocation before the new model is loaded.

### torchaudio not needed — use soundfile instead

**Problem:** `torchaudio` was listed as a dependency for audio loading but caused version conflicts on Windows (requires a matching `torch` build; CPU wheel indices differ between torchaudio and torch).

**Fix:** Load audio with `soundfile.read()` instead of any torchaudio API. `soundfile` is pure Python / libsndfile and has no torch version coupling. See Learning #4.

### numpy 2.x breaks pyannote on Windows pip installs

See Learning #3. On Windows, `pip install -r requirements.txt` without pre-installing torch tends to resolve to the latest torch, which triggers a numpy 2.x upgrade that breaks pyannote. The explicit install order (torch first, then numpy pin, then requirements) is especially important on Windows where pip caching behaviour differs from Linux.

---

## 13. google-generativeai is deprecated — use google-genai instead

**Problem:** `import google.generativeai` emitted a `FutureWarning` stating that all support for the `google.generativeai` package has ended and it will no longer receive updates or bug fixes.

**Fix:** Switch to the `google-genai` package (`pip install google-genai`). The new client API:
```python
from google import genai

client = genai.Client(api_key=api_key)
response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
return response.text
```

**Note:** `google-generativeai` and `google-genai` are separate packages with different import paths (`google.generativeai` vs `google.genai`) and different client patterns. Do not mix them.

---

## 14. FastAPI + threading gotchas

### Form booleans must be cast and stored explicitly

FastAPI's `bool = Form(False)` parses HTML form values correctly in most cases, but values routed through background threads can lose type information. Store flag values explicitly on the job dict with an explicit cast:

```python
jobs[job_id]["generate_report"] = bool(generate_report)
```

Then in the background thread, read from job state rather than function arguments:

```python
generate_report = bool(job.get("generate_report", False))
```

This also handles any string residue (`"true"`, `"1"`) that survives form parsing.

### Background thread crashes kill uvicorn unless excepthook is set

Python's default `threading.excepthook` re-raises uncaught exceptions, which terminates the entire process. Install a custom hook at module level to log crashes without killing the server:

```python
def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
    print(f"[error] Unhandled thread exception: {args.exc_value}")
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_tb)

threading.excepthook = _handle_thread_exception
```

### WebSocket messages from background threads require run_coroutine_threadsafe

Background CPU threads cannot call `await websocket.send_json()` directly. Use the event loop captured at startup:

```python
future = asyncio.run_coroutine_threadsafe(ws.send_json(payload), _loop)
future.result(timeout=5)  # confirm delivery or time out
```

### CUDA memory must be freed between pyannote and Whisper

See Learning #12 (Windows-specific issues). The pattern `synchronize() + empty_cache() + gc.collect() + empty_cache() + sleep(2)` between Stage 2 and Stage 3 is required in the API server's background thread, not just in `diarize.run()` itself.

---

## 15. Windows file path gotchas in glob patterns

### Never filter on "input" in filenames — all output files start with "input_"

When `api.py` processes an uploaded file it saves it as `input.{ext}` and all output files are named with that stem:

```
input_clean.wav
input_20260331_014537.txt
input_20260331_014537.json
input_20260331_014548_report.md
```

Filtering glob results with `"input" not in filename` removes **every** output file. Filter by specific suffixes instead:
- Exclude `_report` from txt results (to avoid hypothetical `*_report.txt` files)
- Exclude `named` from json results (to separate `transcript_named.json` from timestamped JSON)

### Glob patterns must use os.path.join or forward slashes

On Windows, `glob.glob("output\\jobs\\{job_id}\\*.txt")` silently returns no results because the glob module expects forward slashes. Use `os.path.join()`:

```python
candidates = glob.glob(os.path.join("output", "jobs", job_id, "*.txt"))
```

### Content-Disposition header must be set for correct browser filename

`FileResponse(path)` without `filename=` lets the browser infer the filename from the URL path (e.g. `/download/{job_id}/report`), which it saves as `report` with no extension. Always pass `filename=os.path.basename(path)` so the browser saves with the correct extension (`.txt`, `.json`, `.md`, `.wav`).

---

## 16. Speaker diarization limitations on single-mic recordings

### Majority vote smoothing fails for asymmetric conversations

Majority vote smoothing (window=5) assumes speakers alternate roughly evenly. In an interview or boss/reportee dynamic where one speaker holds the floor for long stretches, the minority speaker's segments get outvoted into the majority speaker's label. The smoothing makes things worse, not better.

**Mitigation:** Only apply majority vote smoothing to recordings known to have roughly balanced speaker time. For interviews, skip it or use a very small window (2–3 segments).

### MFCC re-identification helps with label drift, not misattribution

The MFCC clustering step is designed to fix *label drift* — the same physical voice being assigned `SPEAKER_00` in one chunk and `SPEAKER_01` in another on long recordings. It cannot reliably fix *short-segment misattribution* where a segment genuinely contains overlapping speech or a very brief interjection at a conversation boundary.

### Single-mic vs stereo: a fundamental ceiling

On a single-microphone recording both voices share the same acoustic channel. The diarization model must separate them purely from spectral features. Stereo recordings (one speaker per channel) allow per-channel processing and near-perfect separation. For portfolio demos and personal use, single-mic is fine — but document the limitation honestly.

### Practical workflow for label flipping

1. Run the full pipeline to get the transcript.
2. Open the `.txt` file and identify which label belongs to which person.
3. Re-run with `--from-json output/<name>.json --speaker-names "Alice,Bob"` to write a corrected `_named.json` with real names.
4. Generate the report from the corrected JSON: `--from-json output/<name>_named.json --report`.

This two-pass approach is more reliable than trying to engineer perfect automatic diarization on single-mic audio.
