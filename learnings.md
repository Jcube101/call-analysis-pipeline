# Learnings — Call Analysis Pipeline

Lessons discovered during development and testing. Kept here so future sessions don't repeat the same debugging cycles.

---

## 1. pyannote 3.x changed its pipeline return type

**Problem:** `diarization.itertracks(yield_label=True)` raised `AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'`.

**Root cause:** pyannote.audio 3.x no longer returns a `pyannote.core.Annotation` directly. It returns a `DiarizeOutput` dataclass. The actual annotation lives in `exclusive_speaker_diarization`.

**Fix:** Resolve the annotation with a fallback chain:
```python
if hasattr(diarization, "itertracks"):
    annotation = diarization
elif hasattr(diarization, "exclusive_speaker_diarization"):
    annotation = diarization.exclusive_speaker_diarization
elif hasattr(diarization, "speaker_diarization"):
    annotation = diarization.speaker_diarization
```

Use `exclusive_speaker_diarization` (not `speaker_diarization`) — it resolves overlapping speech so each window maps to exactly one speaker.

**Diagnostic pattern:** When hitting an unknown object type, add:
```python
print(type(obj))
print([a for a in dir(obj) if not a.startswith('_')])
```

---

## 2. torchcodec warning is harmless — pass audio in-memory instead

**Problem:** pyannote emits a `UserWarning` about `torchcodec` on import.

**Fix:** Pass audio as a pre-loaded dict — pyannote's other supported input format:
```python
data, sample_rate = sf.read(clean_wav_path)
waveform = torch.from_numpy(data).float()
audio_input = {"waveform": waveform, "sample_rate": sample_rate}
diarization = pipeline(audio_input, ...)
```
Suppress the warning at import since it is irrelevant to this path.

---

## 3. pip resolves incompatible numpy when given free rein

**Problem:** `pip install -r requirements.txt` pulled in `torch 2.10.0` + `numpy 2.4.x`, breaking pyannote.

**Fix:** Install torch from the wheel index first, then pin numpy:
```bash
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

---

## 4. pyannote 4.0.4 requires torch>=2.8.0 which doesn't exist

**Problem:** `pip install pyannote.audio` installs 4.0.4 which declares `torch>=2.8.0`. No such torch version exists on PyPI.

**Fix:** Pin `pyannote.audio<4.0` — version 3.4.0 works correctly with torch 2.1.0.

---

## 5. HuggingFace auth API changed across package versions

pyannote 3.x went through three auth API changes:

| pyannote call | huggingface_hub version | Status |
|--------------|------------------------|--------|
| `Pipeline.from_pretrained(token=...)` | older | Removed in pyannote 3.4.0 |
| `Pipeline.from_pretrained(use_auth_token=...)` | <1.0.0 | Removed in huggingface_hub 1.0 |
| `huggingface_hub.login(token=...)` + no kwarg | any | **Works** |

**Fix:** Always use `huggingface_hub.login()` before `from_pretrained()`. Do not pass the token as a keyword argument to `from_pretrained()`.

---

## 6. huggingface_hub 1.x removed use_auth_token from internal API

**Problem:** pyannote 3.4.0's internal `pipeline.py` calls `hf_hub_download(use_auth_token=...)`. huggingface_hub 1.0+ removed that parameter, causing a `TypeError` inside pyannote's own code — unfixable from our side.

**Fix:** Pin `huggingface_hub<1.0.0`. Version 0.36.2 is confirmed working. Note: faster-whisper pulls in huggingface_hub 1.x if installed first — install in the right order or force-reinstall after.

---

## 7. float16 compute type exceeds 4 GB VRAM on GTX 1650

**Problem:** `WhisperModel(model_name, device="cuda", compute_type="float16")` raised `RuntimeError: CUDA failed with error out of memory`.

**Fix:** Use `compute_type="int8_float16"` — weights stored as int8 (half the VRAM), computed in float16 (still fast). Fits comfortably in 4 GB.

```python
device, compute_type = "cuda", "int8_float16"   # GPU, 4 GB VRAM safe
device, compute_type = "cpu",  "int8"            # CPU fallback
```

---

## 8. ctranslate2 calls exit() when WhisperModel is deleted mid-process on Windows

**Problem:** Pipeline completed Stage 3 and printed "Transcription complete." but then silently exited — Stage 4 never ran, no error shown.

**Root cause:** ctranslate2's `__del__` method triggers CUDA teardown which calls `exit()` when the `WhisperModel` local variable goes out of scope at the end of `transcribe.run()`.

**Diagnosis:** Added `del model` explicitly before `return` — the debug print after it never appeared, confirming `del model` was the kill point.

**Fix:** Hold a module-level reference so the model is not garbage-collected until process exit:
```python
_active_model = None   # module level

def run(...):
    global _active_model
    model = WhisperModel(...)
    _active_model = model  # prevents mid-process GC
    ...
    return transcribed
```
At process exit, CUDA teardown is handled safely via atexit handlers.

**Rule:** Never call `del model` inside `transcribe.run()`. Never let the WhisperModel go out of scope before `main()` returns.

---

## 9. Output files were silently overwritten between runs

**Problem:** `transcript.txt` and `transcript.json` were hardcoded filenames — every run overwrote the previous output with no indication.

**Fix:** Include source filename + timestamp in the output stem:
```python
source_stem = os.path.splitext(os.path.basename(source_file))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_stem = f"{source_stem}_{timestamp}"
```
Produces e.g. `First_Test_File_20260327_143022.txt`.

---

## 10. Diagnosing silent exits: use targeted debug prints, not trial-and-error

When a stage silently terminates without error:
1. Add a debug print at the entry of the *next* stage — confirms whether control returned
2. Add a debug print immediately before the `return` in the current stage
3. Add an explicit `del <resource>` before the return to isolate teardown crashes
4. Remove debug commits promptly once the cause is confirmed

In this case: `[DEBUG] Reached Stage 4` never appeared → control didn't return from `transcribe.run()`. Adding `del model` + debug confirmed ctranslate2 teardown was the culprit.

---

## 11. soundfile is the right loader for pyannote input

Use `soundfile.read()` — it returns a clean numpy array with explicit shape:
```python
data, sample_rate = sf.read(clean_wav_path)
if data.ndim == 1:
    data = data[np.newaxis, :]   # mono → (1, samples)
else:
    data = data.T                 # stereo → (channels, samples)
waveform = torch.from_numpy(data).float()
```
pyannote expects `(channels, time)` — opposite of numpy's default `(time, channels)`.

---

## 12. Pre-transcription diarization segment merging collapses transcript output

**Problem:** Adding a merge step that combined consecutive same-speaker diarization segments (with a gap threshold) reduced a 20+ line transcript to 2 lines.

**Root cause:** pyannote commonly splits a single speaking turn into many segments with tiny gaps (<100ms) between them. Any gap threshold — even 0.1s — merges these into one giant segment per speaker, collapsing the entire conversation into one block per speaker.

**Fix:** Do not pre-merge diarization segments before transcription. The post-transcription merge in `_assign_speakers()` (which merges consecutive same-speaker *Whisper* output segments) is the correct place to merge, and it works correctly.

**Lesson:** Merging should happen on Whisper output (sentence-level), not on diarization input (which has no meaningful minimum gap).

---

## 13. vad_filter=True causes Whisper to produce too few segments

**Problem:** Using `vad_filter=True` in `model.transcribe()` for whole-file fast mode transcription produced only 4 Whisper segments for a 2-minute file, resulting in 2 output lines.

**Root cause:** faster-whisper's VAD filter groups audio into large speech chunks before transcribing. This is designed to skip silence, but it also collapses the audio into very few large chunks.

**Fix:** Remove `vad_filter=True`. Without it, Whisper segments at natural speech boundaries and produces more granular output.

**Diagnosis method:** Added a debug print showing `whisper segments: X → assigned: Y → after merge: Z`. The X=4 immediately identified that Whisper itself was only producing 4 segments, pointing to the transcription call rather than the alignment logic.

---

## 14. Fast mode is fundamentally coarser than accurate mode

**Observation:** For a 10:56 file, fast mode produced 49 segments vs accurate mode's 134, and was only ~20% faster (480s vs 624s on GTX 1650).

**Root cause:** Whisper processes audio in ~30s internal chunks regardless of mode. One whole-file call → ~22 chunks → ~49 segments. Per-segment accurate mode → one call per diarization turn → 134 finer-grained segments.

**Conclusion:** The time saving from eliminating per-call overhead is modest (~20%) while the quality loss is significant. `accurate` is the correct default. `fast` is only worth using when coarse output is explicitly acceptable.

**Future improvement:** A smarter fast mode would merge same-speaker diarization *turns* (not micro-segments) and transcribe per turn — fewer calls than accurate, better granularity than current fast.

---

## 15. google-generativeai is deprecated — use google-genai instead

**Problem:** `import google.generativeai` emitted a `FutureWarning` stating that all support for the `google.generativeai` package has ended and it will no longer receive updates or bug fixes.

**Fix:** Switch to the `google-genai` package (`pip install google-genai`). The new client API:
```python
from google import genai

client = genai.Client(api_key=api_key)
response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
return response.text
```

**Note:** `google-generativeai` and `google-genai` are separate packages with different import paths (`google.generativeai` vs `google.genai`) and different client patterns. Do not mix them.
