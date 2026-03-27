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
