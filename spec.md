# Technical Specification — Call Analysis Pipeline

## Overview

A local Python pipeline that processes a recorded voice call and produces a clean, structured transcript with speaker attribution. Designed to run entirely locally (no cloud audio processing), with all secrets managed via `.env`.

**Tested and working on Windows 11, Python 3.11, CPU-only torch.**

---

## Inputs

| Parameter | Source | Description |
|-----------|--------|-------------|
| `--input FILE` | CLI arg | Path to source audio file (MP3, M4A, WAV, or any ffmpeg-supported format) |
| `--context CTX` | CLI / `.env` | Conversation type: `friend`, `work`, `interview`, `date` |
| `--num-speakers N` | CLI / `.env` | Integer speaker count, or omit for auto-detection |
| `--whisper-model SIZE` | CLI / `.env` | Whisper model size: `tiny`, `base`, `small`, `medium` (default), `large` |

CLI flags override `.env` values when both are present.

---

## Outputs

All outputs land in `output/`. Each run produces uniquely named transcript files:

| File | Format | Description |
|------|--------|-------------|
| `<name>_clean.wav` | WAV, mono, 16 kHz | Noise-reduced, normalized audio (overwritten each run) |
| `<name>_<YYYYMMDD_HHMMSS>.txt` | Plain text | Human-readable labeled transcript |
| `<name>_<YYYYMMDD_HHMMSS>.json` | JSON | Structured transcript with metadata header |

### transcript.txt format

```
# Call Transcript
# Source:  call.m4a
# Context: friend
# Speakers: 2
# Processed: 2026-03-27T14:30:22

[00:00:04] Speaker A: "Hey, how are you doing..."
[00:01:12] Speaker B: "I'm good, just got back from..."
```

### transcript.json schema

```json
{
  "metadata": {
    "source_file": "call.m4a",
    "context": "friend",
    "num_speakers": 2,
    "processed_at": "2026-03-27T14:30:22"
  },
  "transcript": [
    {
      "start": 4.2,
      "end": 9.8,
      "speaker": "Speaker A",
      "text": "Hey, how are you doing..."
    }
  ]
}
```

---

## Pipeline stages

### Stage 1 — Audio Pre-processing (`stages/preprocess.py`)

**Goal:** Produce a clean, standardised WAV that downstream stages can rely on.

| Step | Implementation |
|------|---------------|
| Load audio | `pydub.AudioSegment.from_file()` — supports any ffmpeg format including M4A |
| Standardise | Convert to mono, 16 kHz |
| Noise reduction | `noisereduce.reduce_noise()` — spectral subtraction using first 0.5 s as noise profile; `stationary=False`, `prop_decrease=0.75` |
| Loudness normalization | `pydub.effects.normalize()` |
| Export | WAV, written to `output/` |

**Output:** `output/<name>_clean.wav`

---

### Stage 2 — Speaker Diarization (`stages/diarize.py`)

**Goal:** Identify who spoke when.

| Step | Implementation |
|------|---------------|
| Model | `pyannote/speaker-diarization-3.1` via HuggingFace |
| Auth | `HUGGINGFACE_TOKEN` from `.env` |
| Audio input | Pre-loaded in-memory waveform dict `{"waveform": Tensor, "sample_rate": int}` — avoids torchcodec dependency |
| Speaker count | Passed as `num_speakers` int (or `None` for auto-detection) |
| GPU acceleration | Used automatically if `torch.cuda.is_available()` |
| Output unwrapping | pyannote 3.x returns `DiarizeOutput`; `exclusive_speaker_diarization` attribute is the `Annotation` used for iteration |
| Label mapping | `SPEAKER_00` → `Speaker A`, `SPEAKER_01` → `Speaker B`, etc. |
| Per-speaker normalization | Each speaker's segments concatenated and normalized independently |

**Output:** List of segment dicts:
```python
{"start": float, "end": float, "speaker": "Speaker A", "label": "SPEAKER_00"}
```

**Version notes:**
- pyannote 4.0+ requires torch>=2.8.0 (does not exist) — pin `pyannote.audio<4.0`
- huggingface_hub 1.0+ removed `use_auth_token` used internally by pyannote — pin `huggingface_hub<1.0.0`

**pyannote version note:** pyannote.audio 3.x wraps diarization results in a `DiarizeOutput` dataclass rather than returning a `pyannote.core.Annotation` directly. The pipeline resolves the correct annotation object with a fallback chain (see `diarize.py`).

**torchcodec warning:** pyannote emits a `UserWarning` about `torchcodec` on import. This is suppressed because the pipeline uses the in-memory waveform path, which does not require torchcodec.

---

### Stage 3 — Transcription (`stages/transcribe.py`)

**Goal:** Add text to each diarized segment using faster-whisper (CTranslate2-based).

| Step | Implementation |
|------|---------------|
| Model | faster-whisper (`WhisperModel`) — local, offline after first download |
| GPU mode | `device="cuda", compute_type="int8_float16"` — fits in 4 GB VRAM |
| CPU fallback | `device="cpu", compute_type="int8"` |
| Input | Per-segment audio slices from Stage 1 clean WAV, converted to float32 numpy arrays |
| Language | English (`language="en"`) — hardcoded for now |
| Short segment filter | Segments < 500 ms are skipped (prevents hallucinations) |
| Output | `(segments_generator, info)` — generator consumed and joined per segment |

**CUDA teardown fix:** A module-level `_active_model` reference keeps the `WhisperModel` alive until process exit. Without it, ctranslate2's `__del__` calls `exit()` when the model is garbage-collected mid-process on Windows.

**Output:** Segment list extended with `"text"` key per segment.

---

### Stage 4 — Export (`stages/export.py`)

**Goal:** Write human-readable and machine-readable output files.

| Step | Implementation |
|------|---------------|
| Filename | `<source_stem>_<YYYYMMDD_HHMMSS>.txt/.json` — unique per run |
| Metadata | Source file name, context tag, speaker count, ISO 8601 timestamp |
| TXT | `[HH:MM:SS] Speaker X: "..."` per segment, with header comment block |
| JSON | `{"metadata": {...}, "transcript": [...]}` with float timestamps rounded to 3 dp |

---

## Configuration (`config.py`)

| Setting | Env var | Type | Default |
|---------|---------|------|---------|
| `huggingface_token` | `HUGGINGFACE_TOKEN` | str | `""` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | str | `""` |
| `context` | `CONVERSATION_CONTEXT` | str | `"friend"` |
| `num_speakers` | `NUM_SPEAKERS` | int \| None | `None` |
| `whisper_model` | `WHISPER_MODEL` | str | `"medium"` |

---

## System requirements

| Requirement | Notes |
|-------------|-------|
| Python | 3.9+ (tested on 3.11) |
| ffmpeg | Must be on PATH; checked at startup |
| RAM | ~4 GB minimum |
| VRAM | 4 GB sufficient with int8_float16 compute type |
| Disk | ~1.5 GB for faster-whisper medium model cache |
| GPU | Optional; CUDA 12.1 tested on GTX 1650 (sm_75) |
| OS | macOS, Linux, Windows (no Unix-only shell commands) |

### Dependency install order (Windows / CPU-only)

Due to pip resolution behaviour, torch must be installed before other packages or pip will pull in an incompatible torch+numpy combination:

```bash
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

---

## Security / privacy

- All audio processing is **fully local** — no audio is sent to any external service
- faster-whisper runs offline after initial model download
- pyannote model weights downloaded from HuggingFace once, cached locally
- `.env` is gitignored; secrets never committed
- `input/` and `output/` are gitignored; recordings and transcripts never committed

---

## Deferred / out of scope (current version)

- Large file chunking (>160 MB)
- Multi-language support — Whisper `language` hardcoded to `"en"`
- Report generation via Claude API — `ANTHROPIC_API_KEY` reserved for future Stage 5
- Real-time / streaming processing
- Web UI or REST API wrapper
