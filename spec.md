# Technical Specification — Call Analysis Pipeline

## Overview

A local Python pipeline that processes a recorded voice call and produces a clean, structured transcript with speaker attribution. Runs fully locally (no cloud audio processing), GPU-accelerated, with all secrets managed via `.env`.

**Tested: Windows 11, Python 3.11, NVIDIA GeForce GTX 1650 (4 GB VRAM), CUDA 12.1.**

---

## Inputs

| Parameter | Source | Description |
|-----------|--------|-------------|
| `--input FILE` | CLI arg | Path to source audio file (MP3, M4A, WAV, MPEG, or any ffmpeg-supported format) |
| `--context CTX` | CLI / `.env` | Conversation type: `friend`, `work`, `interview`, `date` |
| `--num-speakers N` | CLI / `.env` | Integer speaker count, or omit for auto-detection |
| `--whisper-model SIZE` | CLI / `.env` | Whisper model size: `tiny`, `base`, `small`, `medium` (default), `large` |
| `--transcription-mode MODE` | CLI / `.env` | `accurate` (default) or `fast` |
| `--language LANG` | CLI / `.env` | BCP-47 language code, e.g. `en`, `fr`, `es` (default: `en`) |
| `--dry-run` | CLI flag | Validate config and input without running any stage |
| `--skip-preprocess` | CLI flag | Skip Stage 1; `--input` must be a clean 16 kHz mono WAV |

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
| Load audio | `pydub.AudioSegment.from_file()` — supports any ffmpeg format |
| Standardise | Convert to mono, 16 kHz |
| Noise reduction | `noisereduce.reduce_noise()` — spectral subtraction using first 0.5 s as noise profile; `stationary=False`, `prop_decrease=0.75` |
| Loudness normalization | `pydub.effects.normalize()` |
| Export | WAV, written to `output/` |

**Output:** `output/<name>_clean.wav`

Can be skipped with `--skip-preprocess` when a clean WAV already exists.

---

### Stage 2 — Speaker Diarization (`stages/diarize.py`)

**Goal:** Identify who spoke when.

| Step | Implementation |
|------|---------------|
| Model | `pyannote/speaker-diarization-3.1` via HuggingFace |
| Auth | `huggingface_hub.login(token=...)` before `Pipeline.from_pretrained()` |
| Audio input | Pre-loaded in-memory dict `{"waveform": Tensor, "sample_rate": int}` — avoids torchcodec dependency |
| Speaker count | Passed as `num_speakers` int (or `None` for auto-detection) |
| GPU acceleration | `pipeline.to(torch.device("cuda"))` if `torch.cuda.is_available()` |
| Output unwrapping | pyannote 3.x returns `DiarizeOutput`; resolved via `itertracks` → `exclusive_speaker_diarization` fallback chain |
| Label mapping | `SPEAKER_00` → `Speaker A`, `SPEAKER_01` → `Speaker B`, etc. |
| Per-speaker normalization | Each speaker's segments concatenated and normalized independently |

**Output:** List of segment dicts:
```python
{"start": float, "end": float, "speaker": "Speaker A", "label": "SPEAKER_00"}
```

**Version notes:**
- pyannote 4.0+ requires torch>=2.8.0 (does not exist) — pin `pyannote.audio<4.0`
- huggingface_hub 1.0+ removed `use_auth_token` used internally by pyannote — pin `huggingface_hub<1.0.0`

---

### Stage 3 — Transcription (`stages/transcribe.py`)

**Goal:** Add text to each diarized segment using faster-whisper (CTranslate2-based).

Two modes controlled by `TRANSCRIPTION_MODE` / `--transcription-mode`:

#### accurate (default)

| Step | Implementation |
|------|---------------|
| Model | `WhisperModel` — local, offline after first download |
| GPU mode | `device="cuda", compute_type="int8_float16"` — fits in 4 GB VRAM |
| CPU fallback | `device="cpu", compute_type="int8"` |
| Per-segment | One `model.transcribe()` call per diarization segment |
| Short segment filter | Segments < 500 ms are skipped (prevents hallucinations) |
| Language | Configurable via `WHISPER_LANGUAGE` (default: `en`) |

Produces fine-grained sentence-level output. ~1x real-time on GTX 1650 for a 10-minute file.

#### fast

Same model and device setup, but transcribes the entire WAV in one call, then assigns each Whisper segment to a speaker via max-overlap with diarization windows. Consecutive same-speaker Whisper segments are merged. ~20% faster than accurate for long files, but produces fewer output lines (Whisper's ~30s internal chunking limits granularity).

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

| Setting | Env var | CLI flag | Default |
|---------|---------|----------|---------|
| `huggingface_token` | `HUGGINGFACE_TOKEN` | — | `""` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | — | `""` |
| `context` | `CONVERSATION_CONTEXT` | `--context` | `"friend"` |
| `num_speakers` | `NUM_SPEAKERS` | `--num-speakers` | `None` (auto) |
| `whisper_model` | `WHISPER_MODEL` | `--whisper-model` | `"medium"` |
| `transcription_mode` | `TRANSCRIPTION_MODE` | `--transcription-mode` | `"accurate"` |
| `whisper_language` | `WHISPER_LANGUAGE` | `--language` | `"en"` |

---

## Error handling

Each stage is wrapped in `try/except` in `main.py`. On failure:
- Prints `[error] Stage N (name) failed: <message>`
- Exits with code 1
- Stage 2 `EnvironmentError` (missing token) is caught separately with a more specific message

`--dry-run` exits cleanly after config validation without executing any stage.

---

## Performance (GTX 1650, accurate mode)

| Audio length | Stage 2 | Stage 3 | Total |
|-------------|---------|---------|-------|
| 2:01 | ~82s | ~76s | ~2:41 |
| 10:56 | ~56s | ~624s | ~11:27 |

Stage 2 scales sublinearly (pyannote batches better on longer audio). Stage 3 scales linearly with segment count (~4.7s/segment on GTX 1650). Overall ratio is ~1x real-time for longer files.

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

### Dependency install order

```bash
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

### Key version constraints

| Package | Constraint | Reason |
|---------|-----------|--------|
| `torch` | `==2.1.0+cu121` | Newer torch pulls in numpy 2.x |
| `numpy` | `<2.0` | pyannote compiled against numpy 1.x |
| `pyannote.audio` | `<4.0` | 4.0.4 requires torch>=2.8.0 (doesn't exist) |
| `huggingface_hub` | `<1.0.0` | 1.x removed `use_auth_token` used by pyannote 3.x internals |

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
- Report generation via Claude API — `ANTHROPIC_API_KEY` reserved for future Stage 5
- Real-time / streaming processing
- Web UI or REST API wrapper
