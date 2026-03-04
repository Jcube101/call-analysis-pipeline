# Technical Specification — Call Analysis Pipeline

## Overview

A local Python pipeline that processes a recorded two-person phone or voice call and produces a clean, structured transcript with speaker attribution. Designed to be run locally (no cloud audio processing), with all secrets managed via `.env`.

---

## Inputs

| Parameter | Source | Description |
|-----------|--------|-------------|
| `--input FILE` | CLI arg | Path to source audio file (MP3, WAV, M4A, or any ffmpeg-supported format) |
| `--context CTX` | CLI / `.env` | Conversation type: `friend`, `work`, `interview`, `date` |
| `--num-speakers N` | CLI / `.env` | Integer speaker count, or omit for auto-detection |
| `--whisper-model SIZE` | CLI / `.env` | Whisper model size: `tiny`, `base`, `small`, `medium` (default), `large` |

CLI flags override `.env` values when both are present.

---

## Outputs

All outputs land in `output/`:

| File | Format | Description |
|------|--------|-------------|
| `<name>_clean.wav` | WAV, mono, 16 kHz | Noise-reduced, normalized audio |
| `transcript.txt` | Plain text | Human-readable labeled transcript |
| `transcript.json` | JSON | Structured transcript with metadata header |

### transcript.txt format

```
# Call Transcript
# Source:  test_call.mp3
# Context: friend
# Speakers: 2
# Processed: 2024-01-15T14:30:00

[00:00:04] Speaker A: "Hey, how are you doing..."
[00:01:12] Speaker B: "I'm good, just got back from..."
```

### transcript.json schema

```json
{
  "metadata": {
    "source_file": "test_call.mp3",
    "context": "friend",
    "num_speakers": 2,
    "processed_at": "2024-01-15T14:30:00"
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

---

### Stage 2 — Speaker Diarization (`stages/diarize.py`)

**Goal:** Identify who spoke when.

| Step | Implementation |
|------|---------------|
| Model | `pyannote/speaker-diarization-3.1` via HuggingFace |
| Auth | `HUGGINGFACE_TOKEN` from `.env` |
| Speaker count | Passed as `num_speakers` int (or `None` for auto-detection) |
| GPU acceleration | Used automatically if `torch.cuda.is_available()` |
| Label mapping | `SPEAKER_00` → `Speaker A`, `SPEAKER_01` → `Speaker B`, etc. |
| Per-speaker normalization | Each speaker's segments concatenated and normalized independently |

**Output:** List of segment dicts:
```python
{"start": float, "end": float, "speaker": "Speaker A", "label": "SPEAKER_00"}
```

**Prerequisites:** HuggingFace account with accepted license at `huggingface.co/pyannote/speaker-diarization-3.1`.

---

### Stage 3 — Transcription (`stages/transcribe.py`)

**Goal:** Add text to each diarized segment.

| Step | Implementation |
|------|---------------|
| Model | OpenAI Whisper (open-source, runs locally) |
| Model size | Configurable via `WHISPER_MODEL` env var (default: `medium`) |
| Input | Per-segment audio slices from Stage 1 clean WAV |
| Language | English (`language="en"`) — hardcoded for now |
| Short segment filter | Segments < 500 ms are skipped (prevents Whisper hallucinations) |
| FP16 | Disabled (`fp16=False`) for CPU compatibility |

**Output:** Segment list extended with `"text"` key per segment.

**First-run note:** Whisper `medium` model (~1.5 GB) downloads to Whisper's default cache on first run.

---

### Stage 4 — Export (`stages/export.py`)

**Goal:** Write human-readable and machine-readable output files.

| Step | Implementation |
|------|---------------|
| Metadata | Source file name, context tag, speaker count, ISO 8601 timestamp |
| TXT | `[HH:MM:SS] Speaker X: "..."` per segment, with header comment block |
| JSON | `{"metadata": {...}, "transcript": [...]}` with float timestamps rounded to 3 dp |

---

## Configuration (`config.py`)

`Settings` is a dataclass that reads from `.env` via `python-dotenv`. A singleton `settings` object is imported by all modules.

| Setting | Env var | Type | Default |
|---------|---------|------|---------|
| `huggingface_token` | `HUGGINGFACE_TOKEN` | str | `""` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | str | `""` |
| `context` | `CONVERSATION_CONTEXT` | str | `"friend"` |
| `num_speakers` | `NUM_SPEAKERS` | int \| None | `None` |
| `whisper_model` | `WHISPER_MODEL` | str | `"medium"` |

`settings.override()` applies CLI argument values on top of `.env` values.
`settings.validate_for_diarization()` raises `EnvironmentError` if the HF token is missing.

---

## System requirements

| Requirement | Notes |
|-------------|-------|
| Python | 3.9+ |
| ffmpeg | Must be on PATH; checked at startup |
| RAM | ~4 GB minimum; Whisper `medium` needs ~3 GB |
| Disk | ~2 GB for Whisper model cache |
| GPU | Optional; used automatically by pyannote if CUDA is available |
| OS | macOS, Linux, Windows (no Unix-only shell commands) |

---

## Security / privacy

- All audio processing is **fully local** — no audio is sent to any external service
- Whisper runs offline after the initial model download
- pyannote model weights are downloaded from HuggingFace once and cached locally
- `.env` is gitignored; secrets are never committed
- `input/` and `output/` are gitignored; recordings and transcripts are never committed

---

## Deferred / out of scope (current version)

- Large file chunking (>160 MB) — architecture supports it, not yet implemented
- Multi-language support — Whisper `language` is hardcoded to `"en"`
- Report generation via Claude API — `ANTHROPIC_API_KEY` is reserved for a future Stage 5
- Real-time / streaming processing
- Web UI or REST API wrapper
