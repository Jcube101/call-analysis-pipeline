# Technical Specification — Call Analysis Pipeline

## Overview

A local Python pipeline that processes a recorded voice call and produces a clean, structured transcript with speaker attribution and an optional AI-generated analysis report. Audio processing runs fully locally (no cloud audio processing), GPU-accelerated, with all secrets managed via `.env`. Report generation uses the Gemini API.

**Tested and working on Windows 11, Python 3.11, CUDA 12.1, GTX 1650.**

---

## Inputs

| Parameter | Source | Description |
|-----------|--------|-------------|
| `--input FILE` | CLI arg | Path to source audio file (MP3, M4A, WAV, MPEG, or any ffmpeg-supported format). Required unless `--from-json` is used. |
| `--from-json FILE` | CLI arg | Skip Stages 1–4; generate a Gemini report from an existing transcript JSON (implies `--report`) |
| `--context CTX` | CLI / `.env` | Conversation type: `friend`, `work`, `interview`, `date`, `public_interview` |
| `--num-speakers N` | CLI / `.env` | Integer speaker count, or omit for auto-detection |
| `--whisper-model SIZE` | CLI / `.env` | Whisper model size: `tiny`, `base`, `small`, `medium` (default), `large` |
| `--transcription-mode MODE` | CLI / `.env` | `accurate` (default) or `fast` |
| `--language LANG` | CLI / `.env` | BCP-47 language code, e.g. `en`, `fr`, `es` (default: `en`) |
| `--speaker-names NAMES` | CLI / `.env` | Comma-separated real names to replace generic labels, e.g. `Alice,Bob` |
| `--word-timestamps` | CLI / `WORD_TIMESTAMPS=true` | Include per-word start/end times and probabilities in JSON output |
| `--dry-run` | CLI flag | Validate config and input without running any stage |
| `--skip-preprocess` | CLI flag | Skip Stage 1; `--input` must be a clean 16 kHz mono WAV |
| `--report` | CLI flag | Run Stage 5: generate an analysis report via Gemini API |

CLI flags override `.env` values when both are present.

---

## Outputs

All outputs land in `output/`. Each run produces uniquely named transcript files:

| File | Format | Description |
|------|--------|-------------|
| `<name>_clean.wav` | WAV, mono, 16 kHz | Noise-reduced, normalized audio (overwritten each run) |
| `<name>_<YYYYMMDD_HHMMSS>.txt` | Plain text | Human-readable labeled transcript |
| `<name>_<YYYYMMDD_HHMMSS>.json` | JSON | Structured transcript with metadata header |
| `<name>_<YYYYMMDD_HHMMSS>_report.md` | Markdown | AI analysis report (`--report` only) |

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
    "processed_at": "2026-03-27T14:30:22",
    "speaker_names": ["Alice", "Bob"]
  },
  "transcript": [
    {
      "start": 4.2,
      "end": 9.8,
      "speaker": "Speaker A",
      "text": "Hey, how are you doing...",
      "confidence": 0.91,
      "words": [
        {"word": "Hey,", "start": 4.2, "end": 4.5, "probability": 0.99}
      ]
    }
  ]
}
```

- `speaker_names` in metadata: only present when `--speaker-names` was used
- `confidence`: always present (0–1, duration-weighted `exp(avg_logprob)`)
- `words`: only present when `--word-timestamps` was passed

When `--from-json --speaker-names` is used, a `<stem>_named.json` is written alongside the original with updated speaker labels and `relabelled_at` added to metadata.

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
| Auth | `HUGGINGFACE_TOKEN` from `.env` |
| Audio input | Pre-loaded in-memory waveform dict `{"waveform": Tensor, "sample_rate": int}` — avoids torchcodec dependency |
| Speaker count | Passed as `num_speakers` int (or `None` for auto-detection) |
| GPU acceleration | Used automatically if `torch.cuda.is_available()` |
| Output unwrapping | pyannote 3.x returns `DiarizeOutput`; `exclusive_speaker_diarization` attribute is the `Annotation` used for iteration |
| Re-identification | MFCC + delta features (librosa, 20 coefficients) per segment ≥0.5 s, mean-pooled over time; KMeans clustering (`n_clusters=num_speakers`) with L2-normalised feature vectors; cluster IDs remapped to `SPEAKER_XX` in first-appearance order; segments <0.5 s inherit the label of their nearest longer segment |
| Label mapping | `SPEAKER_00` → `Speaker A`, `SPEAKER_01` → `Speaker B`, etc. (applied after re-identification) |

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

Same model and device setup. Merges consecutive same-speaker diarization segments into "turns" (gap ≤ 1 s), then makes one `model.transcribe()` call per turn. Produces one output segment per speaker turn — 10-20x fewer Whisper calls than accurate mode on long recordings, while preserving speaker-accurate boundaries (turns are built from diarization, not from Whisper's internal chunking).

**CUDA teardown fix:** A module-level `_active_model` reference keeps the `WhisperModel` alive until process exit. Without it, ctranslate2's `__del__` calls `exit()` when the model is garbage-collected mid-process on Windows.

**Output:** Segment list extended with `"text"`, `"confidence"`, and optionally `"words"` per segment.

---

### Stage 4 — Export (`stages/export.py`)

**Goal:** Write human-readable and machine-readable output files.

| Step | Implementation |
|------|---------------|
| Filename | `<source_stem>_<YYYYMMDD_HHMMSS>.txt/.json` — unique per run |
| Metadata | Source file name, context tag, speaker count, ISO 8601 timestamp; `speaker_names` included if provided |
| TXT | `[HH:MM:SS] Speaker X: "..."` per segment, with header comment block |
| JSON | `{"metadata": {...}, "transcript": [...]}` with timestamps rounded to 3 dp; `confidence` always present; `words` present when `--word-timestamps` was passed |
| Relabelling | `write_relabelled()` writes `<stem>_named.json` when `--from-json --speaker-names` is used |

---

### Stage 5 — Analysis Report (`stages/report.py`)

**Goal:** Generate a context-aware analysis report from the transcript using the Gemini API.

Only runs when `--report` is passed. `GEMINI_API_KEY` is validated early (before Stages 1–4 begin) so the run fails fast if the key is missing.

| Step | Implementation |
|------|---------------|
| API | `google-genai` SDK, `gemini-3-flash-preview` model |
| Prompt | Loaded from `prompts/<context>.md`; falls back to built-in defaults |
| Metadata | Source file, context, speaker count, audio duration, per-speaker segment counts |
| Chunking | Transcripts >500k chars split into overlapping chunks; partial reports synthesised in a second Gemini call |
| Output | `output/<name>_<YYYYMMDD_HHMMSS>_report.md` |
| Terminal preview | First 20 lines of the report are printed after Stage 5 completes |

**Prompt customisation:** Edit `prompts/<context>.md` to change what Gemini focuses on for each conversation type. The four built-in contexts are `friend`, `work`, `interview`, and `date`. Each prompt defines specific output sections and includes a speaker label reliability warning — pyannote's automatic diarisation can flip speaker labels on long recordings, so prompts instruct Gemini to base analysis on content rather than assuming label consistency.

**`--from-json` mode:** `main.py` supports `python main.py --from-json output/<name>.json` to run Stage 5 on an existing transcript without re-running Stages 1–4. Context and speaker count are read from the JSON metadata and can be overridden with `--context`.

---

## Configuration (`config.py`)

| Setting | Env var | CLI flag | Default |
|---------|---------|----------|---------|
| `huggingface_token` | `HUGGINGFACE_TOKEN` | — | `""` |
| `gemini_api_key` | `GEMINI_API_KEY` | — | `""` |
| `context` | `CONVERSATION_CONTEXT` | `--context` | `"friend"` |
| `num_speakers` | `NUM_SPEAKERS` | `--num-speakers` | `None` (auto) |
| `whisper_model` | `WHISPER_MODEL` | `--whisper-model` | `"medium"` |
| `transcription_mode` | `TRANSCRIPTION_MODE` | `--transcription-mode` | `"accurate"` |
| `whisper_language` | `WHISPER_LANGUAGE` | `--language` | `"en"` |
| `speaker_names` | `SPEAKER_NAMES` | `--speaker-names` | `[]` (generic) |
| `word_timestamps` | `WORD_TIMESTAMPS` | `--word-timestamps` | `False` |

---

## Error handling

Each stage is wrapped in `try/except` in `main.py`. On failure:
- Prints `[error] Stage N (name) failed: <message>`
- Exits with code 1
- Stage 2 `EnvironmentError` (missing token) is caught separately with a more specific message
- Stage 5 `EnvironmentError` (missing Gemini key) is caught and reported before any stage runs

`--dry-run` exits cleanly after config validation without executing any stage.

---

## Performance (GTX 1650, accurate mode)

| Audio length | Stage 2 | Stage 3 | Total (no report) |
|-------------|---------|---------|-------|
| 2:01 | ~82s | ~76s | ~2:41 |
| 10:56 | ~56s | ~624s | ~11:27 |

Stage 5 (Gemini report) adds ~4–10s depending on transcript length and API latency.

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
| Internet | Required for Stage 5 (Gemini API calls) |

### Dependency install order (Windows / CPU-only)

Due to pip resolution behaviour, torch must be installed before other packages or pip will pull in an incompatible torch+numpy combination:

```bash
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
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
| `google-genai` | `>=1.0.0` | Replaces deprecated `google-generativeai` package |

---

## Security / privacy

- All audio processing is **fully local** — no audio is sent to any external service
- faster-whisper runs offline after initial model download
- pyannote model weights downloaded from HuggingFace once, cached locally
- Stage 5 sends transcript text to the Gemini API — do not use `--report` if the recording is confidential
- `.env` is gitignored; secrets never committed
- `input/` and `output/` are gitignored; recordings and transcripts never committed

---

## Deferred / out of scope (current version)

- Real-time / streaming processing
- Batch processing (`--input-dir`)
- Docker packaging
