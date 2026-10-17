# Technical Specification — Call Analysis Pipeline

## Overview

A local Python pipeline that processes a recorded voice call and produces a clean, structured transcript with speaker attribution and an optional AI-generated analysis report. Audio processing runs fully locally (no cloud audio processing), GPU-accelerated, with all secrets managed via `.env`. Report generation uses the Claude API by default (Anthropic), with Gemini API as an alternative.

**Tested and working on Windows 11, Python 3.12, torch 2.5.1+cu124 (NVIDIA driver 581.15), GTX 1650.**

---

## Inputs

| Parameter | Source | Description |
|-----------|--------|-------------|
| `--input FILE` | CLI arg | Path to source audio file (MP3, M4A, WAV, MPEG, or any ffmpeg-supported format). Required unless `--from-json` is used. |
| `--from-json FILE` | CLI arg | Skip Stages 1–4; generate a report from an existing transcript JSON (implies `--report`) |
| `--context CTX` | CLI / `.env` | Conversation type: `friend`, `work`, `work_interview`, `user_interview`, `public_interview`, `date` |
| `--num-speakers N` | CLI / `.env` | Integer speaker count, or omit for auto-detection |
| `--whisper-model SIZE` | CLI / `.env` | Whisper model size: `tiny`, `base`, `small`, `medium` (default), `large` |
| `--transcription-mode MODE` | CLI / `.env` | `accurate` (default) or `fast` |
| `--language LANG` | CLI / `.env` | BCP-47 language code, e.g. `en`, `fr`, `es` (default: `en`) |
| `--speaker-names NAMES` | CLI / `.env` | Comma-separated real names to replace generic labels, e.g. `Alice,Bob` |
| `--word-timestamps` | CLI / `WORD_TIMESTAMPS=true` | Include per-word start/end times and probabilities in JSON output |
| `--dry-run` | CLI flag | Validate config and input without running any stage |
| `--skip-preprocess` | CLI flag | Skip Stage 1; `--input` must be a clean 16 kHz mono WAV |
| `--report` | CLI flag | Run Stage 5: generate an analysis report via the Claude or Gemini API |
| `--report-model MODEL` | CLI arg | Stage 5 model: `claude-haiku-5-5` (default), `claude-sonnet-5-5`, `gemini-3.8-flash`, `gemini-4-argon`. Legacy IDs are remapped. |

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

When the run has warnings (see Stage 2 re-identification), each one adds a `# Warning: <text>` line to the header after `# Processed:`. Clean runs have none.

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
- `warnings` in metadata: a list of short strings, only present when non-empty (Stage 2 re-identification was `partial` or `failed`, or a fixed speaker count came back short)

When `--from-json --speaker-names` is used, a `<stem>_named.json` is written alongside the original with updated speaker labels and `relabelled_at` added to metadata.

When produced via `api.py`, `"job_id"` is inserted as the first key in `metadata`:
```json
{
  "metadata": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "source_file": "call.m4a",
    ...
  }
}
```
This allows `POST /report-from-json` to find the correct job folder when the JSON is re-uploaded.

---

## Pipeline stages

### Stage 1 — Audio Pre-processing (`stages/preprocess.py`)

**Goal:** Produce a clean, standardised WAV that downstream stages can rely on.

| Step | Implementation |
|------|---------------|
| Load + standardise | One streaming `ffmpeg` pass decodes any ffmpeg format straight to mono, 16 kHz, 16-bit PCM on disk |
| Noise reduction | `noisereduce.reduce_noise()` — spectral subtraction using first 0.5 s as noise profile; `stationary=False`, `prop_decrease=0.75`; 60 s chunks with 0.5 s overlap, streamed from file to file |
| Loudness normalization | `_normalize_in_place()` — a block-wise second pass matching `pydub.effects.normalize`'s 0.1 dB headroom |
| Progress | Phases `decode`, `noise reduction` (counted), `normalization`; tqdm bar over noise reduction chunks |
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
| Audio input | Clean WAV **file path** — pyannote reads it through torchaudio's soundfile backend, so `Audio.crop()` seeks each window from disk (see CLAUDE.md, "Audio input to pyannote") |
| Speaker count | Passed as `num_speakers` int (or `None` for auto-detection) |
| GPU acceleration | Used automatically if `torch.cuda.is_available()` |
| Clustering cap | `max_num_embeddings = 1000` (`MAX_CLUSTERING_EMBEDDINGS`); the 3.1 config disables it, which made clustering O(N²) in memory on long files (KNOWN-ISSUES.md §1) |
| Seeded subsample | Above the cap pyannote subsamples with an unseeded `random.shuffle()`; `random` is seeded with `CLUSTERING_SUBSAMPLE_SEED` around the call (state saved and restored) so repeat runs give identical segments |
| Output unwrapping | pyannote 3.x returns `DiarizeOutput`; `exclusive_speaker_diarization` attribute is the `Annotation` used for iteration |
| Re-identification | MFCC + delta features (librosa, 20 coefficients) per segment ≥0.5 s, mean-pooled over time; KMeans clustering (`n_clusters=num_speakers`) with L2-normalised feature vectors; cluster IDs remapped to `SPEAKER_XX` in first-appearance order; segments <0.5 s inherit the label of their nearest longer segment |
| Re-identification outcome | Reported through `run(on_reidentification=...)` as a dict whose shape lives in `stages/reid_outcome.py`: `ok` (every segment it tried was analysed), `partial` (clustering ran, but at least 1 segment — `PARTIAL_MIN_FAILED_SEGMENTS` — failed feature extraction and took its nearest neighbour's label), or `failed` (labels are pyannote's: scikit-learn missing, an exception, or too few usable segments). New labels are written back all at once or not at all. `warnings_for()` turns the outcome into warning strings; a fixed `NUM_SPEAKERS` that comes back short also warns, auto-detect does not |
| Label mapping | `SPEAKER_00` → `Speaker A`, `SPEAKER_01` → `Speaker B`, etc. (applied after re-identification) |
| Progress | Phases `model load`, `segmentation` (counted), `speaker counting`, `embeddings` (counted), `clustering`, `re-identification`; one plain line per phase plus a tqdm bar (redrawn at most every 5 s when stdout is not a TTY) |

**Output:** List of segment dicts:
```python
{"start": float, "end": float, "speaker": "Speaker A", "label": "SPEAKER_00"}
```

**Version notes:**
- pyannote 4.0+ requires torch>=2.8.0 (does not exist) — pin `pyannote.audio<4.0`
- huggingface_hub 1.0+ removed `use_auth_token` used internally by pyannote — pin `huggingface_hub<1.0.0`

**pyannote version note:** pyannote.audio 3.x wraps diarization results in a `DiarizeOutput` dataclass rather than returning a `pyannote.core.Annotation` directly. The pipeline resolves the correct annotation object with a fallback chain (see `diarize.py`).

**torchcodec warning:** pyannote emits a `UserWarning` about `torchcodec` on import whatever the input form. It is suppressed because torchcodec is never needed: torchaudio uses the soundfile backend here.

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
| Progress | Phases `model load`, `transcription` (counted); tqdm bar per segment (accurate) or per turn (fast) |

Produces fine-grained sentence-level output. ~1x real-time on GTX 1650 for a 10-minute file.

#### fast

Same model and device setup. Merges consecutive same-speaker diarization segments into "turns" (gap ≤ 1 s), then makes one `model.transcribe()` call per turn. Produces one output segment per speaker turn — 10-20x fewer Whisper calls than accurate mode on long recordings, while preserving speaker-accurate boundaries (turns are built from diarization, not from Whisper's internal chunking).

**Model lifetime:** `run()` keeps the `WhisperModel` in the module-level `_active_model`. The API server frees it after each job with `release_model()` (off with `RELEASE_WHISPER_AFTER_JOB=0`); the CLI leaves it to process exit. It used to be held until exit because of a March 2026 ctranslate2 teardown crash that no longer reproduces on the pinned versions — see KNOWN-ISSUES.md §2.

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
| Warnings | `run(warnings=...)` writes `metadata.warnings` and `# Warning:` header lines only when the list is non-empty, so clean runs are unchanged; `clean_warnings()` sanitises warnings read back from an uploaded JSON (malformed values are dropped) |
| Relabelling | `write_relabelled()` writes `<stem>_named.json` when `--from-json --speaker-names` is used |

---

### Stage 5 — Analysis Report (`stages/report.py`)

**Goal:** Generate a context-aware analysis report from the transcript using an LLM API.

Only runs when `--report` is passed. The appropriate API key is validated early (before Stages 1–4 begin) so the run fails fast if the key is missing.

Supports both Anthropic Claude (`claude-*` models) and Google Gemini (`gemini-*` models). Model IDs, labels, providers, the default, fallbacks and legacy aliases live in `models.py`.

| Step | Implementation |
|------|---------------|
| Default API | `anthropic` SDK, `claude-haiku-5-5` model (`claude-sonnet-5-5` also available) |
| Alternative API | `google-genai` SDK, `gemini-3.8-flash` or `gemini-4-argon` |
| Prompt | Loaded from `prompts/<context>.md`; falls back to built-in defaults |
| Metadata | Source file, context, speaker count, audio duration, per-speaker segment counts |
| Context hints | Optional `context_hints` parameter prepended to system prompt for correct spelling of names, places, and terms |
| Chunking | Transcripts >500k chars split into overlapping chunks; partial reports synthesised in a second API call |
| Output | `output/<name>_<YYYYMMDD_HHMMSS>_report.md` — header includes source, context, speakers, actual model used, and timestamp |
| Terminal preview | First 20 lines of the report are printed after Stage 5 completes |
| Fallback | Claude models fall back to `claude-haiku-5-5` on failure; Gemini models fall back to `gemini-3.8-flash` on 503 |
| Truncation | A response stopped by the output token limit raises an error rather than writing a truncated report |
| Warnings | `run(warnings=...)` adds `Warning:` lines to the metadata block the model sees and `**Warning:** ...` lines to the report header after `**Generated:**`; with no warnings both are unchanged. `--from-json` and `/report-from-json` take them from the uploaded file's `metadata.warnings` |

**Prompt customisation:** Edit `prompts/<context>.md` to change what the LLM focuses on for each conversation type. The six built-in contexts are `friend`, `work`, `work_interview`, `user_interview`, `public_interview`, and `date`. Each prompt defines specific output sections and includes a speaker label reliability warning — pyannote's automatic diarisation can flip speaker labels on long recordings, so prompts instruct the model to base analysis on content rather than assuming label consistency.

**`--from-json` mode:** `main.py` supports `python main.py --from-json output/<name>.json` to run Stage 5 on an existing transcript without re-running Stages 1–4. Context and speaker count are read from the JSON metadata and can be overridden with `--context`.

---

## Configuration (`config.py`)

| Setting | Env var | CLI flag | Default |
|---------|---------|----------|---------|
| `huggingface_token` | `HUGGINGFACE_TOKEN` | — | `""` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | — | `""` |
| `gemini_api_key` | `GEMINI_API_KEY` | — | `""` |
| `context` | `CONVERSATION_CONTEXT` | `--context` | `"friend"` |
| `num_speakers` | `NUM_SPEAKERS` | `--num-speakers` | `None` (auto) |
| `whisper_model` | `WHISPER_MODEL` | `--whisper-model` | `"medium"` |
| `transcription_mode` | `TRANSCRIPTION_MODE` | `--transcription-mode` | `"accurate"` |
| `whisper_language` | `WHISPER_LANGUAGE` | `--language` | `"en"` |
| `speaker_names` | `SPEAKER_NAMES` | `--speaker-names` | `[]` (generic) |
| `word_timestamps` | `WORD_TIMESTAMPS` | `--word-timestamps` | `False` |
| `release_whisper_after_job` | `RELEASE_WHISPER_AFTER_JOB` | — (API server only) | `True` |

---

## Error handling

Each stage is wrapped in `try/except` in `main.py`. On failure:
- Prints `[error] Stage N (name) failed: <message>`
- Exits with code 1
- Stage 2 `EnvironmentError` (missing token) is caught separately with a more specific message
- Stage 5 `EnvironmentError` (missing API key for the selected model) is caught and reported before any stage runs

`--dry-run` exits cleanly after config validation without executing any stage.

---

## Performance (GTX 1650, accurate mode)

| Audio length | Stage 2 | Stage 3 | Total (no report) |
|-------------|---------|---------|-------|
| 2:01 | ~82s | ~76s | ~2:41 |
| 10:56 | ~56s | ~624s | ~11:27 |

Stage 5 (AI report) adds ~4–10s depending on transcript length, model choice, and API latency.

---

## System requirements

| Requirement | Notes |
|-------------|-------|
| Python | 3.9+ (tested on 3.12) |
| ffmpeg | Must be on PATH; checked at startup |
| RAM | ~4 GB minimum |
| VRAM | 4 GB sufficient with int8_float16 compute type |
| Disk | ~1.5 GB for faster-whisper medium model cache |
| GPU | Optional; torch 2.5.1+cu124 tested on GTX 1650 (sm_75), NVIDIA driver 581.15 (CUDA 13.0) |
| OS | macOS, Linux, Windows (no Unix-only shell commands) |
| Internet | Required for Stage 5 (Claude or Gemini API calls) |

### Dependency install order (Windows / CPU-only)

Due to pip resolution behaviour, torch must be installed before other packages or pip will pull in an incompatible torch+numpy combination:

```bash
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

### Key version constraints

| Package | Constraint | Reason |
|---------|-----------|--------|
| `torch` | `==2.5.1+cu124` | Newer torch pulls in numpy 2.x; cu124 wheels run on CUDA 13.x drivers |
| `numpy` | `<2.0` | pyannote compiled against numpy 1.x |
| `pyannote.audio` | `<4.0` | 4.0.4 requires torch>=2.8.0 (doesn't exist) |
| `huggingface_hub` | `<1.0.0` | 1.x removed `use_auth_token` used by pyannote 3.x internals |
| `google-genai` | `>=1.0.0` | Replaces deprecated `google-generativeai` package |
| `anthropic` | `>=0.40.0` | Claude API for Stage 5 report generation (default provider) |
| `ctranslate2` / `faster-whisper` | `==4.8.2` / `==1.2.1` | Freeing the Whisper model after each API job was only verified on these versions (KNOWN-ISSUES.md §2) |

---

## Security / privacy

- All audio processing is **fully local** — no audio is sent to any external service
- faster-whisper runs offline after initial model download
- pyannote model weights downloaded from HuggingFace once, cached locally
- Stage 5 sends transcript text to the Claude or Gemini API — do not use `--report` if the recording is confidential
- `.env` is gitignored; secrets never committed
- `input/` and `output/` are gitignored; recordings and transcripts never committed

---

## REST API wrapper (api.py)

`api.py` exposes the full pipeline over HTTP + WebSocket. It is separate from `main.py` and has no effect on CLI usage. Production-ready for local use; frontend at job-joseph.com/projects/call-analysis connects via ngrok tunnel.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| POST | `/analyse` | Multipart upload (audio file + form params, incl. `analysis_model`); starts pipeline; returns `{"job_id": "..."}` |
| POST | `/report-from-json` | Multipart upload (transcript JSON + form params, incl. `analysis_model`); runs Stage 5 only; returns `{"job_id": "..."}` |
| GET | `/status/{job_id}` | Returns the job dict as is, including `status`, the current stage, `warnings` and `reid` (raw Stage 2 outcome, or `null`) |
| GET | `/reconnect/{job_id}` | Returns full job state: transcript, report, metadata, status, `warnings` |
| GET | `/download/{job_id}/transcript` | `.txt` transcript file |
| GET | `/download/{job_id}/json` | `transcript_named.json` if present, else timestamped `.json` |
| GET | `/download/{job_id}/report` | `*_report.md` file (`text/markdown`) |
| GET | `/download/{job_id}/wav` | `*_clean.wav` file (`audio/wav`) |
| WS | `/ws/{job_id}` | WebSocket — progress messages, final `complete` payload |

Download endpoints serve files directly from disk without checking job status — files are returned as soon as they exist.

**Model field:** both POST endpoints take `analysis_model` (allowed values in `models.py`; legacy IDs remapped, unknown values fall back to the default). The old `gemini_model` alias was removed on 2026-10-14: if a request still sends it, the value is ignored (default model unless `analysis_model` is also sent), the request still succeeds, and the server prints `[models] Ignored removed field 'gemini_model'; send 'analysis_model'`. `GET /status` no longer echoes a `gemini_model` mirror.

### Full pipeline flow (frontend)

1. User uploads audio at job-joseph.com/projects/call-analysis
2. Frontend POSTs to `/analyse` with `generate_report=true` (if selected) → `job_id` returned
3. Frontend opens `WS /ws/{job_id}` → receives progress messages per stage
4. Pipeline completes → `{"type": "complete", "transcript": [...], "report": "...", "metadata": {...}, "warnings": [...]}` sent (`warnings` is empty for a clean run)
5. Frontend displays transcript + rendered report; download buttons call `/download/{job_id}/{type}`
6. If WS drops (ngrok timeout), frontend calls `GET /reconnect/{job_id}` → full state recovered

### Report-from-JSON flow (frontend)

1. User uploads existing `.json` transcript + optional speaker names
2. Frontend POSTs to `/report-from-json` → `job_id` returned
3. Stage 5 runs; heartbeat keeps WS alive during the LLM API call (30–60 s)
4. `complete` message delivers rendered report; download button calls `/download/{job_id}/report`

### Output structure (API mode)

All output files use the uploaded filename stem (`input`) plus a timestamp:
```
output/jobs/{job_id}/
├── input.{ext}                        — uploaded audio
├── input_clean.wav                    — Stage 1 output
├── input_{YYYYMMDD_HHMMSS}.txt        — Stage 4 transcript
├── input_{YYYYMMDD_HHMMSS}.json       — Stage 4 JSON (includes job_id in metadata)
├── transcript_named.json              — relabelled JSON (report-from-json with speaker names)
└── input_{YYYYMMDD_HHMMSS}_report.md  — Stage 5 report
```

### Threading and concurrency

- `ThreadPoolExecutor(max_workers=1)` — one pipeline job at a time (GPU constraint)
- asyncio event loop captured at startup; background threads push WS messages via `asyncio.run_coroutine_threadsafe(...).result(timeout=5)`
- All WS messages stored in `job["message_queue"]` — replayed to clients that reconnect
- `threading.excepthook` installed to prevent thread crashes killing uvicorn
- Progress messages are queued on the pipeline thread and sent from a single `_progress_sender` thread, which skips the live send once a newer message is queued, so a slow socket cannot stall a stage
- After each job, `_run_pipeline`'s `finally` frees the Whisper model (`_release_whisper()`), unless `RELEASE_WHISPER_AFTER_JOB=0`

### Progress messages

`progress` messages have four fields: `{"type": "progress", "stage", "stage_name", "message"}`. Within Stages 1–3, sub-progress messages add `phase` (string) and `percent` (integer, or `null` for a phase that does not count its work); the phases are listed under each stage above. A counted phase starts at `0` and updates at most every 5% and every 10 s; `100` is never sent — a phase ends when the next message arrives. Stages 4–5 send only the four-field shape (Stage 5 as a 10 s heartbeat). `main.py` passes no callback; the CLI shows tqdm bars instead.

### Starting the server

`.\start_server.ps1` runs `python -m uvicorn api:app --host 0.0.0.0 --port 8010` with `PYTHONIOENCODING=utf-8`, after refusing to start if port 8010 is held and warning when commit headroom is below 6 GB. `-LogFile` also appends output to `output\server_8010.log`; `-Stop` stops the port-8010 listener only if it is this project's uvicorn. UTF-8 output matters: piped stdout on Windows defaults to cp1252, which cannot encode the em dashes and tqdm bar characters the stages print (`api.py` also reconfigures its streams to UTF-8 at startup).

### Settings isolation

`settings` is a global singleton. Each job resets it to startup defaults (`_default_settings` snapshot), then applies job-specific overrides. This avoids per-job `Settings()` instantiation overhead.

### CORS

Three-layer setup for ngrok compatibility:
1. `CORSMiddleware(allow_origins=["*"])` — standard FastAPI CORS
2. `CORSFallbackMiddleware` (BaseHTTPMiddleware) — stamps headers on every response
3. `@app.options("/{path:path}")` — explicit OPTIONS → 200 handler

### Stage 5 heartbeat

LLM API calls can take 30–60 s. A `_heartbeat_worker` daemon thread sends a progress ping every 10 s during Stage 5 to prevent ngrok from closing the idle WebSocket connection.

---

## Tools

`tools/merge_transcripts.py part1.json part2.json merged.json [--gap 1.0] [--swap-speakers A B] [--dry-run]` merges two transcript JSONs of one recording. Part 2's timestamps are offset to follow part 1 plus `--gap` seconds; `--swap-speakers` swaps two labels in part 2. The output keeps part 1's metadata, with `metadata.warnings` replaced by both parts' warnings prefixed `Part 1: ` / `Part 2: ` (omitted when neither has any). It supersedes the untracked `output/merge_transcripts.py`.

`tools/swap_speakers.py in.json out.json A B [--dry-run]` swaps two speaker labels throughout a transcript JSON: every `speaker` string field at any depth, a `label` field holding A or B, and every list of strings under a key containing `speaker` (e.g. `metadata.speaker_names`). All other fields, `metadata.warnings` included, are unchanged. It prints the segment count changed in each direction, refuses an output path equal to the input, and exits with an error if A equals B or either label appears in no speaker field. `--dry-run` writes nothing.

---

## Deferred / out of scope (current version)

- Real-time / streaming processing
- Batch processing (`--input-dir`)
- Docker packaging
