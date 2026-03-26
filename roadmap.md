# Roadmap — Call Analysis Pipeline

Tracks what's built, what's next, and what's planned further out.

---

## Current state — v0.1 (complete, GPU-accelerated)

The core pipeline is functional end-to-end and validated on a real M4A call recording with GPU acceleration:

- [x] Repo setup — `.gitignore`, `.env.example`, `README.md`, directory structure
- [x] `config.py` — Settings dataclass, `.env` loading, CLI override support
- [x] **Stage 1** — Audio pre-processing (noise reduction + normalization)
- [x] **Stage 2** — Speaker diarization (pyannote/speaker-diarization-3.1, GPU)
- [x] **Stage 3** — faster-whisper transcription (per-segment, local, GPU int8_float16)
- [x] **Stage 4** — Structured export (timestamped `.txt` + `.json` per run)
- [x] `main.py` — CLI entry point with ffmpeg preflight + GPU/device startup banner
- [x] **First real-world test run** — validated on `First_Test_File.m4a` (121 s, 2 speakers)
- [x] **GPU acceleration** — pyannote on CUDA, faster-whisper int8_float16 on GTX 1650
- [x] **pyannote 3.x/4.x API compatibility** — DiarizeOutput unwrapping, huggingface_hub.login() auth
- [x] **ctranslate2 Windows CUDA fix** — module-level model ref prevents mid-process teardown
- [x] **Unique output filenames** — `<source>_<YYYYMMDD_HHMMSS>.txt/json` per run

---

## Near-term — v0.2

### Must-have

- [ ] **Error handling pass** — wrap each stage in try/except with clear failure messages; partial outputs should not silently corrupt the JSON
- [ ] **Segment merging** — consecutive segments from the same speaker (< N ms apart) should be merged before transcription to reduce Whisper calls and improve context
- [ ] **Language config** — make Whisper `language` param configurable via `.env` (`WHISPER_LANGUAGE`, default `en`)

### Nice-to-have

- [ ] **Dry-run mode** — `--dry-run` flag that validates config and input file without running the pipeline
- [ ] **Stage skipping** — `--skip-preprocess` flag to pass a pre-cleaned WAV directly to Stage 2
- [ ] **Progress summary** — print a clean summary table at the end (duration, segment count, speaker breakdown, elapsed time)

---

## Medium-term — v0.3

### Stage 5 — Analysis Report (Claude API)

The primary next major feature. After the transcript is produced:

- [ ] Send the structured JSON transcript to the Claude API
- [ ] Prompt varies by `context` tag:
  - `friend` — emotional tone, recurring themes, mood
  - `work` — action items, decisions made, open questions
  - `interview` — candidate strengths/weaknesses, follow-up questions
  - `date` — compatibility signals, conversation balance, topics of interest
- [ ] Output a Markdown report to `output/<name>_<timestamp>_report.md`
- [ ] Add `--skip-analysis` flag to run pipeline without calling Claude API

### Large file support

- [ ] **Audio chunking** for files >160 MB — split into overlapping chunks, transcribe independently, stitch with speaker continuity preserved
- [ ] Progress reporting per chunk

---

## Longer-term — v1.0

### Quality improvements

- [ ] **Speaker name mapping** — `--speaker-names "Alice,Bob"` to replace generic labels
- [ ] **Whisper word-level timestamps** — `word_timestamps=True` for finer-grained JSON
- [ ] **Confidence scores** — include Whisper segment-level log-probability in JSON
- [ ] **Multi-language** — detect language per segment or accept `--language` override

### Usability

- [ ] **Batch mode** — `python main.py --input-dir input/` to process all audio files
- [ ] **Watch mode** — monitor `input/` and auto-process new files as they appear
- [ ] **Config profiles** — named `.env` profiles (e.g. `--profile interview`)

### Infrastructure

- [ ] **Docker image** — single-container setup with ffmpeg, Python deps, and Whisper model baked in
- [ ] **Pre-commit hooks** — lint (`ruff`) and type-check (`mypy`) on commit
- [ ] **Unit tests** — pytest suite covering config loading, label mapping, timestamp formatting, JSON schema

---

## Icebox (no timeline)

- Web UI wrapper (Flask or FastAPI + simple HTML front-end)
- Speaker identification (match `Speaker A` to a known voice profile)
- Real-time streaming transcription
- Cloud storage integration (S3/GCS for input/output)
- Webhook on completion (e.g. post JSON to a URL)
