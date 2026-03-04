# Roadmap — Call Analysis Pipeline

Tracks what's built, what's next, and what's planned further out.

---

## Current state — v0.1 (complete)

The core pipeline is functional end-to-end:

- [x] Repo setup — `.gitignore`, `.env.example`, `README.md`, directory structure
- [x] `config.py` — Settings dataclass, `.env` loading, CLI override support
- [x] **Stage 1** — Audio pre-processing (noise reduction + normalization)
- [x] **Stage 2** — Speaker diarization (pyannote/speaker-diarization-3.1)
- [x] **Stage 3** — Whisper transcription (per-segment, local, offline)
- [x] **Stage 4** — Structured export (`.txt` + `.json` with metadata header)
- [x] `main.py` — CLI entry point with `ffmpeg` preflight check

---

## Near-term — v0.2

Improvements to make the pipeline more robust and useful before adding new stages.

### Must-have

- [ ] **First real-world test run** — validate output quality on an actual call recording
- [ ] **Error handling pass** — wrap each stage in try/except with clear failure messages; partial outputs should not silently corrupt the JSON
- [ ] **Segment merging** — consecutive segments from the same speaker (< N ms apart) should be merged before transcription to reduce Whisper API calls and improve context
- [ ] **Language config** — make Whisper `language` param configurable via `.env` (`WHISPER_LANGUAGE`, default `en`)

### Nice-to-have

- [ ] **Dry-run mode** — `--dry-run` flag that validates config and input file without running the pipeline
- [ ] **Stage skipping** — `--skip-preprocess` flag to pass a pre-cleaned WAV directly to Stage 2 (useful when re-running diarization on the same file)
- [ ] **Progress summary** — print a clean summary table at the end (duration, segment count, speaker breakdown)

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
- [ ] Output a Markdown report to `output/report.md`
- [ ] Add `--skip-analysis` flag to run pipeline without calling Claude API

### Large file support

- [ ] **Audio chunking** for files >160 MB — split into overlapping chunks, transcribe independently, stitch with speaker continuity preserved
- [ ] Progress reporting per chunk

---

## Longer-term — v1.0

### Quality improvements

- [ ] **Speaker name mapping** — allow user to provide `--speaker-names "Alice,Bob"` to replace generic `Speaker A / Speaker B` labels in output
- [ ] **Whisper word-level timestamps** — use Whisper's `word_timestamps=True` for finer-grained JSON output
- [ ] **Confidence scores** — include Whisper segment-level log-probability in JSON output
- [ ] **Multi-language** — detect language per segment or accept `--language` override

### Usability

- [ ] **Batch mode** — `python main.py --input-dir input/` to process all audio files in a directory
- [ ] **Watch mode** — monitor `input/` and auto-process new files as they appear
- [ ] **Config profiles** — named `.env` profiles (e.g. `--profile interview`) for quick context switching

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
