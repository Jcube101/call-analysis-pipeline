# Roadmap — Call Analysis Pipeline

Tracks what's built, what's next, and what's planned further out.

---

## v0.1 — complete

Core pipeline functional end-to-end, validated on a real M4A call recording with GPU acceleration:

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

## v0.2 — complete

- [x] **Error handling pass** — each stage wrapped in try/except with clear `[error] Stage N` messages; exits with code 1 on failure
- [x] **Language config** — `WHISPER_LANGUAGE` env var + `--language` CLI flag; passed to both transcription modes
- [x] **Transcription modes** — `accurate` (default, per-segment) and `fast` (whole-file, ~20% faster but coarser output); `TRANSCRIPTION_MODE` env var + `--transcription-mode` CLI flag
- [x] **Dry-run mode** — `--dry-run` validates config and input file without running any stage
- [x] **Stage skipping** — `--skip-preprocess` passes a pre-cleaned WAV directly to Stage 2
- [x] **Progress summary** — completion banner shows segment count, speaker breakdown, audio duration, total elapsed, and per-stage timing
- [x] **Validated on longer recordings** — tested on 10:56 MPEG file (134 segments, 2 speakers); ~1x real-time on GTX 1650 in accurate mode

### Known limitation: fast mode

`fast` mode transcribes the full file in one Whisper call. Whisper internally processes in ~30s chunks, producing ~4 segments for a 2-minute file and ~49 for an 11-minute file — significantly fewer lines than `accurate` mode. Kept for cases where coarse output is acceptable. A smarter fast mode (e.g. per-merged-turn transcription) is a future improvement.

---

## v0.3 — in progress

Priority order: Stage 5 (Claude report) → large file support → UI wrapper.

### Stage 5 — Analysis Report (Claude API) ← in progress

- [x] `--report` flag triggers Stage 5 after Stage 4
- [x] Prompt loaded from `prompts/<context>.md` — user-editable per context
- [x] Metadata header included: source file, context, speakers, audio duration
- [x] Large transcripts chunked (~80k tokens each), partial reports synthesised
- [x] Output: `output/<name>_<timestamp>_report.md`
- [x] Terminal preview: first 20 lines of report printed after Stage 5
- [x] `ANTHROPIC_API_KEY` validated early when `--report` is passed
- [ ] **Real-world test** — run on an actual call recording with `--report`

### Large file support

- [ ] **Audio chunking** for files >160 MB — split into overlapping chunks, transcribe independently, stitch with speaker continuity preserved
- [ ] Progress reporting per chunk

### UI wrapper

- [ ] Local web UI (FastAPI + minimal HTML) so the pipeline can be triggered via a browser without using the terminal
- [ ] File upload, progress display, download links for outputs

---

## v1.0 — longer-term

### Quality improvements

- [ ] **Speaker name mapping** — `--speaker-names "Alice,Bob"` to replace generic labels
- [ ] **Whisper word-level timestamps** — `word_timestamps=True` for finer-grained JSON
- [ ] **Confidence scores** — include Whisper segment-level log-probability in JSON
- [ ] **Smarter fast mode** — merge same-speaker diarization turns, transcribe per turn; fewer Whisper calls than accurate, better granularity than current fast

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

- Speaker identification (match `Speaker A` to a known voice profile)
- Real-time streaming transcription
- Cloud storage integration (S3/GCS for input/output)
- Webhook on completion (e.g. post JSON to a URL)
