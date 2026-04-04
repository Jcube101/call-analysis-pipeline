# Roadmap — Call Analysis Pipeline

Tracks what's built, what's next, and what's planned further out.

---

## Current state — v1.0 (complete, tested)

Speaker re-identification and name mapping ship in v1.0. All core quality
improvements are done. See below for per-version history.

---

## v0.1 (complete, tested)

The core pipeline is functional end-to-end and has been validated on a real M4A call recording:

- [x] Repo setup — `.gitignore`, `.env.example`, `README.md`, directory structure
- [x] `config.py` — Settings dataclass, `.env` loading, CLI override support
- [x] **Stage 1** — Audio pre-processing (noise reduction + normalization)
- [x] **Stage 2** — Speaker diarization (pyannote/speaker-diarization-3.1)
- [x] **Stage 3** — Whisper transcription (per-segment, local, offline)
- [x] **Stage 4** — Structured export (`.txt` + `.json` with metadata header)
- [x] `main.py` — CLI entry point with `ffmpeg` preflight check
- [x] **First real-world test run** — validated on `First_Test_File.m4a` (121 s, 2 speakers, CPU)
- [x] **pyannote 3.x API compatibility** — `DiarizeOutput` unwrapping + in-memory waveform passthrough
- [x] **torchcodec warning suppression** — harmless warning filtered at import; root cause documented

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

## v0.3 — complete

Priority order: Stage 5 (Gemini report) ✓ → large file support ✓ → UI wrapper (deferred to v1.0).

### Stage 5 — Analysis Report (Gemini API) — complete

- [x] `--report` flag triggers Stage 5 after Stage 4
- [x] Prompt loaded from `prompts/<context>.md` — user-editable per context
- [x] Metadata header included: source file, context, speakers, audio duration
- [x] Large transcripts chunked (~500k chars each), partial reports synthesised
- [x] Output: `output/<name>_<timestamp>_report.md`
- [x] Terminal preview: first 20 lines of report printed after Stage 5
- [x] `GEMINI_API_KEY` validated early when `--report` is passed
- [x] **Real-world test** — validated on First_Test_File (work context) with `gemini-3-flash-preview`

### Large file support — complete

- [x] **Chunked noise reduction** — Stage 1 processes 60s slices with 0.5s overlap; peak RAM ~350 MB regardless of file length (was 3–4 GB for 90+ min files)
- [x] **Memory freed between stages** — waveform tensors deleted after pyannote; dead normalization code removed from Stage 2
- [x] **Duration via header only** — `soundfile.info()` replaces full pydub WAV load in summary
- [x] **Progress bar** for noise reduction chunks
- [x] **Real-world test** — validated on 2h40m M4A (9756s, 163 noise-reduction chunks, 3102 diarization segments, 1819 transcript segments); total elapsed 2h43m on GTX 1650

### UI wrapper — deferred to v1.0

Moved to v1.0 — terminal workflow is sufficient for current use.

---

## v1.0 — complete

### Quality improvements

- [x] **Speaker re-identification** — MFCC + delta features (librosa) per segment; KMeans clustering; cluster IDs remapped to globally consistent `SPEAKER_XX` labels in first-appearance order. Fixes label-flipping on long recordings.
- [x] **Speaker name mapping** — `--speaker-names "Alice,Bob"` (or `SPEAKER_NAMES=Alice,Bob` in `.env`) replaces `Speaker A/B` with real names in transcript and report output. `--from-json --speaker-names` writes a `_named.json` alongside the original.
- [x] **Confidence scores** — segment-level 0–1 score (duration-weighted `exp(avg_logprob)`) included in every JSON segment.
- [x] **Word-level timestamps** — `--word-timestamps` (or `WORD_TIMESTAMPS=true`) adds a `words` list to each JSON segment with per-word start/end times and probability scores (~10-15% overhead).
- [x] **Smarter fast mode** — merges consecutive same-speaker diarization segments into turns (gap ≤ 1s), transcribes one turn at a time; 10-20x fewer Whisper calls than accurate on long recordings while preserving speaker-accurate boundaries.

### API / Integration

- [x] **FastAPI wrapper fully operational** — HTTP + WebSocket API with job management, real-time progress, message queue replay on reconnect
- [x] **WebSocket progress updates** — per-stage progress pushed to client; heartbeat during Stage 5 keeps connection alive across ngrok idle timeout
- [x] **Frontend deployed** — job-joseph.com/projects/call-analysis; upload audio, watch progress, view transcript + report, download all files
- [x] **Demo tab** — real Verstappen interview sample output shown on the site
- [x] **All download endpoints working** — `transcript`, `json`, `report`, `wav`; correct `Content-Type` and `Content-Disposition` headers; named JSON takes priority
- [x] **Markdown report rendered in browser UI** — frontend renders `.md` report content correctly
- [x] **Report from JSON writes back to original job folder** — `POST /report-from-json` links to existing job by `metadata.job_id`
- [x] **Job disk recovery after server restart** — files persist in `output/jobs/{job_id}/`, served directly from disk with no status check
- [x] **generate_report flag threading fix** — stored explicitly in job dict; explicit `bool()` cast handles string form values
- [x] **Selectable Gemini model** — `gemini_model` param on both POST endpoints; allowed values: Flash (default), Pro, Flash Lite; invalid values silently replaced with default
- [x] **Automatic fallback to Flash on Pro 503** — `_call_gemini` catches 503 and retries with `gemini-3-flash-preview`; actual model used is logged
- [x] **5 minute Gemini API timeout** — `HttpOptions(timeout=300)` prevents indefinite hangs when Pro is overloaded
- [x] **Stale state fix** — full reset between pipeline runs; `runId` pattern discards stale WebSocket messages from previous runs
- [x] **Sub-modes renamed** — "Analyse Audio" and "Report from Transcript"
- [x] **Model selector helper text** — explains Flash vs Pro vs Flash Lite tradeoffs
- [x] **Demo tab Markdown rendering fixed**
- [x] **Unit test suite** — pytest, 4 modules (config, export, api, report), 74 tests, all passing
- [x] **Integration tests completed manually** — full pipeline on real M4A files, all stages, all three Gemini models, two-pass renaming workflow
- [x] **API tests completed manually** — all endpoints validated: /analyse, /report-from-json, /status, /reconnect, /download/*, WebSocket, disk recovery

### Usability

- [ ] **Batch mode** — `python main.py --input-dir input/` to process all audio files
- [ ] **Watch mode** — monitor `input/` and auto-process new files as they appear
- [ ] **Config profiles** — named `.env` profiles (e.g. `--profile interview`)

### Infrastructure

- [ ] **Docker image** — single-container setup with ffmpeg, Python deps, and Whisper model baked in
- [ ] **Pre-commit hooks** — lint (`ruff`) and type-check (`mypy`) on commit

---

---

## v1.1 — Next improvements

- [ ] **WebSocket UI catches up correctly after ngrok reconnect** — UI should resume correct stage display without requiring Start Over
- [ ] **Automated integration tests** — short fixture audio file run through the full pipeline in CI
- [ ] **Automated API tests** — FastAPI TestClient covering all endpoints without starting a real server
- [ ] **CI/CD pipeline** — GitHub Actions running `pytest tests/` on every push
- [ ] **Stereo audio support** — dedicated per-channel diarization for perfect speaker separation on two-mic recordings
- [ ] **Manual speaker label correction UI** — in-browser editor to reassign speaker labels on the transcript before downloading
- [ ] **Batch mode** — process multiple files from `input/` folder in sequence
- [ ] **Watch mode** — auto-process new files as they appear in `input/`
- [ ] **Update CallAnalysisPipelineDetail.tsx** — reflect full built state including FastAPI, ngrok, web UI, and Gemini model selection

---

## Icebox (no timeline)

- Speaker identification (match `Speaker A` to a known voice profile)
- Fine-tune pyannote on interview-style single-mic recordings to reduce label flipping
- Real-time streaming transcription
- Cloud storage integration (S3/GCS for input/output)
- Webhook on completion (e.g. post JSON to a URL)
- Persistent SQLite job storage (replace in-memory dict; survive process crashes)
- ngrok static domain — paid tier eliminates the URL-change-on-restart limitation
- Docker image with `api.py` included (single container for CLI + API modes)
