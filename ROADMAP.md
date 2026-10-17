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
- [x] **pyannote 3.x API compatibility** — `DiarizeOutput` unwrapping (audio is passed as a file path; see CLAUDE.md)
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

### Known limitation: fast mode — resolved in v1.0

Resolved by the v1.0 **Smarter fast mode** (one Whisper call per merged speaker turn). The original note: `fast` mode transcribes the full file in one Whisper call. Whisper internally processes in ~30s chunks, producing ~4 segments for a 2-minute file and ~49 for an 11-minute file — significantly fewer lines than `accurate` mode. Kept for cases where coarse output is acceptable. A smarter fast mode (e.g. per-merged-turn transcription) is a future improvement.

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
- [x] **Selectable AI model** — model param on both POST endpoints (`gemini_model`, renamed `analysis_model` in v1.1); launched with Claude Haiku 4.5 / Sonnet 4.6 and Gemini 3 Flash / 3.1 Pro (since migrated, see v1.1); routes to correct provider; invalid values replaced with default
- [x] **Automatic fallback** — Claude models fall back to Haiku on failure; Gemini models fall back to Flash on 503; actual model used is logged and shown in report header
- [x] **Context hints** — `context_hints` parameter on both POST endpoints; free-text hints for correct spelling of names, places, and terms prepended to system prompt
- [x] **Stale state fix** — full reset between pipeline runs; `runId` pattern discards stale WebSocket messages from previous runs
- [x] **Sub-modes renamed** — "Analyse Audio" and "Report from Transcript"
- [x] **Model selector helper text** — explains Claude Haiku vs Sonnet and Gemini Flash vs Pro tradeoffs
- [x] **Demo tab Markdown rendering fixed**
- [x] **Unit test suite** — pytest, 4 modules (config, export, api, report), 88 tests, all passing; includes API key leak prevention tests across all output surfaces (now 9 modules, 250 tests — see v1.1)
- [x] **Integration tests completed manually** — full pipeline on real M4A files, all stages, Claude and Gemini models, two-pass renaming workflow
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
- [ ] **Pin NVIDIA driver version in README** to prevent silent CUDA compatibility breaks
- [ ] **Manual speaker label correction UI** — in-browser editor to reassign speaker labels on the transcript before downloading
- [ ] **Batch mode** — process multiple files from `input/` folder in sequence
- [ ] **Watch mode** — auto-process new files as they appear in `input/`
- [ ] **Update CallAnalysisPipelineDetail.tsx** — reflect full built state including FastAPI, ngrok, web UI, and Gemini model selection
- [ ] **Stage 5 progress bar** — Stage 5 sends only a 10 s heartbeat; no `phase` / `percent` while the LLM call or chunk synthesis runs
- [ ] **One-command two-part merge** — a split recording currently takes two pipeline runs, `tools/merge_transcripts.py`, then `--from-json`; do it in one command
- [x] **Model registry and 5.5 / 3.8 migration** — `models.py` is the single source of Stage 5 model IDs; models moved to `claude-haiku-5-5` (default), `claude-sonnet-5-5`, `gemini-3.8-flash`, `gemini-4-argon`; legacy IDs remapped with a log line; Claude thinking blocks handled; truncated output raises; CLI gains `--report-model` and now passes the Anthropic key (CLI Claude reports previously failed auth)
- [x] **Rename `gemini_model` to `analysis_model` (2026-10-14)** — API field, job dict key, `report.run()` / `validate_for_report()` params; `gemini_model` kept as a deprecated form alias (new field wins if both sent) and as a `/status` mirror
- [x] **Remove the `gemini_model` alias (2026-10-14)** — Lovable frontend sends `analysis_model` (confirmed with a real report, no deprecation line); form alias and `/status` mirror removed; a stale client still sending `gemini_model` gets the default model and `[models] Ignored removed field 'gemini_model'; send 'analysis_model'` instead of an error; legacy model-ID remapping kept
- [x] **Stream Stage 3 audio slices** — `transcribe.py` reads each diarization segment from disk via `_WavReader` instead of loading the whole clean WAV; audio cost dropped from ~2x file size to a flat ~25 MB
- [x] **Shrink the Stage 2 waveform buffers** — `diarize.py` reads the clean WAV as `float32` instead of `soundfile`'s float64 default; 1.16 GiB array down to 595 MB and one full-length copy removed, output bit-identical
- [x] **Drop `pydub`** — removed from `requirements.txt`; nothing in the pipeline imports it
- [x] **Stage 2 memory ceiling on long recordings** — was neither VRAM nor the waveform: pyannote fed every embedding to scipy's `linkage`, an O(N²) distance matrix (2.1 GB on a 2h43m file). Fixed by restoring pyannote's own `max_num_embeddings` cap, which its 3.1 config disables, plus reading audio from disk. Peak 7656 → 5670 MB and the 2h43m file now completes 3 of 3 (see [KNOWN-ISSUES.md](KNOWN-ISSUES.md))
- [x] **Remove `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` (2026-10-14)** — it made Stage 2 spill past the 4 GB card into shared memory (embeddings 0:07 → 1:19 on the 2-minute file, CUDA OOM on the 1.72 h file); see KNOWN-ISSUES.md §4
- [x] **Stage 1–3 progress over the WebSocket** — `phase` / `percent` sub-progress messages (5% / 10 s throttle; `stages/progress.py` for Stages 1 and 3), sent from a single `_progress_sender` thread; Stage 2 prints plain phase lines and a tqdm bar that stays readable in a piped log
- [x] **Seed pyannote's clustering subsample** — `CLUSTERING_SUBSAMPLE_SEED` makes Stage 2 repeatable on files above the 1000-embedding cap (unseeded runs of one 10-minute file gave 270–283 segments)
- [x] **Fix the silent re-identification skip under a non-UTF-8 stdout** — Stage 2 printed `≥`, which cp1252 cannot encode; see KNOWN-ISSUES.md §3
- [x] **Re-identification outcome and warnings** — `stages/reid_outcome.py` (`ok` / `partial` at 1 or more failed segments / `failed`), labels written back in one pass; `warnings` in `complete`, `/reconnect` and `/status`, `job["reid"]`, `metadata.warnings`, `# Warning:` lines in the .txt, `**Warning:**` lines in the Stage 5 report header
- [x] **Track `tools/merge_transcripts.py`** — merges part 1 + part 2 JSONs and carries both parts' warnings as `Part N: ...`; supersedes the untracked `output/merge_transcripts.py`
- [x] **`start_server.ps1`** — starts the API on 8010 with `PYTHONIOENCODING=utf-8`, port and commit-headroom pre-checks, `-LogFile` and `-Stop`
- [x] **Free the Whisper model after each API job (2026-10-14)** — `_release_whisper()` after `complete` / `error`; `RELEASE_WHISPER_AFTER_JOB=0` restores holding it until exit; `ctranslate2` and `faster-whisper` pinned (KNOWN-ISSUES.md §2)
- [x] **Track `tools/swap_speakers.py`** — swaps two speaker labels in a transcript JSON; refuses to overwrite the input or run with an unknown label
- [x] **Unit tests at 250** — 9 modules: config, export, api, report, models, progress, merge_transcripts, diarize_reid, swap_speakers

---

## Alternative stack candidates

### Voxtral (Mistral AI) — evaluate as pyannote + Whisper replacement
Released July 2025, Apache 2.0. Combines transcription and speaker diarization in a single model — potentially replacing Stages 2 and 3 of the current pipeline with one model.

Two relevant variants:
- Voxtral Mini (3B) — local deployment, ~2-4GB, may fit GTX 1650
- Voxtral Realtime (4B, Apache 2.0) — sub-200ms latency, built-in diarization, 13 languages, open weights on HuggingFace

Why it matters:
- Built-in diarization could fix the label drift problem on long single-mic recordings
- Combining Stages 2+3 into one model simplifies the pipeline
- Apache 2.0 means fully local, privacy-preserving deployment

Evaluation plan:
1. Run Verstappen interview through Voxtral Transcribe 2 API ($0.003/min, ~$0.025 for 8 mins) to compare diarization quality
2. If quality is better, test Voxtral Realtime weights locally on GTX 1650 with quantization
3. If VRAM fits, replace stages/diarize.py + stages/transcribe.py with a single stages/voxtral.py module

Blocker: GTX 1650 has 4GB VRAM — Voxtral Realtime at 4B parameters needs quantization to fit. Current pipeline already hits VRAM pressure between Stage 2 and Stage 3.

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
