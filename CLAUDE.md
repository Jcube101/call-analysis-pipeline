# CLAUDE.md — Project Context for Claude Code

This file gives Claude context about the call-analysis-pipeline project so it can assist effectively without re-exploring the codebase each session.

## What this project does

Takes an audio recording of a two-person conversation (MP3, M4A, WAV, or any ffmpeg-supported format) and outputs:
1. A noise-reduced WAV (`output/*_clean.wav`)
2. A speaker-diarized, timestamped transcript (`output/<name>_<timestamp>.txt`)
3. A structured JSON file ready for downstream analysis (`output/<name>_<timestamp>.json`)
4. An AI-generated analysis report (`output/<name>_<timestamp>_report.md`) — triggered with `--report`, uses Gemini API

## Status

v1.0 complete and fully operational. FastAPI wrapper with WebSocket progress, frontend deployed at job-joseph.com/projects/call-analysis. All pipeline stages, downloads, and report generation working end to end.

**Known limitations:**
- Speaker diarization on single-mic recordings has occasional label flipping — use `--from-json` with `--speaker-names` to correct after reviewing the transcript.
- WebSocket UI can get stuck on the last known stage after an ngrok reconnect — the pipeline completes correctly and files are saved; use Start Over and re-check the output folder as a workaround.
- ngrok free tier changes URL on every restart — paste the new URL into the Backend URL field on the site.

## How to run

```bash
python main.py --input input/call.mp3
# Common overrides:
python main.py --input input/call.mp3 --context work --num-speakers 3
python main.py --input input/call.mp3 --transcription-mode fast
python main.py --input input/call.mp3 --speaker-names "Alice,Bob"
python main.py --input input/call.mp3 --language fr
# Generate analysis report via Gemini API:
python main.py --input input/call.mp3 --context work --report
# Utility flags:
python main.py --input input/call.mp3 --dry-run
python main.py --input output/call_clean.wav --skip-preprocess
# Generate a report from an existing JSON (skips Stages 1–4):
python main.py --from-json output/call_20260328_151811.json --context work
```

All config (tokens, context, speaker count, transcription mode, language) lives in `.env`. See `.env.example`.

## Project layout

```
main.py          — entry point, orchestrates all stages (CLI)
api.py           — FastAPI HTTP + WebSocket wrapper (programmatic access)
config.py        — Settings dataclass, loads .env via python-dotenv
stages/
  preprocess.py  — Stage 1: noise reduction + normalization (pydub, noisereduce)
  diarize.py     — Stage 2: speaker diarization (pyannote/speaker-diarization-3.1)
  transcribe.py  — Stage 3: faster-whisper transcription (runs locally, GPU-accelerated)
  export.py      — Stage 4: writes timestamped .txt and .json output
  report.py      — Stage 5: Gemini API analysis report (--report flag only)
prompts/
  friend.md      — analysis prompt for friend conversations (user-editable)
  work.md        — analysis prompt for work conversations (user-editable)
  interview.md   — analysis prompt for job interviews (user-editable)
  date.md        — analysis prompt for date conversations (user-editable)
input/           — place source audio files here (gitignored, must exist locally)
output/          — pipeline outputs land here (gitignored, must exist locally)
```

## Key conventions

- **All secrets and config come from `.env`** — nothing hardcoded. Config is accessed via the `settings` singleton in `config.py`.
- **Each stage is a module** with a single `run()` function. `main.py` calls them in order and passes the output of one stage as input to the next.
- **Segment dicts** are the internal data structure passed between stages. Each is a dict with keys `start`, `end`, `speaker`, `label` (and `text`, `confidence`, optionally `words` after Stage 3).
- **Windows compatibility** — avoid Unix-only shell commands or hardcoded `/` paths; use `os.path` instead.
- **No chunking yet** — large file support is deferred, but don't architect it out. Keep it in mind when touching Stage 1 or 3.
- **Error handling** — each stage call in `main.py` is wrapped in try/except. Failures print `[error] Stage N (name) failed: <message>` and exit with code 1.
- **`generate_report` flag** must be stored explicitly on the job dict as `job["generate_report"] = bool(generate_report)` when the job is created — do not rely on function arguments alone as the value can be lost between threads.
- **Download endpoints use glob patterns** to find files by type — do not filter out files with `"input"` in the name as all timestamped output files start with `input_`. Filter by specific suffixes instead (`_report`, `_named`).
- **txt glob** excludes `"_report"` to avoid matching report files; **json glob** excludes `"named"` to avoid matching `transcript_named.json` as the fallback (named JSON takes priority over generic JSON).
- **Majority vote smoothing** should NOT be applied to interview-style recordings where one speaker dominates — it collapses minority-speaker segments into the majority speaker. MFCC re-identification helps with label drift on long recordings but cannot fix short-segment misattribution at conversation boundaries. Practical fix for label flipping: use `--from-json` with `--speaker-names` after reviewing the transcript.
- **Job state must be set atomically before `_push_complete()`** — read the report file first (catching and printing any errors with `except Exception as e: print(...)`), then set `job["report"]`, `job["report_path"]`, `job["status"]`, and all other fields as a single block, then call `_push_complete()`. Never send a progress "Done" message before the file read completes. A bare `except: pass` on a report file read silently leaves `job["report"] = None`, causing the frontend to receive `"report": null` in the complete message.
- **`/reconnect` report check**: use `if report is None:` (not `if not report:`) when deciding whether to re-read the report from disk — an empty string report is valid and should not trigger a redundant disk read.
- **`gemini_model` parameter** — accepted by `POST /analyse` and `POST /report-from-json`. Allowed values: `gemini-3-flash-preview` (default), `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`. Invalid values are silently replaced with the default. Stored on the job dict so background threads read it from job state rather than function arguments.
- **Automatic fallback to Flash on 503** — if the Gemini API returns a 503, `_call_gemini` retries the same request with `gemini-3-flash-preview` before raising. The model that actually ran is logged at Stage 5 completion.
- **5 minute timeout on all Gemini API calls** — `HttpOptions(timeout=300)` is set on every `generate_content` call. Without it the process waits indefinitely when Pro is overloaded.
- **Report-from-JSON folder linking** — `POST /report-from-json` reads `metadata.job_id` from the uploaded JSON before choosing an output directory. If that job folder exists on disk the report is written there, keeping all files for a call together. Otherwise a new `job_id` and folder are created.
- **Each new pipeline run must generate a fresh `job_id`** — never reuse a previous `job_id`. The frontend must clear all state (jobId, transcript, report, error, WebSocket) before starting a new run. A `runId` (e.g. `Date.now()`) is useful to discard stale WebSocket messages from a previous run that arrive after the new run starts.

## Output file naming

Each run produces uniquely named files using the source filename + timestamp:
```
output/First_Test_File_20260327_143022.txt
output/First_Test_File_20260327_143022.json
output/First_Test_File_20260327_143022_report.md   (--report only)
output/First_Test_File_clean.wav
```
The clean WAV is overwritten each run (Stage 1 output). Transcripts and reports are never overwritten.

## GPU / transcription stack

Transcription uses **faster-whisper** (CTranslate2-based) instead of openai-whisper:
- On CUDA: `device="cuda", compute_type="int8_float16"` — fits in 4 GB VRAM
- On CPU: `device="cpu", compute_type="int8"`

**ctranslate2 CUDA teardown bug (Windows):** ctranslate2's `__del__` calls `exit()` when the WhisperModel is garbage-collected mid-process on Windows. Fixed by holding a module-level reference (`_active_model`) so cleanup is deferred to process exit. Do not add `del model` inside `transcribe.run()`.

## Transcription modes

Two modes selectable via `TRANSCRIPTION_MODE` env var or `--transcription-mode` CLI flag:

- **accurate** (default): one `model.transcribe()` call per diarization segment. Produces fine-grained sentence-level output. ~1x real-time on GTX 1650.
- **fast**: merges consecutive same-speaker diarization segments into "turns" (gap ≤ 1s), then one `model.transcribe()` call per turn. Produces one output line per speaker turn — 10-20x fewer Whisper calls than accurate on long recordings while preserving speaker-accurate boundaries.

**Do not** pre-merge diarization segments before transcription **in accurate mode**. pyannote commonly outputs many same-speaker segments with tiny gaps (<100ms) between them — merging them collapses entire speaking turns into one line. Fast mode intentionally merges into turns as that is its design goal.

## Stage 5 — Gemini report

Stage 5 is optional and only runs when `--report` is passed. It sends the transcript to `gemini-3-flash-preview` via the `google-genai` SDK.

- Prompt loaded from `prompts/<context>.md` — user can edit these freely. Each prompt defines specific structured output sections and embeds a speaker label reliability warning.
- Falls back to built-in defaults if the prompt file is missing
- Large transcripts (>500k chars) are chunked and partial reports synthesised
- Output: `output/<name>_<timestamp>_report.md`
- First 20 lines printed to terminal as a preview

**SDK note:** Use `google-genai` (not the deprecated `google-generativeai`). Import path is `from google import genai`. Client pattern: `client = genai.Client(api_key=...)`, then `client.models.generate_content(model=..., contents=...)`.

## pyannote API compatibility

pyannote.audio 3.x no longer returns a `pyannote.core.Annotation` directly — it returns a `DiarizeOutput` dataclass. `diarize.py` handles this with a fallback chain checking `itertracks` → `exclusive_speaker_diarization` → `speaker_diarization`.

## HuggingFace authentication

pyannote 3.4.0 + huggingface_hub 0.x: `Pipeline.from_pretrained()` no longer accepts `token=` or `use_auth_token=` as keyword arguments. Use `huggingface_hub.login(token=...)` before calling `from_pretrained()`:

```python
if hasattr(diarization, "itertracks"):          # old API
    annotation = diarization
elif hasattr(diarization, "exclusive_speaker_diarization"):  # new API (3.x)
    annotation = diarization.exclusive_speaker_diarization
elif hasattr(diarization, "speaker_diarization"):
    annotation = diarization.speaker_diarization
```

Do not revert this to a direct `.itertracks()` call on the pipeline output.

## Audio input to pyannote (important)

Audio is passed to the pyannote pipeline as a pre-loaded in-memory dict — **not** as a file path:

```python
audio_input = {"waveform": waveform, "sample_rate": sample_rate}
diarization = pipeline(audio_input, ...)
```

This avoids a dependency on `torchcodec` (which is not installed). A `UserWarning` about torchcodec is suppressed at import time since it is irrelevant when using this approach.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_TOKEN` | Yes | For pyannote diarization model download |
| `GEMINI_API_KEY` | With `--report` | For Stage 5 report generation via Gemini API |
| `CONVERSATION_CONTEXT` | No | `friend` / `work` / `interview` / `date` / `public_interview` (default: `friend`) |
| `NUM_SPEAKERS` | No | Integer or blank for auto-detect (default: auto) |
| `WHISPER_MODEL` | No | `tiny` / `base` / `small` / `medium` / `large` (default: `medium`) |
| `TRANSCRIPTION_MODE` | No | `accurate` / `fast` (default: `accurate`) |
| `WHISPER_LANGUAGE` | No | BCP-47 language code, e.g. `en`, `fr`, `es` (default: `en`) |
| `SPEAKER_NAMES` | No | Comma-separated real names, e.g. `Alice,Bob` (default: generic labels) |
| `WORD_TIMESTAMPS` | No | `true` / `1` to include per-word timestamps in JSON (~10–15% overhead) |

## Dependencies and install order

Key packages: `pydub`, `noisereduce`, `pyannote.audio`, `faster-whisper`, `torch`, `soundfile`, `librosa`, `python-dotenv`, `tqdm`, `google-genai`, `fastapi`, `uvicorn`, `python-multipart`.
System dependency: `ffmpeg` must be on PATH (`main.py` checks this on startup).

Key version constraints (all in `requirements.txt`):
- `torch==2.1.0+cu121` — newer torch requires numpy 2.x which breaks pyannote
- `numpy<2.0` — pyannote compiled against numpy 1.x
- `pyannote.audio<4.0` — 4.0.4 requires torch>=2.8.0 which doesn't exist yet
- `huggingface_hub<1.0.0` — 1.x removed `use_auth_token` used internally by pyannote 3.x
- `google-genai>=1.0.0` — replaces deprecated `google-generativeai`

## Install order matters (Windows / CPU-only)

Install torch **before** everything else so pip doesn't pull in a newer incompatible version later:

```bash
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

## FastAPI wrapper (api.py)

`api.py` exposes the full pipeline over HTTP + WebSocket. Start it with:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Requires `fastapi`, `uvicorn`, `python-multipart` (not in `requirements.txt` — install separately or add them).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyse` | Upload audio file, start pipeline job; returns `job_id` |
| POST | `/report-from-json` | Upload transcript JSON, run Stage 5 only; returns `job_id` |
| GET | `/status/{job_id}` | Poll current job status and stage |
| GET | `/reconnect/{job_id}` | Full job state recovery (transcript + report + metadata) |
| GET | `/download/{job_id}/transcript` | Download `.txt` transcript |
| GET | `/download/{job_id}/json` | Download transcript JSON (`transcript_named.json` if exists, else timestamped `.json`) |
| GET | `/download/{job_id}/report` | Download `*_report.md` file |
| GET | `/download/{job_id}/wav` | Download `*_clean.wav` file |
| WS | `/ws/{job_id}` | WebSocket — real-time progress, final `complete` message |

Download endpoints serve directly from disk with no job status check — files are returned as soon as they exist on disk regardless of in-memory job state. The `Content-Disposition` header carries the original filename so browsers save with the correct extension.

**POST `/analyse` form parameters** (all optional except `file`):
- `file`: audio upload (required)
- `context`, `num_speakers`, `transcription_mode`, `language`, `speaker_names`, `word_timestamps`, `generate_report`, `skip_preprocess`, `whisper_model` — as documented elsewhere
- `gemini_model: str` (default: `gemini-3-flash-preview`) — Gemini model used for Stage 5. Allowed values: `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`. Invalid values are silently replaced with the default.

**POST `/report-from-json` form parameters**:
- `file`: transcript JSON upload (required)
- `context`, `speaker_names` — as documented elsewhere
- `gemini_model: str` (default: `gemini-3-flash-preview`) — same allowed values and fallback as above.

### Job folder file naming

All output files from `api.py` use the uploaded filename stem (`input`) plus a timestamp:
```
output/jobs/{job_id}/
├── input.{ext}                      — uploaded audio
├── input_clean.wav                  — Stage 1 output
├── input_{YYYYMMDD_HHMMSS}.txt      — Stage 4 transcript
├── input_{YYYYMMDD_HHMMSS}.json     — Stage 4 JSON
├── transcript_named.json            — relabelled JSON (report-from-json with speaker names)
└── input_{YYYYMMDD_HHMMSS}_report.md — Stage 5 report
```

### Job model

Each job is stored in an in-memory `jobs` dict and in `output/jobs/{job_id}/`. On server restart, `get_or_recover_job()` scans that directory and hydrates the job from disk (transcript JSON + report MD).

WebSocket messages are stored in `job["message_queue"]` so clients that disconnect mid-job can replay all messages on reconnect via the WS endpoint's flush-on-connect logic.

The `complete` WebSocket message shape:
```json
{"type": "complete", "transcript": [...], "report": "...", "metadata": {...}}
```

### Threading model

All pipeline jobs run in a `ThreadPoolExecutor(max_workers=1)`. This serialises jobs since pyannote + Whisper together consume the full 4 GB VRAM budget. The `asyncio` event loop is captured at startup in `_loop`; background threads push WebSocket messages via `asyncio.run_coroutine_threadsafe(...).result(timeout=5)`.

`threading.excepthook` is installed at module level so crashes in pipeline threads print a traceback and exit gracefully without killing uvicorn.

### Settings per job

`settings` is a global singleton. Each job resets it via `settings.__dict__.update(_default_settings)` where `_default_settings` is a snapshot taken at startup. Job-specific overrides (context, num_speakers, etc.) are then applied on top.

### CUDA cleanup between stages

After Stage 2 (pyannote) and before Stage 3 (Whisper):
```python
torch.cuda.synchronize()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
time.sleep(2)
```
Without this, Whisper fails to allocate VRAM because pyannote has not fully released it.

### CORS

Three layers to handle ngrok headers:
1. `CORSMiddleware` — standard FastAPI CORS middleware with `allow_origins=["*"]`
2. `CORSFallbackMiddleware` (BaseHTTPMiddleware) — stamps CORS headers on every response
3. `@app.options("/{path:path}")` — explicit OPTIONS handler returning 200

### Heartbeat during Stage 5

The Gemini API call takes 30–60 s. ngrok free tier drops idle WebSocket connections after ~30 s. A `_heartbeat_worker` daemon thread sends a progress ping every 10 s during Stage 5 to keep the connection alive.

### report-from-json folder linking

`POST /report-from-json` reads the uploaded JSON before choosing an output directory. If `metadata.job_id` in the JSON matches an existing `output/jobs/{job_id}/` folder, the report is written there (keeping all files for a call together). Otherwise a new `job_id` and folder are created.

### BaseException handling

Pipeline threads catch `BaseException` (not just `Exception`) because ctranslate2's CUDA teardown raises `SystemExit` on Windows when the WhisperModel is garbage-collected mid-process. This is the same bug handled by the `_active_model` module-level reference in `transcribe.py`.

## Frontend (job-joseph.com)

A web frontend at job-joseph.com consumes the API. It connects to the ngrok-exposed URL, uploads audio via `POST /analyse`, then opens a WebSocket for real-time progress. On job completion the `complete` message delivers the transcript and report inline.

**ngrok reconnect flow:** If the WebSocket drops (ngrok free tier timeout), the client calls `GET /reconnect/{job_id}` to retrieve the full job state and resume display without re-processing.

**Sub-modes:** "Analyse Audio" (full pipeline) and "Report from Transcript" (Stage 5 only from uploaded JSON). Switching between sub-modes clears all result state.

**Gemini model selector:** Available in advanced options. Selection is persisted to `localStorage` key `"cap_gemini_model"`. Helper text explains the tradeoffs (Flash for long calls, Pro for shorter calls where deeper analysis matters, Flash Lite for quick summaries).

**State reset between runs:** All state — `jobId`, transcript, report, error, and the active WebSocket — is fully cleared before each new run. A `runId` (set to `Date.now()` at run start) is compared on every incoming WebSocket message; messages from a previous run are discarded. "Try Again" and sub-mode switching both trigger a full reset.

## What NOT to commit

`.env`, `input/`, `output/`, `venv/`, `whisper_models/`, `*.mp3`, `*.wav`, `*.m4a`, `*.mpeg` — all covered by `.gitignore`.
