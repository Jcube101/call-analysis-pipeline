# Call Analysis Pipeline

A Python pipeline that takes a recorded conversation and produces a clean, speaker-diarized, timestamped transcript and an AI-generated analysis report. Runs fully locally with GPU acceleration (report generation uses the Claude API by default, with Gemini as an alternative).

**Status: v1.0 — fully functional and tested end-to-end (GPU, Windows 11, GTX 1650). Validated on recordings up to 3h42m (one measured run on this hardware, not a guarantee; see [Integration tests](#integration-tests-manual-completed)). Includes speaker re-identification and speaker name mapping.**

> **Picking this up after a break?** This machine has **no `venv/`** for this project — all dependencies live in the **global Python 3.12** install (`C:\Users\jobjo\AppData\Local\Programs\Python\Python312\python.exe`). There's also a separate **Python 3.11** on PATH (`...\Python311\python.exe`) that is *not* set up for this project — if `python`/`pip` resolve to 3.11 instead of 3.12, you'll hit `ModuleNotFoundError` for everything (`uvicorn`, `dotenv`, `torch`, ...) even though it "should" be installed. Check with `where python` — the 3.12 one needs to come first, or invoke it explicitly (`C:\Users\jobjo\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8010`). If you do see `ModuleNotFoundError`, just re-run the [install steps below](#clone-and-install) against the 3.12 interpreter — order matters (torch first, then pin numpy, then the rest).

## What it does

| Stage | Description |
|-------|-------------|
| 1 — Preprocess | Noise reduction + volume normalization via `noisereduce` and `soundfile`, decoded by a streaming `ffmpeg` pass |
| 2 — Diarize | Speaker separation using `pyannote/speaker-diarization-3.1`; global re-identification via voice embeddings + KMeans clustering to fix label flipping on long recordings |
| 3 — Transcribe | Transcription with faster-whisper (local, GPU-accelerated); two modes: `accurate` (default) and `fast` |
| 4 — Export | Structured `.txt` and `.json` output with metadata header, uniquely named per run |
| 5 — Report | AI-generated analysis report via Claude API (default) or Gemini API; triggered with `--report` |

## Output

Each run produces uniquely named files — no overwrites:
```
output/
├── First_Test_File_20260327_143022.txt          # Human-readable labeled transcript
├── First_Test_File_20260327_143022.json         # Structured JSON for downstream analysis
├── First_Test_File_20260327_143022_report.md    # AI analysis report (--report only)
└── First_Test_File_clean.wav                    # Noise-reduced audio (overwritten each run)
```

**transcript.txt** looks like:
```
# Call Transcript
# Source:  call.m4a
# Context: friend
# Speakers: 2
# Processed: 2026-03-27T14:30:22

[00:00:04] Speaker A: "Hey, how are you doing..."
[00:01:12] Speaker B: "I'm good, just got back from..."
```

**transcript.json** includes a metadata header and per-segment objects:
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
      "text": "Hey, how are you doing...",
      "confidence": 0.91,
      "words": [
        {"word": "Hey,", "start": 4.2, "end": 4.5, "probability": 0.99}
      ]
    }
  ]
}
```

`confidence` is always present (0–1 score). `words` is only included when `--word-timestamps` is passed.

If Stage 2's speaker re-identification only partly worked or failed, the JSON gets a `metadata.warnings` list and the `.txt` header gets one `# Warning: ...` line per warning (for example `# Warning: Speaker re-identification failed for 3 of 120 segments; labels may be less accurate.`). Clean runs have neither, so their files are unchanged. The same warnings appear in the Stage 5 report header.

## Two ways to use the pipeline

1. **Terminal** — `python main.py --input input/call.mp3`
2. **Web UI** — start the API with `.\start_server.ps1`, run `ngrok http 8010`, then open [job-joseph.com/projects/call-analysis](https://job-joseph.com/projects/call-analysis) and paste the ngrok URL into the Backend URL field

---

## Setup

### Prerequisites

- Python 3.9+ (the dev machine runs 3.12)
- PyTorch 2.5.1+cu124 (compatible with NVIDIA driver 581.15+)
- `ffmpeg` on your PATH
- [ngrok](https://ngrok.com/) (optional — only needed to expose the API server to a browser frontend)

**Install ffmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html and add to PATH

### Clone and install

```bash
git clone <repo-url>
cd call-analysis-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

**Install dependencies — order matters on Windows/CPU setups:**

```bash
# 1. PyTorch first (CUDA build; use +cpu suffix for CPU-only)
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# 2. Pin numpy before anything else can upgrade it
pip install "numpy<2.0" --force-reinstall

# 3. Remaining dependencies
pip install -r requirements.txt
```

> **Why the order?** If `pip install -r requirements.txt` runs first, PyPI resolves `torch` to the latest version which pulls in `numpy 2.x`, breaking `pyannote.audio`. Installing torch from the PyTorch wheel index first prevents this.

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

- `HUGGINGFACE_TOKEN` — required for speaker diarization. Get a token at https://huggingface.co/settings/tokens, then accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
- `ANTHROPIC_API_KEY` — required for `--report` with Claude models (default). Get a key at https://console.anthropic.com/settings/keys
- `GEMINI_API_KEY` — required for `--report` with Gemini models. Get a free key at https://aistudio.google.com/app/apikey
- Supported models (selectable via the web UI, the `analysis_model` API parameter, or `--report-model` on the CLI): `claude-haiku-5-5` (default, fast and cost-effective), `claude-sonnet-5-5` (deeper analysis), `gemini-3.8-flash` (reliable for long transcripts), `gemini-4-argon` (deeper analysis). The list lives in `models.py`. Old IDs from earlier versions are remapped to their successors.
- **Removed (2026-10-14):** the old API field `gemini_model` (renamed `analysis_model`, since it carries Claude and Gemini models) is no longer read. A request that still sends it is not rejected: the value is ignored, the default model is used unless `analysis_model` is also sent, and the server prints `[models] Ignored removed field 'gemini_model'; send 'analysis_model'`.
- `CONVERSATION_CONTEXT` — `friend`, `work`, `work_interview`, `user_interview`, `public_interview`, or `date`
- `NUM_SPEAKERS` — integer (e.g. `2`) or leave blank for auto-detection
- `TRANSCRIPTION_MODE` — `accurate` (default) or `fast`
- `WHISPER_LANGUAGE` — BCP-47 language code, e.g. `en`, `fr`, `es` (default: `en`)
- `SPEAKER_NAMES` — comma-separated real names, e.g. `Alice,Bob` (optional; replaces generic `Speaker A/B` labels)
- `WORD_TIMESTAMPS` — `true` to include per-word timestamps in JSON output (off by default; ~10-15% overhead)
- `RELEASE_WHISPER_AFTER_JOB` — API server only. On by default: the Whisper model is freed after each job. `0` keeps it loaded until the server exits

### Create local directories

```bash
mkdir input output
```

## Running the pipeline

Supports MP3, M4A, WAV, and any other ffmpeg-supported audio format:

```bash
python main.py --input input/your_recording.m4a
```

Optional overrides:

```bash
# Override context and speaker count
python main.py --input input/call.mp3 --context work --num-speakers 3

# Generate an AI analysis report after transcription
python main.py --input input/call.mp3 --context work --report

# Choose the report model (default: claude-haiku-5-5)
python main.py --input input/call.mp3 --report --report-model gemini-3.8-flash

# Use fast transcription mode (per-turn, 10-20x fewer Whisper calls than accurate)
python main.py --input input/call.mp3 --transcription-mode fast

# Replace generic speaker labels with real names
python main.py --input input/call.mp3 --speaker-names "Alice,Bob"

# Include per-word timestamps and probabilities in the JSON output
python main.py --input input/call.mp3 --word-timestamps

# Transcribe a non-English recording
python main.py --input input/call.mp3 --language fr

# Validate config and input without running the pipeline
python main.py --input input/call.mp3 --dry-run

# Skip noise reduction (re-run on an already-cleaned WAV)
python main.py --input output/call_clean.wav --skip-preprocess

# Generate a report from an existing transcript JSON (skips Stages 1–4 entirely)
python main.py --from-json output/call_20260327_143022.json
python main.py --from-json output/call_20260327_143022.json --context work --speaker-names "Alice,Bob"
```

The startup banner shows settings and device info:
```
============================================================
  Call Analysis Pipeline
  Input:    input/call.m4a
  Context:  work
  Speakers: 2
  Names:    Alice, Bob
  Whisper:  medium
  Tx Mode:  accurate
  Language: en
  Report:   ON (Stage 5 will run after transcription)
  CUDA:         available (NVIDIA GeForce GTX 1650)
  Diarization:  GPU (pyannote → CUDA)
  Transcription: GPU (faster-whisper int8_float16)
============================================================
```

The completion banner shows a full run summary:
```
============================================================
  Pipeline complete!
  Clean audio:  output/call_clean.wav
  Transcript:   output/call_20260327_143022.txt
  JSON:         output/call_20260327_143022.json
  Report:       output/call_20260327_143022_report.md
  Segments:     134  |  Speakers: 2  (Speaker A: 77  Speaker B: 57)
  Audio:        10:56  |  Elapsed: 11:32  (Stage 1: 7.5s  Stage 2: 56.3s  Stage 3: 623.8s  Stage 4: 0.0s  Stage 5: 4.1s)
============================================================
```

### Transcription modes

| Mode | How it works | Speed | Output quality |
|------|-------------|-------|----------------|
| `accurate` (default) | One Whisper call per diarization segment | ~1x real-time on GTX 1650 | Fine-grained, sentence-level lines |
| `fast` | Merges same-speaker segments into turns, one Whisper call per turn | 10-20x fewer calls than accurate | One line per speaker turn; speaker-accurate boundaries |

`accurate` is the right default for archival transcripts. `fast` is useful when you need a quick readable pass on a long recording — it still uses diarization for speaker attribution, just at turn granularity instead of segment granularity.

### Analysis report (Stage 5)

When `--report` is passed, Stage 5 sends the transcript to an LLM and writes a context-aware Markdown report. By default it uses `claude-haiku-5-5` via the Anthropic API; Gemini models are also supported. Pick the model with `--report-model` (CLI) or the `analysis_model` parameter (API); the provider comes from the registry in `models.py`. The prompt is loaded from `prompts/<context>.md` — edit these files freely to customise the analysis focus. The report header includes the model that was actually used (including any fallback).

Each prompt defines structured output sections and includes a speaker reliability warning (pyannote's diarisation can flip labels on long recordings — the LLM is instructed to base analysis on content, not label consistency).

| Context | Output sections |
|---------|----------------|
| `friend` | Summary, mood/energy, main topics, highlights, concerns, conversation dynamics, recurring themes |
| `work` | Executive summary, participants, decisions, action item table (what/owner/deadline), open questions, topics, risks, meeting effectiveness |
| `work_interview` | Speaker ID inference, candidate overview, strengths/concerns with evidence, comms style, cultural/motivational fit, candidate questions, suggested follow-ups, recommendation |
| `user_interview` | Speaker ID, participant profile, core problem, current behaviour and workarounds, trigger moments, goals, notable quotes, surprises/contradictions, open questions, signals summary |
| `public_interview` | Speaker ID, context and stakes, key messages pushed, tough question handling, evasion patterns, credibility/consistency, memorable quotes, journalist performance, overall assessment |
| `date` | Vibe, conversation balance, common ground, differences, highlights, awkward moments, self-revelations, green/red flags, overall assessment |

To generate a report on an already-processed transcript without re-running the pipeline:
```bash
python main.py --from-json output/call_20260327_143022.json --context work
```

### Merging a recording processed in two parts

`tools/merge_transcripts.py` joins two transcript JSONs from one recording (part 1 + part 2) into one file that `--from-json` or Report from Transcript can use:

```bash
python tools/merge_transcripts.py part1.json part2.json merged.json
python tools/merge_transcripts.py part1.json part2.json merged.json --gap 1.0 --swap-speakers "Speaker A" "Speaker B"
python tools/merge_transcripts.py part1.json part2.json merged.json --dry-run
```

Part 2's timestamps are offset to follow part 1 (plus `--gap` seconds, default 1.0). `--swap-speakers` swaps two labels in part 2 when its labels came out flipped. The output keeps part 1's metadata. `metadata.warnings` is replaced by both parts' warnings, each prefixed `Part 1: ` or `Part 2: `, and left out when neither part has any. This replaces the old untracked `output/merge_transcripts.py`.

### Swapping two speaker labels

`tools/swap_speakers.py` swaps two speaker labels throughout a transcript JSON and writes the result to a new file:

```bash
python tools/swap_speakers.py in.json out.json "Speaker A" "Speaker B" [--dry-run]
```

Use it when a single-mic recording came out with its labels flipped, as a quicker alternative to re-running `--from-json` or Report from Transcript with speaker names. It swaps every `speaker` field, a `label` field holding either name, and speaker lists such as `metadata.speaker_names`; everything else, `metadata.warnings` included, is unchanged. It prints how many segments changed each way, refuses to overwrite the input, and exits with an error if either label is not in the file or both are the same. `--dry-run` prints the summary and writes nothing.

### First-run note

The faster-whisper `medium` model (~1.5 GB) downloads automatically on first run. This is expected and only happens once.

### Known warnings

You may see this on startup:

```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
```

**Harmless.** pyannote 3.4.0 reads audio through `torchaudio`, which uses the `soundfile` backend here, so `torchcodec` is never needed — the warning is emitted on import regardless and is suppressed in the code.

## Running the API server

`api.py` exposes the pipeline over HTTP + WebSocket. Its dependencies (`fastapi`, `uvicorn`, `python-multipart`) are in `requirements.txt`.

Start the full stack in two terminals:

**Terminal 1 — API server:**
```powershell
.\start_server.ps1
```

### Starting the server

`start_server.ps1` (repo root, Windows PowerShell) runs `uvicorn api:app --host 0.0.0.0 --port 8010` from the repo folder with `PYTHONIOENCODING=utf-8`, in the foreground so the output is live. Ctrl+C stops it.

Before starting it:
- refuses to start if anything already listens on port 8010, and prints that process's PID, name and command line
- prints free RAM and commit headroom (RAM + page file), and warns below 6 GB of commit. A full job peaks at about 5.4 GB, and Windows fails allocations on commit, not on physical RAM (see KNOWN-ISSUES.md)
- uses the global Python 3.12 if present, and checks that it can import `uvicorn` and `fastapi`

```powershell
.\start_server.ps1            # visible window, no log file
.\start_server.ps1 -LogFile   # also append output to output\server_8010.log (UTF-8)
.\start_server.ps1 -Stop      # stop the server on 8010, only if it is this project's uvicorn
```

`-Stop` refuses any process whose command line does not show `uvicorn ... api:app --port 8010`, including a server started with `python api.py`. Stop that one with Ctrl+C in its window.

`PYTHONIOENCODING=utf-8` matters when output is piped (as with `-LogFile`): Windows gives piped stdout the cp1252 encoding, which cannot encode the em dashes and tqdm bar characters the stages print. `api.py` also switches its own streams to UTF-8 at startup. If you start uvicorn by hand with piped output, set the variable yourself.

After each job the server frees the Whisper model (`[job ...] Releasing Whisper model...` then `[job ...] Whisper model released; VRAM free X of Y GB`), so the next job starts with the full VRAM. Set `RELEASE_WHISPER_AFTER_JOB=0` to keep it loaded until the server exits. If a log ends on the first of those two lines, see KNOWN-ISSUES.md §2.

If scripts are blocked, run `powershell -ExecutionPolicy Bypass -File .\start_server.ps1`.

**Terminal 2 — ngrok tunnel:**
```bash
ngrok http 8010
```

Then paste the ngrok URL into the Backend URL field at **job-joseph.com/projects/call-analysis**.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyse` | Upload audio, start pipeline; returns `job_id` |
| POST | `/report-from-json` | Upload transcript JSON, run Stage 5 only; returns `job_id` |
| GET | `/status/{job_id}` | Poll job status |
| GET | `/reconnect/{job_id}` | Full job state recovery (transcript + report + metadata) |
| GET | `/download/{job_id}/transcript` | Download `.txt` transcript |
| GET | `/download/{job_id}/json` | Download transcript JSON |
| GET | `/download/{job_id}/report` | Download `.md` report |
| GET | `/download/{job_id}/wav` | Download clean audio |
| WS | `/ws/{job_id}` | WebSocket — real-time progress, `complete` message on finish |

Jobs run one at a time (GPU constraint). All outputs land in `output/jobs/{job_id}/`. Job state is persisted to disk so the server can recover it after a restart.

Both `POST /analyse` and `POST /report-from-json` accept an optional `context_hints` parameter (free text) for correct spelling of names, places, companies, and technical terms. When provided, the hints are prepended to the system prompt so the LLM uses them for proper nouns in the report.

### Progress and warnings

WebSocket `progress` messages carry `stage`, `stage_name` and `message`. During Stages 1–3 they also carry `phase` and `percent`:

| Stage | Phases (counted ones send a percent) |
|-------|--------------------------------------|
| 1 | `decode`, `noise reduction` (counted), `normalization` |
| 2 | `model load`, `segmentation` (counted), `speaker counting`, `embeddings` (counted), `clustering`, `re-identification` |
| 3 | `model load`, `transcription` (counted) |

A counted phase starts at `0` and updates at most every 5% and every 10 s. Other phases send `percent: null`. `100` is never sent: a phase ends when the next message arrives. Stage 5 sends only the heartbeat ping. The terminal shows the same stages as tqdm bars.

The `complete` message, `GET /reconnect/{job_id}` and `GET /status/{job_id}` all carry `warnings`, a list of short strings that is empty for a clean run. It is set when Stage 2's speaker re-identification is `partial` (one or more segments failed feature extraction and took their neighbour's label) or `failed` (labels are pyannote's own), or when a fixed `NUM_SPEAKERS` comes back short. `/status` also carries `reid`, the raw outcome with its counts. The server prints each one as `[job ...] Warning: ...`.

## Download endpoints

Once a job completes, all output files are downloadable:

```
GET /download/{job_id}/transcript  →  .txt transcript
GET /download/{job_id}/json        →  .json transcript (named JSON takes priority if present)
GET /download/{job_id}/report      →  .md AI report
GET /download/{job_id}/wav         →  clean audio (.wav)
```

Files are served directly from disk — no status check required. The server can be restarted and downloads still work as long as `output/jobs/{job_id}/` exists.

## Testing

### Unit tests

Run the pytest unit test suite:

```bash
pytest tests/ -v
```

250 tests across 9 modules. Covers: config loading, export formatting (including warning lines), API job recovery, the `analysis_model` field and the removed `gemini_model` field, re-identification warnings end to end, the Whisper release after each job, report prompt loading and warning headers, model fallback and the model registry, the progress throttle, the Stage 2 re-identification outcome, `tools/merge_transcripts.py`, `tools/swap_speakers.py`, and API key leak prevention across all output surfaces.

### Integration tests (manual, completed)

The following integration scenarios have been validated manually end-to-end:
- Full pipeline on real M4A files (2 min, 8 min, 30 min)
- Full pipeline on a 3h42m recording (job `b6d7567a`, 2026-10-17, untrimmed, 2 speakers, accurate mode; Windows 11, GTX 1650 4 GB, 7.4 GB RAM). One measured run on this hardware, not a guarantee of longer files:
  - Duration 13,356 s (3h42m36s). Total pipeline 2:11:20, about 0.6x real time.
  - Stage 1 noise reduction 2:23. Stage 2 diarization 15:27 (embeddings 13:38), 4,576 segments. Stage 3 transcription 1:52:51, 3,361 transcribed segments.
  - Stage 3 dominates the runtime, at roughly 1.5 s per diarization segment, so runtime grows with the amount of speech, not only the duration.
  - Stage 5 with `claude-haiku-5-5`: 227,838-character transcript, report in 0:38.
  - Whisper model released after the job: VRAM free 3.16 of 4.00 GB, process commit 3.10 GB at job end.
  - The commit-headroom preflight budget is about 5.4 GB plus 100 MB per audio hour (see [KNOWN-ISSUES.md](KNOWN-ISSUES.md) §1). Stage 2 still loads the whole waveform for the segmentation pass, so memory still grows with length. Recordings longer than 3h42m are untested.
- Stage 1 noise reduction producing clean WAV
- Stage 2 diarization identifying 2 speakers
- Stage 3 transcription in accurate and fast modes
- Stage 4 export producing valid TXT and JSON with correct schema
- Stage 5 report generation with Claude and Gemini models
- Two-pass approach (pipeline → review → rename → report)
- Report from Transcript mode with speaker renaming

### API tests (manual, completed)

The following API endpoints have been validated manually:
- `GET /health` → 200 OK
- `POST /analyse` → immediate `job_id` response, pipeline runs in background
- `POST /report-from-json` → writes report to original job folder
- `GET /status/{job_id}` → correct job state
- `GET /reconnect/{job_id}` → full job state including report content
- `GET /download/{job_id}/transcript` → correct `.txt` file
- `GET /download/{job_id}/json` → named JSON takes priority
- `GET /download/{job_id}/report` → correct `.md` file
- `WebSocket /ws/{job_id}` → real-time stage progress messages
- Job disk recovery after server restart

## Known limitations

- **Speaker diarization accuracy** depends on recording quality. Single-microphone recordings may have occasional speaker label flipping. Use `--from-json` with `--speaker-names` to correct labels after reviewing the transcript.
- **ngrok free tier** generates a new URL on every restart. Paste the new URL into the Backend URL field on the site.
- **WebSocket UI** may show a stale stage after an ngrok reconnect. The pipeline completes correctly — files are in `output/jobs/`.

## Exposing via ngrok

To make the API reachable from a browser (e.g. a deployed frontend):

```bash
ngrok http 8010
```

ngrok's free tier drops idle WebSocket connections after ~30 s. The API handles this with:
- A **heartbeat thread** during Stage 5 — sends a progress ping every 10 s to keep the WebSocket alive during the API call
- A **`/reconnect/{job_id}` endpoint** — returns the full job state so clients can recover without re-processing

## Project structure

```
call-analysis-pipeline/
├── main.py               # Entry point — runs the full pipeline (CLI)
├── api.py                # FastAPI HTTP + WebSocket wrapper
├── start_server.ps1      # Starts/stops the API on port 8010 with pre-checks (Windows)
├── config.py             # Loads .env, exposes typed settings
├── models.py             # Stage 5 model registry: IDs, labels, default, fallbacks, legacy aliases
├── stages/
│   ├── preprocess.py     # Stage 1: noise reduction + normalization
│   ├── diarize.py        # Stage 2: speaker diarization
│   ├── reid_outcome.py   # Stage 2 re-identification outcome (ok/partial/failed) and warning text
│   ├── transcribe.py     # Stage 3: faster-whisper transcription
│   ├── export.py         # Stage 4: output formatting
│   ├── report.py         # Stage 5: Claude/Gemini API analysis report
│   └── progress.py       # Phase/percent progress reporting for Stages 1 and 3
├── tools/
│   ├── merge_transcripts.py  # Merge part 1 + part 2 transcript JSONs
│   └── swap_speakers.py      # Swap two speaker labels in a transcript JSON
├── tests/                # pytest unit tests (250)
├── prompts/
│   ├── friend.md              # Analysis prompt for friend conversations
│   ├── work.md                # Analysis prompt for work conversations
│   ├── work_interview.md      # Analysis prompt for job interviews
│   ├── user_interview.md      # Analysis prompt for user/PM interviews
│   ├── public_interview.md    # Analysis prompt for public/media interviews
│   └── date.md                # Analysis prompt for date conversations
├── input/                # Place your audio files here (gitignored)
├── output/               # Pipeline outputs land here (gitignored)
├── .env.example          # Template — copy to .env and fill in values
├── requirements.txt
└── README.md
```
