# Call Analysis Pipeline

A Python pipeline that takes a recorded conversation and produces a clean, speaker-diarized, timestamped transcript and an AI-generated analysis report. Runs fully locally with GPU acceleration (report generation uses the Gemini API).

**Status: v1.0 — fully functional and tested end-to-end (GPU, Windows 11, GTX 1650). Validated on recordings up to 2h40m. Includes speaker re-identification and speaker name mapping.**

## What it does

| Stage | Description |
|-------|-------------|
| 1 — Preprocess | Noise reduction + volume normalization via `noisereduce` and `pydub` |
| 2 — Diarize | Speaker separation using `pyannote/speaker-diarization-3.1`; global re-identification via voice embeddings + KMeans clustering to fix label flipping on long recordings |
| 3 — Transcribe | Transcription with faster-whisper (local, GPU-accelerated); two modes: `accurate` (default) and `fast` |
| 4 — Export | Structured `.txt` and `.json` output with metadata header, uniquely named per run |
| 5 — Report | AI-generated analysis report via Gemini API; triggered with `--report` |

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

## Setup

### Prerequisites

- Python 3.9+ (tested on 3.11)
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
# 1. PyTorch first (CPU build)
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. Pin numpy before anything else can upgrade it
pip install "numpy<2.0" --force-reinstall

# 3. Remaining dependencies
pip install -r requirements.txt
```

> **Why the order?** If `pip install -r requirements.txt` runs first, PyPI resolves `torch` to the latest version which pulls in `numpy 2.x`, breaking `pyannote.audio`. Installing torch from the CPU wheel index first prevents this.

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

- `HUGGINGFACE_TOKEN` — required for speaker diarization. Get a token at https://huggingface.co/settings/tokens, then accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
- `GEMINI_API_KEY` — required for `--report`. Get a free key at https://aistudio.google.com/app/apikey
- `CONVERSATION_CONTEXT` — `friend`, `work`, `interview`, or `date`
- `NUM_SPEAKERS` — integer (e.g. `2`) or leave blank for auto-detection
- `TRANSCRIPTION_MODE` — `accurate` (default) or `fast`
- `WHISPER_LANGUAGE` — BCP-47 language code, e.g. `en`, `fr`, `es` (default: `en`)
- `SPEAKER_NAMES` — comma-separated real names, e.g. `Alice,Bob` (optional; replaces generic `Speaker A/B` labels)
- `WORD_TIMESTAMPS` — `true` to include per-word timestamps in JSON output (off by default; ~10-15% overhead)

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

When `--report` is passed, Stage 5 sends the transcript to `gemini-3-flash-preview` and writes a context-aware Markdown report. The prompt is loaded from `prompts/<context>.md` — edit these files freely to customise the analysis focus.

Each prompt defines structured output sections and includes a speaker reliability warning (pyannote's diarisation can flip labels on long recordings — Gemini is instructed to base analysis on content, not label consistency).

| Context | Output sections |
|---------|----------------|
| `friend` | Summary, mood/energy, main topics, highlights, concerns, conversation dynamics, recurring themes |
| `work` | Executive summary, participants, decisions, action item table (what/owner/deadline), open questions, topics, risks, meeting effectiveness |
| `interview` | Speaker ID inference, candidate overview, strengths/concerns with evidence, comms style, cultural/motivational fit, candidate questions, suggested follow-ups, recommendation |
| `date` | Vibe, conversation balance, common ground, differences, highlights, awkward moments, self-revelations, green/red flags, overall assessment |
| `public_interview` | Speaker ID, context and stakes, key messages pushed, tough question handling, evasion patterns, credibility/consistency, memorable quotes, journalist performance, overall assessment |

To generate a report on an already-processed transcript without re-running the pipeline:
```bash
python main.py --from-json output/call_20260327_143022.json --context work
```

### First-run note

The faster-whisper `medium` model (~1.5 GB) downloads automatically on first run. This is expected and only happens once.

### Known warnings

You may see this on startup:

```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
```

**Harmless.** The pipeline passes audio to pyannote as a pre-loaded in-memory waveform, so `torchcodec` is never used. The warning is suppressed in the code.

## Running the API server

`api.py` exposes the pipeline over HTTP + WebSocket so frontends can drive it programmatically:

```bash
# Install additional API dependencies
pip install fastapi uvicorn python-multipart

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyse` | Upload audio, start pipeline; returns `job_id` |
| POST | `/report-from-json` | Upload transcript JSON, run Stage 5 only; returns `job_id` |
| GET | `/status/{job_id}` | Poll job status |
| GET | `/reconnect/{job_id}` | Full job state recovery (transcript + report + metadata) |
| GET | `/download/{job_id}/{type}` | Download output — `type` ∈ `txt`, `json`, `report`, `wav` |
| WS | `/ws/{job_id}` | WebSocket — real-time progress, `complete` message on finish |

Jobs run one at a time (GPU constraint). All outputs land in `output/jobs/{job_id}/`. Job state is persisted to disk so the server can recover it after a restart.

## Exposing via ngrok

To make the API reachable from a browser (e.g. a deployed frontend):

```bash
ngrok http 8000
```

ngrok's free tier drops idle WebSocket connections after ~30 s. The API handles this with:
- A **heartbeat thread** during Stage 5 — sends a progress ping every 10 s to keep the WebSocket alive during the Gemini API call
- A **`/reconnect/{job_id}` endpoint** — returns the full job state so clients can recover without re-processing

## Project structure

```
call-analysis-pipeline/
├── main.py               # Entry point — runs the full pipeline (CLI)
├── api.py                # FastAPI HTTP + WebSocket wrapper
├── config.py             # Loads .env, exposes typed settings
├── stages/
│   ├── preprocess.py     # Stage 1: noise reduction + normalization
│   ├── diarize.py        # Stage 2: speaker diarization
│   ├── transcribe.py     # Stage 3: faster-whisper transcription
│   ├── export.py         # Stage 4: output formatting
│   └── report.py         # Stage 5: Gemini API analysis report
├── prompts/
│   ├── friend.md         # Analysis prompt for friend conversations
│   ├── work.md           # Analysis prompt for work conversations
│   ├── interview.md      # Analysis prompt for job interviews
│   └── date.md           # Analysis prompt for date conversations
├── input/                # Place your audio files here (gitignored)
├── output/               # Pipeline outputs land here (gitignored)
├── .env.example          # Template — copy to .env and fill in values
├── requirements.txt
└── README.md
```
