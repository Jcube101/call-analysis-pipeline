# Call Analysis Pipeline

A Python pipeline that takes a recorded conversation and produces a clean, speaker-diarized, timestamped transcript and an AI-generated analysis report. Runs fully locally with GPU acceleration (report generation uses the Gemini API).

**Status: v0.3 — fully functional and tested end-to-end (GPU, Windows 11, GTX 1650).**

## What it does

| Stage | Description |
|-------|-------------|
| 1 — Preprocess | Noise reduction + volume normalization via `noisereduce` and `pydub` |
| 2 — Diarize | Speaker separation using `pyannote/speaker-diarization-3.1`; per-speaker loudness normalization |
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
      "text": "Hey, how are you doing..."
    }
  ]
}
```

## Setup

### Prerequisites

- Python 3.9+ (tested on 3.11)
- `ffmpeg` on your PATH

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

# Use fast transcription mode (coarser output, ~20% faster)
python main.py --input input/call.mp3 --transcription-mode fast

# Transcribe a non-English recording
python main.py --input input/call.mp3 --language fr

# Validate config and input without running the pipeline
python main.py --input input/call.mp3 --dry-run

# Skip noise reduction (re-run on an already-cleaned WAV)
python main.py --input output/call_clean.wav --skip-preprocess

# Skip preprocessing and generate report only (transcript already exists)
python main.py --input output/call_clean.wav --context work --report --skip-preprocess
```

The startup banner shows settings and device info:
```
============================================================
  Call Analysis Pipeline
  Input:    input/call.m4a
  Context:  work
  Speakers: 2
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
| `fast` | One Whisper call for the full file | ~20% faster | Coarser — ~1 line per 30s chunk |

For most use cases `accurate` is the right choice. `fast` is available for quick previews where exact line count doesn't matter.

### Analysis report (Stage 5)

When `--report` is passed, Stage 5 sends the transcript to `gemini-3-flash-preview` and writes a context-aware Markdown report. The analysis prompt is loaded from `prompts/<context>.md` — edit these files to customise what Gemini focuses on for each conversation type.

| Context | Default focus |
|---------|--------------|
| `friend` | Emotional tone, recurring themes, mood, conversation balance |
| `work` | Action items, decisions made, open questions, risks |
| `interview` | Candidate strengths/weaknesses, follow-up questions, recommendation |
| `date` | Compatibility signals, shared interests, conversation flow |

### First-run note

The faster-whisper `medium` model (~1.5 GB) downloads automatically on first run. This is expected and only happens once.

### Known warnings

You may see:
```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
```
**Harmless** — audio is passed as a pre-loaded waveform so torchcodec is never used.

### Known warnings

You may see this on startup:

```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
```

**This is harmless.** The pipeline passes audio to pyannote as a pre-loaded in-memory waveform (not via file path), so `torchcodec` is never used. The warning is suppressed in the code.

## Project structure

```
call-analysis-pipeline/
├── main.py               # Entry point — runs the full pipeline
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
