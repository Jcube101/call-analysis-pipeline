# Call Analysis Pipeline

A Python pipeline that takes a recorded conversation and produces a clean, speaker-diarized, timestamped transcript ready for post-analysis. Runs fully locally with GPU acceleration.

**Status: v0.1 — fully functional and tested end-to-end (GPU, Windows 11, GTX 1650).**

## What it does

| Stage | Description |
|-------|-------------|
| 1 — Preprocess | Noise reduction + volume normalization via `noisereduce` and `pydub` |
| 2 — Diarize | Speaker separation using `pyannote/speaker-diarization-3.1`; per-speaker loudness normalization |
| 3 — Transcribe | Per-segment transcription with faster-whisper (local, GPU-accelerated) |
| 4 — Export | Structured `.txt` and `.json` output with metadata header, uniquely named per run |

Future stage (not yet implemented): auto-generated analysis report via the Claude API.

## Output

Each run produces uniquely named files — no overwrites:
```
output/
├── First_Test_File_20260327_143022.txt   # Human-readable labeled transcript
├── First_Test_File_20260327_143022.json  # Structured JSON for downstream analysis
└── First_Test_File_clean.wav             # Noise-reduced audio (overwritten each run)
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

**Install dependencies — order matters:**

```bash
# 1. PyTorch with CUDA 12.1 (for NVIDIA GPU)
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU):
# pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. Pin numpy before anything else upgrades it
pip install "numpy<2.0" --force-reinstall

# 3. Remaining dependencies
pip install -r requirements.txt
```

> **Why the order?** pip will resolve `torch` to the latest version if not pre-installed, pulling in numpy 2.x which breaks pyannote. Installing torch first prevents this.

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

- `HUGGINGFACE_TOKEN` — required for speaker diarization. Get a token at https://huggingface.co/settings/tokens, then accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
- `ANTHROPIC_API_KEY` — reserved for future use
- `CONVERSATION_CONTEXT` — `friend`, `work`, `interview`, or `date`
- `NUM_SPEAKERS` — integer (e.g. `2`) or leave blank for auto-detection

### Create local directories

```bash
mkdir input output
```

## Running the pipeline

Supports MP3, M4A, WAV, and any ffmpeg-supported format:

```bash
python main.py --input input/your_recording.m4a
```

Optional overrides:

```bash
python main.py --input input/call.mp3 --context work --num-speakers 3
```

The startup banner shows which device each stage will use:
```
============================================================
  Call Analysis Pipeline
  Input:    input/call.m4a
  Context:  friend
  Speakers: 2
  Whisper:  medium
  CUDA:         available (NVIDIA GeForce GTX 1650)
  Diarization:  GPU (pyannote → CUDA)
  Transcription: GPU (faster-whisper int8_float16)
============================================================
```

### First-run note

The faster-whisper `medium` model (~1.5 GB) downloads automatically on first run. This is expected and only happens once.

### Known warnings

You may see:
```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
```
**Harmless** — audio is passed as a pre-loaded waveform so torchcodec is never used.

## Project structure

```
call-analysis-pipeline/
├── main.py               # Entry point — runs the full pipeline
├── config.py             # Loads .env, exposes typed settings
├── stages/
│   ├── preprocess.py     # Stage 1: noise reduction + normalization
│   ├── diarize.py        # Stage 2: speaker diarization
│   ├── transcribe.py     # Stage 3: faster-whisper transcription
│   └── export.py         # Stage 4: output formatting
├── input/                # Place your audio files here (gitignored)
├── output/               # Pipeline outputs land here (gitignored)
├── .env.example          # Template — copy to .env and fill in values
├── requirements.txt
└── README.md
```
