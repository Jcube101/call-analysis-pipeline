# Call Analysis Pipeline

A Python pipeline that takes a recorded conversation and produces a clean, speaker-diarized, timestamped transcript ready for post-analysis.

**Status: v0.1 — fully functional and tested end-to-end.**

## What it does

| Stage | Description |
|-------|-------------|
| 1 — Preprocess | Noise reduction + volume normalization via `noisereduce` and `pydub` |
| 2 — Diarize | Speaker separation using `pyannote/speaker-diarization-3.1`; per-speaker loudness normalization |
| 3 — Transcribe | Per-segment transcription with OpenAI Whisper (local, runs offline) |
| 4 — Export | Structured `.txt` and `.json` output with metadata header |

Future stage (not yet implemented): auto-generated analysis report via the Claude API.

## Output

```
output/
├── <name>_clean.wav    # Noise-reduced audio
├── transcript.txt      # Human-readable labeled transcript
└── transcript.json     # Structured JSON for downstream analysis
```

**transcript.txt** looks like:
```
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
    "processed_at": "2024-01-15T14:30:00"
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

- Python 3.9+
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
- `ANTHROPIC_API_KEY` — reserved for future use
- `CONVERSATION_CONTEXT` — `friend`, `work`, `interview`, or `date`
- `NUM_SPEAKERS` — integer (e.g. `2`) or leave blank for auto-detection

### Create local directories

The `input/` and `output/` directories are gitignored. Create them manually:

```bash
mkdir input output
```

## Running the pipeline

Supports MP3, M4A, WAV, and any other ffmpeg-supported audio format:

```bash
python main.py --input input/your_recording.m4a
```

Optional overrides (take precedence over `.env`):

```bash
python main.py --input input/call.mp3 --context work --num-speakers 3
```

### First-run note

Whisper's `medium` model (~1.5 GB) downloads automatically on first run. This is expected and only happens once.

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
│   ├── transcribe.py     # Stage 3: Whisper transcription
│   └── export.py         # Stage 4: output formatting
├── input/                # Place your audio files here (gitignored)
├── output/               # Pipeline outputs land here (gitignored)
├── .env.example          # Template — copy to .env and fill in values
├── requirements.txt
└── README.md
```
