# Call Analysis Pipeline

A Python pipeline that takes an MP3 recording of a conversation between two people and produces a clean, speaker-diarized, timestamped transcript ready for post-analysis.

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
├── call_clean.wav          # Noise-reduced audio
├── transcript.txt          # Human-readable labeled transcript
└── transcript.json         # Structured JSON for downstream analysis
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
    "source_file": "test_call.mp3",
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

# Install dependencies
pip install -r requirements.txt
```

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

```bash
python main.py --input input/your_recording.mp3
```

Optional overrides (take precedence over `.env`):

```bash
python main.py --input input/call.mp3 --context work --num-speakers 3
```

### First-run note

Whisper's `medium` model (~1.5 GB) downloads automatically on first run. This is expected and only happens once.

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
├── input/                # Place your .mp3 files here (gitignored)
├── output/               # Pipeline outputs land here (gitignored)
├── .env.example          # Template — copy to .env and fill in values
├── requirements.txt
└── README.md
```
