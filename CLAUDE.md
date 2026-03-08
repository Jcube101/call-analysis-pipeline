# CLAUDE.md — Project Context for Claude Code

This file gives Claude context about the call-analysis-pipeline project so it can assist effectively without re-exploring the codebase each session.

## What this project does

Takes an audio recording of a two-person conversation (MP3, M4A, WAV, or any ffmpeg-supported format) and outputs:
1. A noise-reduced WAV (`output/*_clean.wav`)
2. A speaker-diarized, timestamped transcript (`output/transcript.txt`)
3. A structured JSON file ready for downstream analysis (`output/transcript.json`)

Future: auto-generate an analysis report via the Claude API (not yet implemented).

## Status

**v0.1 is fully functional and tested end-to-end** on a real M4A call recording. All four stages run correctly on Windows with CPU-only torch.

## How to run

```bash
python main.py --input input/call.mp3
# Optional overrides:
python main.py --input input/call.mp3 --context work --num-speakers 3
```

All config (tokens, context, speaker count) lives in `.env`. See `.env.example`.

## Project layout

```
main.py          — entry point, orchestrates all stages
config.py        — Settings dataclass, loads .env via python-dotenv
stages/
  preprocess.py  — Stage 1: noise reduction + normalization (pydub, noisereduce)
  diarize.py     — Stage 2: speaker diarization (pyannote/speaker-diarization-3.1)
  transcribe.py  — Stage 3: Whisper transcription (openai-whisper, runs locally)
  export.py      — Stage 4: writes .txt and .json output with metadata header
input/           — place source audio files here (gitignored, must exist locally)
output/          — pipeline outputs land here (gitignored, must exist locally)
```

## Key conventions

- **All secrets and config come from `.env`** — nothing hardcoded. Config is accessed via the `settings` singleton in `config.py`.
- **Each stage is a module** with a single `run()` function. `main.py` calls them in order and passes the output of one stage as input to the next.
- **Segment dicts** are the internal data structure passed between stages. Each is a dict with keys `start`, `end`, `speaker`, `label` (and `text` after Stage 3).
- **Windows compatibility** — avoid Unix-only shell commands or hardcoded `/` paths; use `os.path` instead.
- **No chunking yet** — large file support is deferred, but don't architect it out. Keep it in mind when touching Stage 1 or 3.

## pyannote API compatibility (important)

pyannote.audio 3.x no longer returns a `pyannote.core.Annotation` directly from the pipeline. It returns a `DiarizeOutput` dataclass with the annotation stored in the `exclusive_speaker_diarization` attribute. `diarize.py` handles this with a fallback chain:

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
| `ANTHROPIC_API_KEY` | No (future) | For report generation stage |
| `CONVERSATION_CONTEXT` | No | `friend` / `work` / `interview` / `date` (default: `friend`) |
| `NUM_SPEAKERS` | No | Integer or blank for auto-detect (default: auto) |
| `WHISPER_MODEL` | No | `tiny` / `base` / `small` / `medium` / `large` (default: `medium`) |

## Dependencies

Key packages: `pydub`, `noisereduce`, `pyannote.audio`, `openai-whisper`, `torch`, `soundfile`, `librosa`, `python-dotenv`, `tqdm`.
System dependency: `ffmpeg` must be on PATH (`main.py` checks this on startup).

Whisper's medium model (~1.5 GB) downloads automatically on first run to the default Whisper cache dir.

## Install order matters (Windows / CPU-only)

Install torch **before** everything else so pip doesn't pull in a newer incompatible version later:

```bash
pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

## What NOT to commit

`.env`, `input/`, `output/`, `venv/`, `whisper_models/`, `*.mp3`, `*.wav`, `*.m4a` — all covered by `.gitignore`.
