# CLAUDE.md — Project Context for Claude Code

This file gives Claude context about the call-analysis-pipeline project so it can assist effectively without re-exploring the codebase each session.

## What this project does

Takes an audio recording of a conversation (MP3, M4A, WAV, or any ffmpeg-supported format) and outputs:
1. A noise-reduced WAV (`output/*_clean.wav`)
2. A speaker-diarized, timestamped transcript (`output/<name>_<timestamp>.txt`)
3. A structured JSON file ready for downstream analysis (`output/<name>_<timestamp>.json`)

Future: auto-generate an analysis report via the Claude API (not yet implemented).

## Status

**v0.1 is fully functional and tested end-to-end with GPU acceleration** on a real M4A call recording (GTX 1650, CUDA 12.1, Windows 11, Python 3.11).

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
  transcribe.py  — Stage 3: faster-whisper transcription (runs locally, GPU-accelerated)
  export.py      — Stage 4: writes timestamped .txt and .json output
input/           — place source audio files here (gitignored, must exist locally)
output/          — pipeline outputs land here (gitignored, must exist locally)
```

## Key conventions

- **All secrets and config come from `.env`** — nothing hardcoded. Config is accessed via the `settings` singleton in `config.py`.
- **Each stage is a module** with a single `run()` function. `main.py` calls them in order and passes the output of one stage as input to the next.
- **Segment dicts** are the internal data structure passed between stages. Each is a dict with keys `start`, `end`, `speaker`, `label` (and `text` after Stage 3).
- **Windows compatibility** — avoid Unix-only shell commands or hardcoded `/` paths; use `os.path` instead.
- **No chunking yet** — large file support is deferred, but don't architect it out. Keep it in mind when touching Stage 1 or 3.

## Output filenames

Each run produces uniquely named files using the source filename + timestamp:
```
output/First_Test_File_20260327_143022.txt
output/First_Test_File_20260327_143022.json
output/First_Test_File_clean.wav
```
The clean WAV is overwritten each run (Stage 1 output). The transcripts are never overwritten.

## GPU / transcription stack

Transcription uses **faster-whisper** (CTranslate2-based) instead of openai-whisper:
- On CUDA: `device="cuda", compute_type="int8_float16"` — fits in 4 GB VRAM
- On CPU: `device="cpu", compute_type="int8"`

**ctranslate2 CUDA teardown bug (Windows):** ctranslate2's `__del__` calls `exit()` when the WhisperModel is garbage-collected mid-process on Windows. Fixed by holding a module-level reference (`_active_model`) so cleanup is deferred to process exit. Do not add `del model` inside `transcribe.run()`.

## pyannote API compatibility

pyannote.audio 3.x no longer returns a `pyannote.core.Annotation` directly — it returns a `DiarizeOutput` dataclass. `diarize.py` handles this with a fallback chain checking `itertracks` → `exclusive_speaker_diarization` → `speaker_diarization`.

## HuggingFace authentication

pyannote 3.4.0 + huggingface_hub 0.x: `Pipeline.from_pretrained()` no longer accepts `token=` or `use_auth_token=` as keyword arguments. Use `huggingface_hub.login(token=...)` before calling `from_pretrained()`:

```python
huggingface_hub.login(token=settings.huggingface_token)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
```

## Audio input to pyannote

Audio is passed as a pre-loaded in-memory dict — not a file path — to avoid the `torchcodec` dependency:
```python
audio_input = {"waveform": waveform, "sample_rate": sample_rate}
diarization = pipeline(audio_input, ...)
```
The `torchcodec` UserWarning on import is suppressed since it is irrelevant to this path.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_TOKEN` | Yes | For pyannote diarization model download |
| `ANTHROPIC_API_KEY` | No (future) | For report generation stage |
| `CONVERSATION_CONTEXT` | No | `friend` / `work` / `interview` / `date` (default: `friend`) |
| `NUM_SPEAKERS` | No | Integer or blank for auto-detect (default: auto) |
| `WHISPER_MODEL` | No | `tiny` / `base` / `small` / `medium` / `large` (default: `medium`) |

## Dependencies and install order

Key packages: `pydub`, `noisereduce`, `pyannote.audio`, `faster-whisper`, `torch`, `soundfile`, `librosa`, `python-dotenv`, `tqdm`.
System dependency: `ffmpeg` must be on PATH.

**Install order matters — run in this exact sequence:**
```bash
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" --force-reinstall
pip install -r requirements.txt
```

Key version constraints (all in `requirements.txt`):
- `torch==2.1.0+cu121` — newer torch requires numpy 2.x which breaks pyannote
- `numpy<2.0` — pyannote compiled against numpy 1.x
- `pyannote.audio<4.0` — 4.0.4 requires torch>=2.8.0 which doesn't exist yet
- `huggingface_hub<1.0.0` — 1.x removed `use_auth_token` used internally by pyannote 3.x

## What NOT to commit

`.env`, `input/`, `output/`, `venv/`, `whisper_models/`, `*.mp3`, `*.wav`, `*.m4a` — all covered by `.gitignore`.
