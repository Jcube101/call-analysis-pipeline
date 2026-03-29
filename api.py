"""
api.py — FastAPI HTTP + WebSocket wrapper for the call analysis pipeline.

Endpoints
---------
  POST /analyse              Upload audio file; runs Stages 1–4 (and 5 if report=true)
  POST /report-from-json     Upload existing transcript JSON; runs Stage 5 only
  GET  /health               {"status": "ok"}
  GET  /status/{job_id}      Full job dict (status, files, error)
  GET  /download/{job_id}/{type}  Stream transcript / json / report file
  WS   /ws/{job_id}          Push progress / complete / error events

Job lifecycle
-------------
  queued → running → complete | error

WebSocket message shapes
------------------------
  {"type": "progress", "stage": 1, "stage_name": "Preprocess", "message": "..."}
  {"type": "complete", "job_id": "...", "files": {"transcript": "...", "json": "..."}}
  {"type": "error", "message": "..."}

GPU constraint
--------------
  ThreadPoolExecutor(max_workers=1) serialises all pipeline jobs because only
  one pipeline can use the GPU at a time.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import VALID_CONTEXTS, Settings, settings

# Snapshot of .env defaults taken once at startup — reused on every job reset
# to avoid re-parsing environment variables per request.
_default_settings: dict = {}
from stages import diarize, export, preprocess, report, transcribe


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}
connections: dict[str, WebSocket] = {}
executor = ThreadPoolExecutor(max_workers=1)
_loop: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _default_settings
    _loop = asyncio.get_running_loop()
    _default_settings = Settings().__dict__.copy()
    os.makedirs("output/jobs", exist_ok=True)
    yield


app = FastAPI(title="Call Analysis Pipeline API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_speaker_names(segments: list[dict], names: list[str]) -> list[dict]:
    """Replace generic Speaker A/B labels with real names in alphabetical label order."""
    unique_speakers = sorted(set(s["speaker"] for s in segments))
    name_map = {spk: names[i] for i, spk in enumerate(unique_speakers) if i < len(names)}
    if not name_map:
        return segments
    for seg in segments:
        if seg["speaker"] in name_map:
            seg["speaker"] = name_map[seg["speaker"]]
    return segments


def _configure_settings(params: dict) -> None:
    """
    Reset the settings singleton to .env defaults, then apply job-specific overrides.

    With max_workers=1 only one job runs at a time, so mutating the shared singleton
    is safe — no concurrent pipeline execution can race against us.
    """
    settings.__dict__.update(_default_settings)
    settings.override(
        context=params.get("context"),
        num_speakers=params.get("num_speakers"),
        transcription_mode=params.get("transcription_mode"),
        language=params.get("language"),
        speaker_names=params.get("speaker_names") or None,
        word_timestamps=params.get("word_timestamps"),
    )
    if params.get("whisper_model"):
        settings.whisper_model = params["whisper_model"]


def _push_progress(job_id: str, stage: int, stage_name: str, message: str) -> None:
    """Push a progress event to the WebSocket client for this job (thread-safe)."""
    ws = connections.get(job_id)
    if ws is None or _loop is None:
        return
    payload = json.dumps({
        "type": "progress",
        "stage": stage,
        "stage_name": stage_name,
        "message": message,
    })
    asyncio.run_coroutine_threadsafe(ws.send_text(payload), _loop)


def _push_complete(job_id: str) -> None:
    ws = connections.get(job_id)
    if ws is None or _loop is None:
        return
    payload = json.dumps({
        "type": "complete",
        "job_id": job_id,
        "files": jobs[job_id].get("files", {}),
    })
    asyncio.run_coroutine_threadsafe(ws.send_text(payload), _loop)


def _push_error(job_id: str, message: str) -> None:
    ws = connections.get(job_id)
    if ws is None or _loop is None:
        return
    payload = json.dumps({"type": "error", "message": message})
    asyncio.run_coroutine_threadsafe(ws.send_text(payload), _loop)


# ---------------------------------------------------------------------------
# Pipeline threads
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str, input_path: str, params: dict) -> None:
    """Run the full pipeline (Stages 1–4, optionally 5). Executes in the thread pool."""
    job = jobs[job_id]
    job["status"] = "running"
    output_dir = f"output/jobs/{job_id}"

    try:
        _configure_settings(params)

        # Stage 1 — Pre-processing
        _push_progress(job_id, 1, "Preprocess", "Noise reduction and normalisation…")
        if params.get("skip_preprocess"):
            clean_wav = input_path
        else:
            clean_wav = preprocess.run(input_path=input_path, output_dir=output_dir)
        _push_progress(job_id, 1, "Preprocess", f"Done → {os.path.basename(clean_wav)}")

        # Stage 2 — Diarization
        _push_progress(job_id, 2, "Diarize", "Speaker diarization…")
        segments = diarize.run(
            clean_wav_path=clean_wav,
            output_dir=output_dir,
            num_speakers=settings.num_speakers,
        )
        if settings.speaker_names:
            segments = _apply_speaker_names(segments, settings.speaker_names)
        _push_progress(job_id, 2, "Diarize", f"Done — {len(segments)} segment(s)")

        # Stage 3 — Transcription
        _push_progress(job_id, 3, "Transcribe", f"Transcribing ({settings.transcription_mode} mode)…")
        transcribed_segments = transcribe.run(
            clean_wav_path=clean_wav,
            segments=segments,
            model_size=settings.whisper_model,
            mode=settings.transcription_mode,
            word_timestamps=settings.word_timestamps,
        )
        _push_progress(job_id, 3, "Transcribe", f"Done — {len(transcribed_segments)} segment(s)")

        # Stage 4 — Export
        _push_progress(job_id, 4, "Export", "Writing transcript files…")
        txt_path, json_path = export.run(
            segments=transcribed_segments,
            source_file=input_path,
            output_dir=output_dir,
            context=settings.context,
            num_speakers=settings.num_speakers,
            speaker_names=settings.speaker_names or None,
        )
        _push_progress(job_id, 4, "Export", "Done")

        files: dict = {"transcript": txt_path, "json": json_path, "clean_wav": clean_wav}

        # Stage 5 — Report (optional)
        if params.get("report"):
            settings.validate_for_report()
            _push_progress(job_id, 5, "Report", "Generating Gemini analysis report…")
            speaker_counts = Counter(s["speaker"] for s in transcribed_segments)
            try:
                import soundfile as _sf
                _info = _sf.info(clean_wav)
                audio_duration: Optional[float] = _info.frames / _info.samplerate
            except Exception:
                audio_duration = None
            report_path = report.run(
                segments=transcribed_segments,
                source_file=input_path,
                output_dir=output_dir,
                context=settings.context,
                num_speakers=len(speaker_counts),
                audio_duration=audio_duration,
                speaker_counts=dict(speaker_counts),
                api_key=settings.gemini_api_key,
                prompts_dir="prompts",
            )
            files["report"] = report_path
            _push_progress(job_id, 5, "Report", "Done")

        job["status"] = "complete"
        job["files"] = files
        _push_complete(job_id)

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        _push_error(job_id, str(exc))


def _run_report_from_json(job_id: str, json_path: str, params: dict) -> None:
    """Run Stage 5 only against an existing transcript JSON. Executes in the thread pool."""
    job = jobs[job_id]
    job["status"] = "running"
    output_dir = f"output/jobs/{job_id}"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_meta = data.get("metadata", {})
        transcribed_segments = data.get("transcript", [])
        if not transcribed_segments:
            raise ValueError("No transcript segments found in the uploaded JSON")

        json_context = json_meta.get("context", "friend")
        json_num_speakers = json_meta.get("num_speakers")
        json_source_file = json_meta.get("source_file", os.path.basename(json_path))

        _configure_settings(params)

        # Inherit context and num_speakers from the JSON when not explicitly overridden
        if params.get("context") is None and json_context in VALID_CONTEXTS:
            settings.context = json_context
        if params.get("num_speakers") is None and json_num_speakers is not None:
            settings.num_speakers = json_num_speakers

        files: dict = {"json": json_path}

        if settings.speaker_names:
            transcribed_segments = _apply_speaker_names(transcribed_segments, settings.speaker_names)
            relabelled = export.write_relabelled(
                source_json_path=json_path,
                segments=transcribed_segments,
                original_metadata=json_meta,
                speaker_names=settings.speaker_names,
                output_dir=output_dir,
            )
            files["json"] = relabelled

        settings.validate_for_report()
        _push_progress(job_id, 5, "Report", "Generating Gemini analysis report…")
        speaker_counts = Counter(s["speaker"] for s in transcribed_segments)
        audio_duration = max((s["end"] for s in transcribed_segments), default=None)

        report_path = report.run(
            segments=transcribed_segments,
            source_file=json_source_file,
            output_dir=output_dir,
            context=settings.context,
            num_speakers=len(speaker_counts),
            audio_duration=audio_duration,
            speaker_counts=dict(speaker_counts),
            api_key=settings.gemini_api_key,
            prompts_dir="prompts",
        )
        _push_progress(job_id, 5, "Report", "Done")

        files["report"] = report_path
        job["status"] = "complete"
        job["files"] = files
        _push_complete(job_id)

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        _push_error(job_id, str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
    num_speakers: Optional[int] = Form(None),
    transcription_mode: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    speaker_names: Optional[str] = Form(None),
    word_timestamps: Optional[bool] = Form(None),
    report: bool = Form(False),
    skip_preprocess: bool = Form(False),
    whisper_model: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    job_dir = f"output/jobs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    input_path = os.path.join(job_dir, f"input{ext}")
    with open(input_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    parsed_names = (
        [n.strip() for n in speaker_names.split(",") if n.strip()]
        if speaker_names else []
    )

    params = {
        "context": context,
        "num_speakers": num_speakers,
        "transcription_mode": transcription_mode,
        "language": language,
        "speaker_names": parsed_names,
        "word_timestamps": word_timestamps,
        "report": report,
        "skip_preprocess": skip_preprocess,
        "whisper_model": whisper_model,
    }

    jobs[job_id] = {"status": "queued", "params": params}
    executor.submit(_run_pipeline, job_id, input_path, params)

    return {"job_id": job_id, "ws_url": f"/ws/{job_id}"}


@app.post("/report-from-json")
async def report_from_json(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
    speaker_names: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    job_dir = f"output/jobs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    json_path = os.path.join(job_dir, "transcript.json")
    with open(json_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    parsed_names = (
        [n.strip() for n in speaker_names.split(",") if n.strip()]
        if speaker_names else []
    )

    params = {
        "context": context,
        "speaker_names": parsed_names,
    }

    jobs[job_id] = {"status": "queued", "params": params}
    executor.submit(_run_report_from_json, job_id, json_path, params)

    return {"job_id": job_id, "ws_url": f"/ws/{job_id}"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}/{file_type}")
async def download(job_id: str, file_type: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Job not complete")
    files = job.get("files", {})
    path = files.get(file_type)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not available for this job")
    return FileResponse(path, filename=os.path.basename(path))


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    connections[job_id] = websocket
    try:
        # If the job already finished before the WS client connected, send final state now
        job = jobs.get(job_id)
        if job:
            if job["status"] == "complete":
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "job_id": job_id,
                    "files": job.get("files", {}),
                }))
            elif job["status"] == "error":
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": job.get("error", "Unknown error"),
                }))

        # Hold the connection open until the client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connections.pop(job_id, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
