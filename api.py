"""
api.py — FastAPI HTTP + WebSocket wrapper for the call analysis pipeline.

Endpoints
---------
  POST /analyse                   Upload audio file; runs Stages 1–4 (and 5 if report=true)
  POST /report-from-json          Upload existing transcript JSON; runs Stage 5 only
  GET  /health                    {"status": "ok"}
  GET  /status/{job_id}           Full job dict
  GET  /reconnect/{job_id}        Full job state for reconnect after a dropped connection
  GET  /download/{job_id}/{type}  Stream transcript / json / report file
  WS   /ws/{job_id}               Push progress / complete / error events

Job lifecycle
-------------
  queued → running → complete | error

WebSocket message shapes
------------------------
  {"type": "progress", "stage": 1, "stage_name": "Preprocess", "message": "..."}
  {"type": "complete", "transcript": [...], "report": "...|null", "metadata": {...}}
  {"type": "error", "message": "..."}

All messages are appended to job["message_queue"] so reconnecting clients can
replay the full history via /reconnect or by connecting to /ws/{job_id}.

GPU constraint
--------------
  ThreadPoolExecutor(max_workers=1) serialises all pipeline jobs because only
  one pipeline can use the GPU at a time.
"""

from __future__ import annotations

import asyncio
import gc
import glob as _glob
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from config import VALID_CONTEXTS, Settings, settings

# Snapshot of .env defaults taken once at startup — reused on every job reset
# to avoid re-parsing environment variables per request.
_default_settings: dict = {}

from stages import diarize, export, preprocess, report, transcribe


# ---------------------------------------------------------------------------
# Thread exception hook — keeps uvicorn alive if a pipeline thread crashes
# ---------------------------------------------------------------------------

def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
    print(f"[error] Unhandled thread exception: {args.exc_value}")
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_tb)

threading.excepthook = _handle_thread_exception


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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class CORSFallbackMiddleware(BaseHTTPMiddleware):
    """Explicitly stamps CORS headers on every response as a belt-and-suspenders
    fallback for proxies (e.g. ngrok) that may strip or rewrite headers."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response


app.add_middleware(CORSFallbackMiddleware)


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


def get_or_recover_job(job_id: str) -> Optional[dict]:
    """
    Return the in-memory job dict, or reconstruct it from disk if the server restarted.

    On recovery the entry is re-inserted into the global jobs dict so subsequent
    lookups are fast.
    """
    if job_id in jobs:
        return jobs[job_id]

    job_dir = f"output/jobs/{job_id}"
    if not os.path.isdir(job_dir):
        return None

    txt_files = _glob.glob(os.path.join(job_dir, "*.txt"))
    json_files = [
        f for f in _glob.glob(os.path.join(job_dir, "*.json"))
        if os.path.basename(f) not in ("transcript.json",)
        and not os.path.basename(f).startswith("input")
    ]
    report_files = _glob.glob(os.path.join(job_dir, "*_report.md"))

    files: dict = {}
    if txt_files:
        files["transcript"] = txt_files[0]
    if json_files:
        files["json"] = json_files[0]
    if report_files:
        files["report"] = report_files[0]

    recovered: dict = {
        "status": "complete",
        "output_dir": job_dir,
        "files": files,
        "message_queue": [],
        "current_stage": None,
        "stage_name": None,
        "progress_message": None,
        "transcript": None,
        "metadata": {},
        "report_path": report_files[0] if report_files else None,
        "report": None,
        "error": None,
    }

    # Hydrate transcript and metadata from the pipeline JSON output
    json_path = files.get("json")
    if json_path and os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            recovered["metadata"] = data.get("metadata", {})
            recovered["transcript"] = data.get("transcript", [])
        except Exception:
            pass

    # Hydrate report text
    if recovered["report_path"] and os.path.isfile(recovered["report_path"]):
        try:
            with open(recovered["report_path"], encoding="utf-8") as f:
                recovered["report"] = f.read()
        except Exception:
            pass

    jobs[job_id] = recovered
    return recovered


def _push_ws(job_id: str, payload: dict) -> None:
    """
    Append a message to the job's queue and attempt delivery via WebSocket (thread-safe).

    The message is stored regardless of WS availability so reconnecting clients
    can replay the full history via /reconnect/{job_id} or /ws/{job_id}.
    """
    job = jobs.get(job_id)
    if job is not None:
        job["message_queue"].append(payload)

    ws = connections.get(job_id)
    if ws is None or _loop is None:
        return

    future = asyncio.run_coroutine_threadsafe(ws.send_json(payload), _loop)
    try:
        future.result(timeout=5)
    except Exception:
        pass  # message is safe in queue; client can replay via /reconnect


def _push_progress(job_id: str, stage: int, stage_name: str, message: str) -> None:
    job = jobs.get(job_id)
    if job is not None:
        job["current_stage"] = stage
        job["stage_name"] = stage_name
        job["progress_message"] = message
    _push_ws(job_id, {"type": "progress", "stage": stage, "stage_name": stage_name, "message": message})


def _push_complete(job_id: str) -> None:
    job = jobs[job_id]
    _push_ws(job_id, {
        "type": "complete",
        "transcript": job.get("transcript") or [],
        "report": job.get("report"),
        "metadata": job.get("metadata", {}),
    })


def _push_error(job_id: str, message: str) -> None:
    _push_ws(job_id, {"type": "error", "message": message})


# ---------------------------------------------------------------------------
# Pipeline threads
# ---------------------------------------------------------------------------

def _heartbeat_worker(job_id: str, stop: threading.Event, interval: float = 10.0) -> None:
    """Send periodic progress pings during Stage 5 to keep the WebSocket alive.

    Stage 5 (Gemini API) can take 30–60 s. Without pings the WebSocket proxy
    (ngrok, nginx) times out the idle connection before the report arrives.
    """
    while not stop.wait(interval):
        _push_progress(job_id, 5, "AI Report", "Stage 5: Generating analysis report...")

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
        # Force VRAM to clear fully before Stage 3 loads Whisper.
        # diarize.run() does this internally, but a synchronize + double flush
        # ensures all CUDA ops have actually completed before the cache is freed.
        try:
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(2)
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
            job_id=job_id,
        )
        _push_progress(job_id, 4, "Export", "Done")

        # Store transcript and metadata in job for WS complete message and /reconnect
        job["transcript"] = transcribed_segments
        job["metadata"] = {
            "job_id": job_id,
            "source_file": os.path.basename(input_path),
            "context": settings.context,
            "num_speakers": settings.num_speakers,
        }

        files: dict = {"transcript": txt_path, "json": json_path, "clean_wav": clean_wav}

        # Stage 5 — Report (optional)
        # _push_complete is called ONLY after this block so the "complete"
        # message always carries the finished report content.
        if params.get("report"):
            settings.validate_for_report()
            _push_progress(job_id, 5, "AI Report", "Stage 5: Generating analysis report...")
            speaker_counts = Counter(s["speaker"] for s in transcribed_segments)
            try:
                import soundfile as _sf
                _info = _sf.info(clean_wav)
                audio_duration: Optional[float] = _info.frames / _info.samplerate
            except Exception:
                audio_duration = None

            # Heartbeat keeps the WebSocket alive during the Gemini API call
            _stop_hb = threading.Event()
            _hb = threading.Thread(
                target=_heartbeat_worker, args=(job_id, _stop_hb), daemon=True
            )
            _hb.start()
            try:
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
            finally:
                _stop_hb.set()
                _hb.join(timeout=1)

            job["report_path"] = report_path
            try:
                with open(report_path, encoding="utf-8") as f:
                    job["report"] = f.read()
            except Exception:
                pass
            files["report"] = report_path
            _push_progress(job_id, 5, "AI Report", "Done")

        job["status"] = "complete"
        job["files"] = files
        _push_complete(job_id)

    except BaseException as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        print(f"[error] Pipeline crashed: {traceback.format_exc()}")
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
        _push_progress(job_id, 5, "AI Report", "Stage 5: Generating analysis report...")
        speaker_counts = Counter(s["speaker"] for s in transcribed_segments)
        audio_duration = max((s["end"] for s in transcribed_segments), default=None)

        # Heartbeat keeps the WebSocket alive during the Gemini API call
        _stop_hb = threading.Event()
        _hb = threading.Thread(
            target=_heartbeat_worker, args=(job_id, _stop_hb), daemon=True
        )
        _hb.start()
        try:
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
        finally:
            _stop_hb.set()
            _hb.join(timeout=1)

        _push_progress(job_id, 5, "AI Report", "Done")

        job["report_path"] = report_path
        try:
            with open(report_path, encoding="utf-8") as f:
                job["report"] = f.read()
        except Exception:
            pass

        job["transcript"] = transcribed_segments
        job["metadata"] = json_meta
        files["report"] = report_path
        job["status"] = "complete"
        job["files"] = files
        _push_complete(job_id)

    except BaseException as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        print(f"[error] Report-from-JSON crashed: {traceback.format_exc()}")
        _push_error(job_id, str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


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

    jobs[job_id] = {
        "status": "queued",
        "params": params,
        "message_queue": [],
        "current_stage": None,
        "stage_name": None,
        "progress_message": None,
        "transcript": None,
        "metadata": {},
        "report_path": None,
        "report": None,
        "error": None,
        "output_dir": job_dir,
    }
    executor.submit(_run_pipeline, job_id, input_path, params)

    return {"job_id": job_id, "ws_url": f"/ws/{job_id}"}


@app.post("/report-from-json")
async def report_from_json(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None),
    speaker_names: Optional[str] = Form(None),
):
    # Read bytes first so we can inspect the metadata before choosing a folder
    content = await file.read()

    # If the JSON carries a job_id from a previous pipeline run, write the
    # report into that job's existing folder so all files stay together.
    existing_job_id: Optional[str] = None
    try:
        existing_job_id = json.loads(content).get("metadata", {}).get("job_id")
    except Exception:
        pass

    if existing_job_id and os.path.isdir(f"output/jobs/{existing_job_id}"):
        job_id = existing_job_id
        job_dir = f"output/jobs/{existing_job_id}"
        prior = get_or_recover_job(existing_job_id) or {}
    else:
        job_id = str(uuid.uuid4())
        job_dir = f"output/jobs/{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        prior = {}

    json_path = os.path.join(job_dir, "transcript.json")
    with open(json_path, "wb") as fh:
        fh.write(content)

    parsed_names = (
        [n.strip() for n in speaker_names.split(",") if n.strip()]
        if speaker_names else []
    )

    params = {
        "context": context,
        "speaker_names": parsed_names,
    }

    # Preserve files and transcript from the prior pipeline run if available
    jobs[job_id] = {
        "status": "queued",
        "params": params,
        "message_queue": prior.get("message_queue", []),
        "current_stage": None,
        "stage_name": None,
        "progress_message": None,
        "transcript": prior.get("transcript"),
        "metadata": prior.get("metadata", {}),
        "report": None,
        "error": None,
        "output_dir": job_dir,
        "files": prior.get("files", {}),
    }
    executor.submit(_run_report_from_json, job_id, json_path, params)

    return {"job_id": job_id, "ws_url": f"/ws/{job_id}"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = get_or_recover_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/reconnect/{job_id}")
async def reconnect(job_id: str):
    """Return full job state for a client reconnecting after a dropped connection."""
    job = get_or_recover_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # If report content isn't cached in memory, try to read it from disk.
    # This covers the case where the server restarted after Stage 5 completed,
    # or where the file read failed silently during the pipeline run.
    report = job.get("report")
    if not report:
        report_path = job.get("report_path")
        if not report_path:
            job_dir = job.get("output_dir", f"output/jobs/{job_id}")
            candidates = _glob.glob(os.path.join(job_dir, "*_report.md"))
            if candidates:
                report_path = candidates[0]
                job["report_path"] = report_path
        if report_path and os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as f:
                    report = f.read()
                    job["report"] = report  # cache for subsequent calls
            except Exception:
                pass

    return {
        "status": job.get("status"),
        "current_stage": job.get("current_stage"),
        "stage_name": job.get("stage_name"),
        "progress_message": job.get("progress_message"),
        "message_queue": job.get("message_queue", []),
        "transcript": job.get("transcript"),
        "report": report,
        "metadata": job.get("metadata", {}),
        "error": job.get("error"),
    }


@app.get("/download/{job_id}/{file_type}")
async def download(job_id: str, file_type: str):
    job = get_or_recover_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Job not complete")

    job_dir = job.get("output_dir", f"output/jobs/{job_id}")
    files = job.get("files", {})

    # The files dict may use different key names than the URL type
    # (e.g. "transcript" vs "txt", "clean_wav" vs "wav"), so fall back
    # to a glob scan of the job directory for each type.
    if file_type == "report":
        path = files.get("report") or job.get("report_path")
        if not path or not os.path.isfile(path):
            candidates = _glob.glob(os.path.join(job_dir, "*_report.md"))
            path = candidates[0] if candidates else None
    elif file_type == "txt":
        path = files.get("txt") or files.get("transcript")
        if not path or not os.path.isfile(path):
            candidates = _glob.glob(os.path.join(job_dir, "*.txt"))
            path = candidates[0] if candidates else None
    elif file_type == "json":
        path = files.get("json")
        if not path or not os.path.isfile(path):
            candidates = [
                f for f in _glob.glob(os.path.join(job_dir, "*.json"))
                if os.path.basename(f) not in ("transcript.json",)
                and not os.path.basename(f).startswith("input")
            ]
            path = candidates[0] if candidates else None
    elif file_type == "wav":
        path = files.get("wav") or files.get("clean_wav")
        if not path or not os.path.isfile(path):
            candidates = _glob.glob(os.path.join(job_dir, "*_clean.wav"))
            path = candidates[0] if candidates else None
    else:
        raise HTTPException(status_code=400, detail=f"Unknown file type '{file_type}'. Use: txt, json, report, wav")

    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not available for this job")
    return FileResponse(path, filename=os.path.basename(path))


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    connections[job_id] = websocket
    try:
        # Flush any messages that arrived before the client connected (handles
        # race conditions and reconnects after ngrok/network drops).
        job = get_or_recover_job(job_id)
        if job:
            for msg in list(job.get("message_queue", [])):
                try:
                    await websocket.send_json(msg)
                except Exception:
                    break

        # Hold the connection open; pipeline pushes messages asynchronously.
        while True:
            try:
                await websocket.receive_text()
            except (WebSocketDisconnect, Exception):
                break
    except Exception:
        pass
    finally:
        connections.pop(job_id, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
