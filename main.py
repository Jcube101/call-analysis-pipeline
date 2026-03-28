"""
main.py — entry point for the call analysis pipeline.

Usage:
    python main.py --input input/call.mp3
    python main.py --input input/call.mp3 --context work --num-speakers 3
    python main.py --input input/call.mp3 --transcription-mode accurate
    python main.py --input input/call.mp3 --language fr
    python main.py --input input/call.mp3 --dry-run
    python main.py --input input/call_clean.wav --skip-preprocess
    python main.py --input input/call.mp3 --report
    python main.py --from-json output/call_20260328_151811.json
    python main.py --from-json output/call_20260328_151811.json --context work
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter

from config import settings
from stages import preprocess, diarize, transcribe, export, report


def _check_ffmpeg() -> None:
    """Abort early with a helpful message if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "[error] ffmpeg is not installed or not on your PATH.\n"
            "  macOS:          brew install ffmpeg\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  Windows:        https://ffmpeg.org/download.html (add to PATH)\n"
        )
        sys.exit(1)


def _print_device_info() -> None:
    """Print CUDA availability and which device each stage will use."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  CUDA:         available ({gpu_name})")
            print(f"  Diarization:  GPU (pyannote → CUDA)")
            print(f"  Transcription: GPU (faster-whisper int8_float16)")
        else:
            print(f"  CUDA:         not available")
            print(f"  Diarization:  CPU")
            print(f"  Transcription: CPU (faster-whisper int8)")
    except ImportError:
        print(f"  CUDA:         torch not found")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end call analysis pipeline: noise reduction → diarization → transcript"
    )
    parser.add_argument(
        "--input",
        required=False,
        metavar="FILE",
        help="Path to the input audio file (MP3, WAV, M4A, …)",
    )
    parser.add_argument(
        "--from-json",
        metavar="FILE",
        default=None,
        dest="from_json",
        help="Skip Stages 1–4 and generate a report from an existing transcript JSON (implies --report)",
    )
    parser.add_argument(
        "--context",
        metavar="CTX",
        default=None,
        help="Conversation context override: friend | work | interview | date",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        metavar="N",
        default=None,
        dest="num_speakers",
        help="Number of speakers (overrides NUM_SPEAKERS in .env; omit for auto-detect)",
    )
    parser.add_argument(
        "--whisper-model",
        metavar="SIZE",
        default=None,
        dest="whisper_model",
        help="Whisper model size: tiny | base | small | medium | large (default: medium)",
    )
    parser.add_argument(
        "--transcription-mode",
        metavar="MODE",
        default=None,
        dest="transcription_mode",
        help="Transcription strategy: fast | accurate (default: accurate)",
    )
    parser.add_argument(
        "--language",
        metavar="LANG",
        default=None,
        dest="language",
        help="Whisper language code, e.g. en, fr, es (default: en)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Validate config and input file without running any pipeline stages",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        default=False,
        dest="skip_preprocess",
        help="Skip Stage 1 — input must already be a clean 16 kHz mono WAV",
    )
    parser.add_argument(
        "--speaker-names",
        metavar="NAMES",
        default=None,
        dest="speaker_names",
        help="Comma-separated real names to replace generic labels, e.g. \"Alice,Bob\"",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        dest="report",
        help="Run Stage 5: generate an analysis report via the Gemini API (requires GEMINI_API_KEY)",
    )
    return parser.parse_args()


def _apply_speaker_names(segments: list[dict], names: list[str]) -> list[dict]:
    """
    Replace generic Speaker A/B/... labels with the provided real names.

    Names are applied in alphabetical label order: the first name maps to
    Speaker A, the second to Speaker B, and so on.  If fewer names are given
    than there are speakers, the remaining speakers keep their generic labels.
    """
    unique_speakers = sorted(set(s["speaker"] for s in segments))
    name_map = {spk: names[i] for i, spk in enumerate(unique_speakers) if i < len(names)}
    if not name_map:
        return segments
    for seg in segments:
        if seg["speaker"] in name_map:
            seg["speaker"] = name_map[seg["speaker"]]
    return segments


def _load_json_transcript(json_path: str) -> tuple[list[dict], dict]:
    """Load segments and metadata from an existing transcript JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    segments = data.get("transcript", [])
    if not segments:
        print(f"[error] No transcript segments found in {json_path}")
        sys.exit(1)
    return segments, metadata


def main() -> None:
    _check_ffmpeg()
    args = _parse_args()

    # Validate: need either --input or --from-json
    if not args.input and not args.from_json:
        print("[error] --input or --from-json is required")
        sys.exit(1)

    if args.from_json and args.input:
        print("[error] --input and --from-json are mutually exclusive")
        sys.exit(1)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # --from-json mode: load existing transcript, run Stage 5 only        #
    # ------------------------------------------------------------------ #
    if args.from_json:
        if not os.path.isfile(args.from_json):
            print(f"[error] JSON file not found: {args.from_json}")
            sys.exit(1)

        transcribed_segments, json_meta = _load_json_transcript(args.from_json)

        # Apply context/num-speakers from JSON metadata, then CLI overrides
        json_context = json_meta.get("context", "friend")
        json_num_speakers = json_meta.get("num_speakers")
        json_source_file = json_meta.get("source_file", os.path.basename(args.from_json))

        parsed_names = (
            [n.strip() for n in args.speaker_names.split(",") if n.strip()]
            if args.speaker_names else None
        )
        settings.override(
            context=args.context or json_context,
            num_speakers=args.num_speakers or json_num_speakers,
            speaker_names=parsed_names,
        )

        relabelled_json_path = None
        if settings.speaker_names:
            transcribed_segments = _apply_speaker_names(transcribed_segments, settings.speaker_names)
            relabelled_json_path = export.write_relabelled(
                source_json_path=args.from_json,
                segments=transcribed_segments,
                original_metadata=json_meta,
                speaker_names=settings.speaker_names,
                output_dir=output_dir,
            )

        try:
            settings.validate_for_report()
        except EnvironmentError as e:
            print(f"\n[error] {e}")
            sys.exit(1)

        speaker_counts = Counter(s["speaker"] for s in transcribed_segments)
        audio_duration = max((s["end"] for s in transcribed_segments), default=None)

        print("=" * 60)
        print("  Call Analysis Pipeline — Report only (--from-json)")
        print(f"  JSON:     {args.from_json}")
        print(f"  Source:   {json_source_file}")
        print(f"  Context:  {settings.context}")
        print(f"  Speakers: {len(speaker_counts)}")
        if settings.speaker_names:
            print(f"  Names:    {', '.join(settings.speaker_names)}")
        print(f"  Segments: {len(transcribed_segments)}")
        if audio_duration:
            print(f"  Duration: {_fmt_duration(audio_duration)}")
        print("=" * 60)

        total_start = time.time()
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
        except Exception as e:
            print(f"\n[error] Stage 5 (report) failed: {e}")
            sys.exit(1)

        elapsed = time.time() - total_start

        print("\n--- Report preview ---")
        with open(report_path, encoding="utf-8") as f:
            preview_lines = f.readlines()
        print("".join(preview_lines[:20]).rstrip())
        if len(preview_lines) > 20:
            print(f"  ... ({len(preview_lines) - 20} more lines in {report_path})")
        print("----------------------")

        print("\n" + "=" * 60)
        print("  Report complete!")
        if relabelled_json_path:
            print(f"  JSON:     {relabelled_json_path}")
        print(f"  Report:   {report_path}")
        print(f"  Elapsed:  {_fmt_duration(elapsed)}")
        print("=" * 60)
        return

    # ------------------------------------------------------------------ #
    # Normal mode: full pipeline (Stages 1–4, optionally Stage 5)         #
    # ------------------------------------------------------------------ #

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"[error] Input file not found: {args.input}")
        sys.exit(1)

    if args.skip_preprocess and not args.input.lower().endswith(".wav"):
        print("[error] --skip-preprocess requires a WAV file as --input")
        sys.exit(1)

    # Parse --speaker-names "Alice,Bob" → ["Alice", "Bob"]
    parsed_names = (
        [n.strip() for n in args.speaker_names.split(",") if n.strip()]
        if args.speaker_names else None
    )

    # Apply CLI overrides on top of .env settings
    settings.override(
        context=args.context,
        num_speakers=args.num_speakers,
        transcription_mode=args.transcription_mode,
        language=args.language,
        speaker_names=parsed_names,
    )
    if args.whisper_model:
        settings.whisper_model = args.whisper_model

    print("=" * 60)
    print("  Call Analysis Pipeline")
    print(f"  Input:    {args.input}")
    print(f"  Context:  {settings.context}")
    print(f"  Speakers: {settings.num_speakers or 'auto-detect'}")
    if settings.speaker_names:
        print(f"  Names:    {', '.join(settings.speaker_names)}")
    print(f"  Whisper:  {settings.whisper_model}")
    print(f"  Tx Mode:  {settings.transcription_mode}")
    print(f"  Language: {settings.whisper_language}")
    if args.skip_preprocess:
        print(f"  Stage 1:  SKIPPED (using input as clean WAV)")
    if args.report:
        print(f"  Report:   ON (Stage 5 will run after transcription)")
    if args.dry_run:
        print(f"  Mode:     DRY RUN")
    _print_device_info()
    print("=" * 60)

    if args.report:
        try:
            settings.validate_for_report()
        except EnvironmentError as e:
            print(f"\n[error] {e}")
            sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] Config and input file are valid. No stages executed.")
        sys.exit(0)

    stage_times: dict[str, float] = {}
    total_start = time.time()

    # --- Stage 1: Pre-processing ---
    if args.skip_preprocess:
        clean_wav = args.input
        stage_times["Stage 1"] = 0.0
    else:
        t = time.time()
        try:
            clean_wav = preprocess.run(
                input_path=args.input,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"\n[error] Stage 1 (pre-processing) failed: {e}")
            sys.exit(1)
        stage_times["Stage 1"] = time.time() - t

    # --- Stage 2: Diarization ---
    t = time.time()
    try:
        segments = diarize.run(
            clean_wav_path=clean_wav,
            output_dir=output_dir,
            num_speakers=settings.num_speakers,
        )
    except EnvironmentError as e:
        print(f"\n[error] Stage 2 (diarization) configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[error] Stage 2 (diarization) failed: {e}")
        sys.exit(1)
    stage_times["Stage 2"] = time.time() - t

    # Apply speaker name mapping if provided
    if settings.speaker_names:
        segments = _apply_speaker_names(segments, settings.speaker_names)

    # --- Stage 3: Transcription ---
    t = time.time()
    try:
        transcribed_segments = transcribe.run(
            clean_wav_path=clean_wav,
            segments=segments,
            model_size=settings.whisper_model,
            mode=settings.transcription_mode,
        )
    except Exception as e:
        print(f"\n[error] Stage 3 (transcription) failed: {e}")
        sys.exit(1)
    stage_times["Stage 3"] = time.time() - t

    # --- Stage 4: Export ---
    t = time.time()
    try:
        txt_path, json_path = export.run(
            segments=transcribed_segments,
            source_file=args.input,
            output_dir=output_dir,
            context=settings.context,
            num_speakers=settings.num_speakers,
            speaker_names=settings.speaker_names or None,
        )
    except Exception as e:
        print(f"\n[error] Stage 4 (export) failed: {e}")
        sys.exit(1)
    stage_times["Stage 4"] = time.time() - t

    # Compute speaker stats — needed by both summary and Stage 5
    speaker_counts = Counter(s["speaker"] for s in transcribed_segments)

    # Read duration from WAV header only — avoids loading the full audio into RAM
    try:
        import soundfile as _sf
        _info = _sf.info(clean_wav)
        audio_duration = _info.frames / _info.samplerate
    except Exception:
        audio_duration = None

    # --- Stage 5: Report (optional) ---
    report_path = None
    if args.report:
        t = time.time()
        try:
            report_path = report.run(
                segments=transcribed_segments,
                source_file=args.input,
                output_dir=output_dir,
                context=settings.context,
                num_speakers=len(speaker_counts),
                audio_duration=audio_duration,
                speaker_counts=dict(speaker_counts),
                api_key=settings.gemini_api_key,
                prompts_dir="prompts",
            )
        except Exception as e:
            print(f"\n[error] Stage 5 (report) failed: {e}")
            sys.exit(1)
        stage_times["Stage 5"] = time.time() - t

        # Print first ~20 lines of the report body as a terminal summary
        print("\n--- Report preview ---")
        with open(report_path, encoding="utf-8") as f:
            preview_lines = f.readlines()
        print("".join(preview_lines[:20]).rstrip())
        if len(preview_lines) > 20:
            print(f"  ... ({len(preview_lines) - 20} more lines in {report_path})")
        print("----------------------")

    # --- Summary ---
    total_elapsed = time.time() - total_start

    speaker_breakdown = "  ".join(
        f"{spk}: {cnt}" for spk, cnt in sorted(speaker_counts.items())
    )
    timing_breakdown = "  ".join(
        f"{name}: {t:.1f}s" for name, t in stage_times.items() if t > 0
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Clean audio:  {clean_wav}")
    print(f"  Transcript:   {txt_path}")
    print(f"  JSON:         {json_path}")
    if report_path:
        print(f"  Report:       {report_path}")
    print(f"  Segments:     {len(transcribed_segments)}  |  Speakers: {len(speaker_counts)}  ({speaker_breakdown})")
    if audio_duration is not None:
        print(f"  Audio:        {_fmt_duration(audio_duration)}  |  Elapsed: {_fmt_duration(total_elapsed)}  ({timing_breakdown})")
    else:
        print(f"  Elapsed:      {_fmt_duration(total_elapsed)}  ({timing_breakdown})")
    print("=" * 60)


if __name__ == "__main__":
    main()
