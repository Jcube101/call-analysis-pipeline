"""
main.py — entry point for the call analysis pipeline.

Usage:
    python main.py --input input/call.mp3
    python main.py --input input/call.mp3 --context work --num-speakers 3
"""

import argparse
import os
import shutil
import sys

from config import settings
from stages import preprocess, diarize, transcribe, export


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end call analysis pipeline: noise reduction → diarization → transcript"
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to the input audio file (MP3, WAV, M4A, …)",
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
    return parser.parse_args()


def main() -> None:
    _check_ffmpeg()
    args = _parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"[error] Input file not found: {args.input}")
        sys.exit(1)

    # Apply CLI overrides on top of .env settings
    settings.override(context=args.context, num_speakers=args.num_speakers)
    if args.whisper_model:
        settings.whisper_model = args.whisper_model

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Call Analysis Pipeline")
    print(f"  Input:    {args.input}")
    print(f"  Context:  {settings.context}")
    print(f"  Speakers: {settings.num_speakers or 'auto-detect'}")
    print(f"  Whisper:  {settings.whisper_model}")
    print("=" * 60)

    # --- Stage 1: Pre-processing ---
    clean_wav = preprocess.run(
        input_path=args.input,
        output_dir=output_dir,
    )

    # --- Stage 2: Diarization ---
    segments = diarize.run(
        clean_wav_path=clean_wav,
        output_dir=output_dir,
        num_speakers=settings.num_speakers,
    )

    # --- Stage 3: Transcription ---
    transcribed_segments = transcribe.run(
        clean_wav_path=clean_wav,
        segments=segments,
        model_size=settings.whisper_model,
    )

    # --- Stage 4: Export ---
    txt_path, json_path = export.run(
        segments=transcribed_segments,
        source_file=args.input,
        output_dir=output_dir,
        context=settings.context,
        num_speakers=settings.num_speakers,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Clean audio:  {clean_wav}")
    print(f"  Transcript:   {txt_path}")
    print(f"  JSON:         {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
