"""
main.py — entry point for the call analysis pipeline.

Usage:
    python main.py --input input/call.mp3
    python main.py --input input/call.mp3 --context work --num-speakers 3
    python main.py --input input/call.mp3 --transcription-mode accurate
    python main.py --input input/call.mp3 --language fr
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
    parser.add_argument(
        "--transcription-mode",
        metavar="MODE",
        default=None,
        dest="transcription_mode",
        help="Transcription strategy: fast | accurate (default: fast)",
    )
    parser.add_argument(
        "--language",
        metavar="LANG",
        default=None,
        dest="language",
        help="Whisper language code, e.g. en, fr, es (default: en)",
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
    settings.override(
        context=args.context,
        num_speakers=args.num_speakers,
        transcription_mode=args.transcription_mode,
        language=args.language,
    )
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
    print(f"  Tx Mode:  {settings.transcription_mode}")
    print(f"  Language: {settings.whisper_language}")
    _print_device_info()
    print("=" * 60)

    # --- Stage 1: Pre-processing ---
    try:
        clean_wav = preprocess.run(
            input_path=args.input,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"\n[error] Stage 1 (pre-processing) failed: {e}")
        sys.exit(1)

    # --- Stage 2: Diarization ---
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

    # --- Stage 3: Transcription ---
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

    # --- Stage 4: Export ---
    try:
        txt_path, json_path = export.run(
            segments=transcribed_segments,
            source_file=args.input,
            output_dir=output_dir,
            context=settings.context,
            num_speakers=settings.num_speakers,
        )
    except Exception as e:
        print(f"\n[error] Stage 4 (export) failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Clean audio:  {clean_wav}")
    print(f"  Transcript:   {txt_path}")
    print(f"  JSON:         {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
