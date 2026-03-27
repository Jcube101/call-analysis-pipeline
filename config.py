"""
config.py — loads .env and exposes typed pipeline settings.

All secrets and tuneable parameters live in .env (never hardcoded here).
Import `settings` anywhere in the pipeline to access them.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

VALID_CONTEXTS = {"friend", "work", "interview", "date"}
VALID_TRANSCRIPTION_MODES = {"fast", "accurate"}


@dataclass
class Settings:
    # Secrets
    huggingface_token: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Conversation metadata
    context: str = field(default_factory=lambda: os.getenv("CONVERSATION_CONTEXT", "friend"))

    # Speaker count — None means let pyannote auto-detect
    num_speakers: Optional[int] = field(default=None)

    # Whisper model size — "tiny", "base", "small", "medium", "large"
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "medium"))

    # Transcription mode — "fast" (one pass, speaker alignment) or "accurate" (per-segment)
    transcription_mode: str = field(
        default_factory=lambda: os.getenv("TRANSCRIPTION_MODE", "accurate")
    )

    # Whisper language — BCP-47 code passed to faster-whisper (e.g. "en", "fr", "es")
    whisper_language: str = field(
        default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "en")
    )

    def __post_init__(self) -> None:
        # Parse NUM_SPEAKERS from env if not overridden programmatically
        if self.num_speakers is None:
            raw = os.getenv("NUM_SPEAKERS", "").strip()
            if raw.isdigit() and int(raw) > 0:
                self.num_speakers = int(raw)
            # else: leave as None for auto-detection

        # Normalise and validate context
        self.context = self.context.strip().lower()
        if self.context not in VALID_CONTEXTS:
            print(
                f"[config] Warning: CONVERSATION_CONTEXT '{self.context}' is not one of "
                f"{sorted(VALID_CONTEXTS)}. Defaulting to 'friend'."
            )
            self.context = "friend"

        # Normalise and validate transcription_mode
        self.transcription_mode = self.transcription_mode.strip().lower()
        if self.transcription_mode not in VALID_TRANSCRIPTION_MODES:
            print(
                f"[config] Warning: TRANSCRIPTION_MODE '{self.transcription_mode}' is not one of "
                f"{sorted(VALID_TRANSCRIPTION_MODES)}. Defaulting to 'fast'."
            )
            self.transcription_mode = "fast"

    def override(
        self,
        context: Optional[str] = None,
        num_speakers: Optional[int] = None,
        transcription_mode: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Apply CLI overrides on top of .env values."""
        if context is not None:
            self.context = context.strip().lower()
            if self.context not in VALID_CONTEXTS:
                raise ValueError(
                    f"--context must be one of {sorted(VALID_CONTEXTS)}, got '{context}'"
                )
        if num_speakers is not None:
            if num_speakers < 1:
                raise ValueError("--num-speakers must be a positive integer")
            self.num_speakers = num_speakers
        if transcription_mode is not None:
            self.transcription_mode = transcription_mode.strip().lower()
            if self.transcription_mode not in VALID_TRANSCRIPTION_MODES:
                raise ValueError(
                    f"--transcription-mode must be one of "
                    f"{sorted(VALID_TRANSCRIPTION_MODES)}, got '{transcription_mode}'"
                )
        if language is not None:
            self.whisper_language = language.strip().lower()

    def validate_for_diarization(self) -> None:
        """Raise if the HuggingFace token is missing (required for pyannote)."""
        if not self.huggingface_token:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN is not set in .env.\n"
                "1. Get a token at https://huggingface.co/settings/tokens\n"
                "2. Accept the model license at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Add HUGGINGFACE_TOKEN=<your-token> to your .env file"
            )

    def validate_for_report(self) -> None:
        """Raise if the Anthropic API key is missing (required for Stage 5 report)."""
        if not self.anthropic_api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set in .env.\n"
                "1. Get a key at https://console.anthropic.com/settings/keys\n"
                "2. Add ANTHROPIC_API_KEY=<your-key> to your .env file"
            )


# Singleton — import this object throughout the pipeline
settings = Settings()
