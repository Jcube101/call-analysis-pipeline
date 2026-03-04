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

    def override(self, context: Optional[str] = None, num_speakers: Optional[int] = None) -> None:
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


# Singleton — import this object throughout the pipeline
settings = Settings()
