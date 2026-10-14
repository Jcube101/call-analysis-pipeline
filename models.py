"""
models.py — single source of truth for Stage 5 analysis models.

Every model ID the pipeline accepts, its display label and provider, the
default, the per-provider fallbacks, and the remapping of retired IDs live
here. api.py, config.py, main.py and stages/report.py import from this module
rather than hardcoding model strings.
"""

from __future__ import annotations

# API ID -> {label, provider}. Order is the order the frontend dropdown shows.
MODELS: dict[str, dict[str, str]] = {
    "claude-haiku-5-5":  {"label": "Claude Haiku 5.5",  "provider": "anthropic"},
    "claude-sonnet-5-5": {"label": "Claude Sonnet 5.5", "provider": "anthropic"},
    "gemini-3.8-flash":  {"label": "Gemini 3.8 Flash",  "provider": "google"},
    "gemini-4-argon":    {"label": "Gemini 4 Argon",    "provider": "google"},
}

DEFAULT_MODEL = "claude-haiku-5-5"

# Model retried when a call to another model of the same provider fails
FALLBACK_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-5-5",
    "google": "gemini-3.8-flash",
}

# Retired IDs still sent by saved frontend state (localStorage) or old scripts
LEGACY_ALIASES: dict[str, str] = {
    "claude-haiku-4-5-20251001": "claude-haiku-5-5",
    "claude-sonnet-4-6": "claude-sonnet-5-5",
    "gemini-3-flash-preview": "gemini-3.8-flash",
    "gemini-3.1-pro-preview": "gemini-4-argon",
}

ALLOWED_MODELS: list[str] = list(MODELS)


def provider_for(model: str) -> str:
    """Return "anthropic" or "google" for a model ID (registry first, then prefix)."""
    if model in MODELS:
        return MODELS[model]["provider"]
    return "anthropic" if model.startswith("claude-") else "google"


def resolve_model(model: str | None) -> str:
    """
    Map a requested model to an allowed ID.

    Legacy IDs are remapped to their successor; unknown or empty values fall
    back to DEFAULT_MODEL. Both cases print a line to the terminal.
    """
    if model in MODELS:
        return model
    if model in LEGACY_ALIASES:
        new = LEGACY_ALIASES[model]
        print(f"[models] Legacy model ID '{model}' remapped to '{new}'")
        return new
    if model:
        print(f"[models] Unknown model ID '{model}', using default '{DEFAULT_MODEL}'")
    return DEFAULT_MODEL
