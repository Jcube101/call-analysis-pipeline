"""Tests for models.py — the Stage 5 model registry, legacy remapping and fallback."""

import pytest

from models import (
    ALLOWED_MODELS,
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    LEGACY_ALIASES,
    MODELS,
    provider_for,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_default_model_is_haiku_5_5():
    assert DEFAULT_MODEL == "claude-haiku-5-5"
    assert DEFAULT_MODEL in MODELS


def test_every_model_has_label_and_known_provider():
    for model_id, info in MODELS.items():
        assert info["label"]
        assert info["provider"] in ("anthropic", "google"), model_id


def test_fallbacks_are_allowed_models_of_their_provider():
    for provider, model_id in FALLBACK_MODELS.items():
        assert model_id in ALLOWED_MODELS
        assert MODELS[model_id]["provider"] == provider


def test_legacy_aliases_point_at_allowed_models():
    for old, new in LEGACY_ALIASES.items():
        assert new in ALLOWED_MODELS, f"{old} remaps to unknown {new}"


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", list(MODELS))
def test_allowed_model_passes_through_silently(model_id, capsys):
    assert resolve_model(model_id) == model_id
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("old,new", [
    ("claude-haiku-4-5-20251001", "claude-haiku-5-5"),
    ("claude-sonnet-4-6", "claude-sonnet-5-5"),
    ("gemini-3-flash-preview", "gemini-3.8-flash"),
    ("gemini-3.1-pro-preview", "gemini-4-argon"),
])
def test_legacy_id_remapped_with_log_line(old, new, capsys):
    assert resolve_model(old) == new
    out = capsys.readouterr().out
    assert "Legacy model ID" in out
    assert old in out and new in out


def test_unknown_id_falls_back_to_default_with_log_line(capsys):
    assert resolve_model("gpt-4o") == DEFAULT_MODEL
    out = capsys.readouterr().out
    assert "Unknown model ID 'gpt-4o'" in out


@pytest.mark.parametrize("value", [None, ""])
def test_missing_value_falls_back_to_default(value):
    assert resolve_model(value) == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# provider_for
# ---------------------------------------------------------------------------

def test_provider_for_registry_models():
    assert provider_for("claude-haiku-5-5") == "anthropic"
    assert provider_for("claude-sonnet-5-5") == "anthropic"
    assert provider_for("gemini-3.8-flash") == "google"
    assert provider_for("gemini-4-argon") == "google"


def test_provider_for_unregistered_ids_uses_prefix():
    assert provider_for("claude-haiku-4-5-20251001") == "anthropic"
    assert provider_for("gemini-3-flash-preview") == "google"
