"""
How Stage 2's speaker re-identification went, and the warnings that follow from it.

Kept apart from stages/diarize.py so that api.py, main.py and the tests can read
an outcome without importing pyannote or torch.

An outcome is a plain dict, so it can sit on the job dict and go out as JSON:

    {
      "status": "ok" | "partial" | "failed",
      "reason": None | "feature_extraction" | "sklearn_missing"
                | "too_few_features" | "exception",
      "total_segments": int,       # segments pyannote returned
      "clustered_segments": int,   # segments whose features went into clustering
      "short_segments": int,       # under the minimum length; inherit a neighbour's label by design
      "failed_segments": int,      # feature extraction raised; inherit a neighbour's label
      "error": str | None,         # first feature error, or the exception that stopped re-identification
      "speakers_requested": int | None,  # fixed speaker count, None in auto-detect mode
      "speakers_found": int | None,      # distinct labels Stage 2 returned
    }

Statuses:
  ok       clustering ran and every segment it tried to analyse was analysed.
  partial  clustering ran, but at least PARTIAL_MIN_FAILED_SEGMENTS segments
           failed feature extraction. Those take the label of the nearest
           segment in time rather than one matched by voice.
  failed   re-identification did not change anything; labels are pyannote's.
"""

from __future__ import annotations

from typing import Optional

OK = "ok"
PARTIAL = "partial"
FAILED = "failed"

# One failed segment is enough for "partial". On a healthy run the count is
# always 0 -- every failure seen so far has been memory pressure (MemoryError
# from numpy inside librosa), and each failed segment is labelled by position
# rather than by voice. A percentage cut-off would hide exactly the early sign
# that preceded the 2026-10-14 job, where 80 of 102 segments failed.
PARTIAL_MIN_FAILED_SEGMENTS = 1

_PYANNOTE_LABELS = "speaker labels come straight from pyannote and may be less accurate"


def new_outcome(total_segments: int, speakers_requested: Optional[int]) -> dict:
    """An outcome for a run that has not finished yet. Defaults to failed/exception,
    which is what it is if nothing gets far enough to say otherwise."""
    return {
        "status": FAILED,
        "reason": "exception",
        "total_segments": total_segments,
        "clustered_segments": 0,
        "short_segments": 0,
        "failed_segments": 0,
        "error": None,
        "speakers_requested": speakers_requested,
        "speakers_found": None,
    }


def status_after_clustering(failed_segments: int) -> str:
    """ok or partial, for a run whose clustering completed."""
    return PARTIAL if failed_segments >= PARTIAL_MIN_FAILED_SEGMENTS else OK


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def warnings_for(outcome: Optional[dict]) -> list[str]:
    """Short, human-readable warnings for an outcome. Empty for a clean run."""
    if not outcome:
        return []
    out: list[str] = []
    status = outcome.get("status")
    total = outcome.get("total_segments", 0)
    failed = outcome.get("failed_segments", 0)

    if status == PARTIAL:
        out.append(
            f"Speaker re-identification failed for {failed} of {total} segments; "
            "labels may be less accurate."
        )
    elif status == FAILED:
        reason = outcome.get("reason")
        if reason == "sklearn_missing":
            out.append(
                f"Speaker re-identification skipped: scikit-learn is not installed; {_PYANNOTE_LABELS}."
            )
        elif reason == "too_few_features":
            clustered = outcome.get("clustered_segments", 0)
            wanted = outcome.get("speakers_requested") or "the detected"
            out.append(
                f"Speaker re-identification skipped: only {clustered} of {total} segments "
                f"could be analysed for {wanted} speakers; {_PYANNOTE_LABELS}."
            )
        elif failed:
            out.append(
                f"Speaker re-identification failed for {failed} of {total} segments and then "
                f"stopped; {_PYANNOTE_LABELS}."
            )
        else:
            out.append(f"Speaker re-identification failed; {_PYANNOTE_LABELS}.")

    requested = outcome.get("speakers_requested")
    found = outcome.get("speakers_found")
    if requested and found is not None and found < requested:
        out.append(
            f"Only {_plural(found, 'speaker')} found but {requested} were requested; "
            "one speaker may have absorbed the other."
        )
    return out
