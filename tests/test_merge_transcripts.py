"""tools/merge_transcripts.py, run through its command line as a user would."""

import json
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tools", "merge_transcripts.py")

_W1 = "Speaker re-identification failed for 80 of 102 segments; labels may be less accurate."
_W2 = "Only 1 speaker found but 2 were requested; one speaker may have absorbed the other."


def _part(tmp_path, name, end, warnings=None):
    metadata = {"job_id": name, "source_file": f"{name}.m4a", "context": "friend"}
    if warnings is not None:
        metadata["warnings"] = warnings
    data = {
        "metadata": metadata,
        "transcript": [
            {"start": 0.0, "end": end / 2, "speaker": "Speaker A", "text": "one"},
            {"start": end / 2, "end": end, "speaker": "Speaker B", "text": "two"},
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _merge(tmp_path, part1, part2, *extra):
    out = str(tmp_path / "merged.json")
    result = subprocess.run(
        [sys.executable, SCRIPT, part1, part2, out, *extra],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    if "--dry-run" in extra:
        return None, result.stdout
    with open(out, encoding="utf-8") as f:
        return json.load(f), result.stdout


def test_warning_from_part2_kept_with_prefix(tmp_path):
    merged, _ = _merge(tmp_path, _part(tmp_path, "p1", 10.0), _part(tmp_path, "p2", 6.0, [_W1]))
    assert merged["metadata"]["warnings"] == [f"Part 2: {_W1}"]
    # Everything else about the merge is unchanged: part 1's metadata, offsets.
    assert merged["metadata"]["job_id"] == "p1"
    assert [s["start"] for s in merged["transcript"]] == [0.0, 5.0, 11.0, 14.0]


def test_warnings_from_both_parts_in_order(tmp_path):
    merged, out = _merge(tmp_path, _part(tmp_path, "p1", 10.0, [_W2]),
                         _part(tmp_path, "p2", 6.0, [_W1]))
    assert merged["metadata"]["warnings"] == [f"Part 1: {_W2}", f"Part 2: {_W1}"]
    assert f"Warning: Part 2: {_W1}" in out


def test_no_warnings_adds_no_key(tmp_path):
    merged, _ = _merge(tmp_path, _part(tmp_path, "p1", 10.0), _part(tmp_path, "p2", 6.0))
    assert "warnings" not in merged["metadata"]


def test_empty_warnings_in_part1_are_dropped(tmp_path):
    merged, _ = _merge(tmp_path, _part(tmp_path, "p1", 10.0, []), _part(tmp_path, "p2", 6.0))
    assert "warnings" not in merged["metadata"]


@pytest.mark.parametrize("value", [{"not": "a list"}, [3, None, "  "]])
def test_malformed_warnings_ignored(tmp_path, value):
    merged, _ = _merge(tmp_path, _part(tmp_path, "p1", 10.0, value), _part(tmp_path, "p2", 6.0))
    assert "warnings" not in merged["metadata"]


def test_cli_options_unchanged(tmp_path):
    merged, _ = _merge(tmp_path, _part(tmp_path, "p1", 10.0), _part(tmp_path, "p2", 6.0, [_W1]),
                       "--gap", "2", "--swap-speakers", "Speaker A", "Speaker B")
    assert merged["transcript"][2]["start"] == 12.0
    assert [s["speaker"] for s in merged["transcript"][2:]] == ["Speaker B", "Speaker A"]
    _, out = _merge(tmp_path, _part(tmp_path, "p1", 10.0), _part(tmp_path, "p2", 6.0), "--dry-run")
    assert "Part 2: 2 segments" in out
