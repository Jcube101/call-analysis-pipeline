"""tools/swap_speakers.py, run through its command line as a user would."""

import copy
import json
import os
import subprocess
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tools", "swap_speakers.py")

_W1 = "Speaker re-identification failed for 80 of 102 segments; labels may be less accurate."


def _data(a="Speaker A", b="Speaker B"):
    return {
        "metadata": {
            "job_id": "abc123", "source_file": "call.m4a", "context": "friend",
            "num_speakers": 2, "speaker_names": [a, b], "warnings": [_W1],
        },
        "transcript": [
            {"start": 0.0, "end": 1.0, "speaker": a, "text": f"I am {a}", "confidence": 0.9},
            {"start": 1.0, "end": 2.0, "speaker": b, "text": "two"},
            {"start": 2.0, "end": 3.0, "speaker": a, "text": "three", "label": a},
        ],
    }


def _write(tmp_path, data):
    path = tmp_path / "in.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True, encoding="utf-8")


def _swap(tmp_path, data, a="Speaker A", b="Speaker B", *extra):
    src, out = _write(tmp_path, data), str(tmp_path / "out.json")
    result = _run(src, out, a, b, *extra)
    assert result.returncode == 0, result.stderr
    with open(out, encoding="utf-8") as f:
        return json.load(f), result.stdout


def test_segments_swapped_both_ways(tmp_path):
    out, stdout = _swap(tmp_path, _data())
    assert [s["speaker"] for s in out["transcript"]] == ["Speaker B", "Speaker A", "Speaker B"]
    assert out["transcript"][2]["label"] == "Speaker B"
    assert "Segments Speaker A -> Speaker B: 2" in stdout
    assert "Segments Speaker B -> Speaker A: 1" in stdout
    assert "Wrote" in stdout


def test_metadata_speaker_list_swapped(tmp_path):
    out, _ = _swap(tmp_path, _data())
    assert out["metadata"]["speaker_names"] == ["Speaker B", "Speaker A"]


def test_warnings_and_unrelated_fields_unchanged(tmp_path):
    original = _data()
    out, _ = _swap(tmp_path, copy.deepcopy(original))
    assert out["metadata"]["warnings"] == [_W1]
    for key in ("job_id", "source_file", "context", "num_speakers"):
        assert out["metadata"][key] == original["metadata"][key]
    for before, after in zip(original["transcript"], out["transcript"]):
        for key in before:
            if key not in ("speaker", "label"):
                assert after[key] == before[key]
    # Text that mentions a name is not a speaker field.
    assert out["transcript"][0]["text"] == "I am Speaker A"


def test_label_with_other_value_untouched(tmp_path):
    data = _data()
    data["transcript"][2]["label"] = "SPEAKER_00"
    out, _ = _swap(tmp_path, data)
    assert out["transcript"][2]["label"] == "SPEAKER_00"


def test_unknown_label_errors_and_writes_nothing(tmp_path):
    src, out = _write(tmp_path, _data()), tmp_path / "out.json"
    result = _run(src, str(out), "Speaker A", "Speaker C")
    assert result.returncode != 0
    assert "'Speaker C'" in result.stderr
    assert not out.exists()


def test_same_labels_error(tmp_path):
    src, out = _write(tmp_path, _data()), tmp_path / "out.json"
    result = _run(src, str(out), "Speaker A", "Speaker A")
    assert result.returncode != 0
    assert "same" in result.stderr
    assert not out.exists()


def test_out_equal_to_input_refused(tmp_path):
    src = _write(tmp_path, _data())
    before = open(src, encoding="utf-8").read()
    for out in (src, os.path.join(str(tmp_path), ".", "in.json")):
        result = _run(src, out, "Speaker A", "Speaker B")
        assert result.returncode != 0
        assert "input" in result.stderr
    assert open(src, encoding="utf-8").read() == before


def test_dry_run_writes_nothing(tmp_path):
    src, out = _write(tmp_path, _data()), tmp_path / "out.json"
    result = _run(src, str(out), "Speaker A", "Speaker B", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Segments Speaker A -> Speaker B: 2" in result.stdout
    assert not out.exists()


def test_non_ascii_names_round_trip(tmp_path):
    a, b = "Zoë", "Rachità 李"
    out, _ = _swap(tmp_path, _data(a, b), a, b)
    assert [s["speaker"] for s in out["transcript"]] == [b, a, b]
    assert out["metadata"]["speaker_names"] == [b, a]
    raw = (tmp_path / "out.json").read_text(encoding="utf-8")
    assert "Zoë" in raw and "\\u" not in raw   # ensure_ascii=False
    assert raw.startswith('{\n  "metadata"')   # indent=2
