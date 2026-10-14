# merge_transcripts.py
"""
Merge two transcript JSONs (part 1 + part 2 of one recording) into one.

    python tools/merge_transcripts.py part1.json part2.json merged.json [--gap 1.0]
        [--swap-speakers "Speaker A" "Speaker B"] [--dry-run]

Part 2's timestamps are offset to follow part 1. The output keeps part 1's
metadata, with metadata.warnings replaced by both parts' warnings, each
prefixed "Part 1: " or "Part 2: " so its source stays clear.

Supersedes output/merge_transcripts.py (that copy is gitignored and untracked).
"""
import json, sys, argparse

START_KEYS = ("start", "start_time")
END_KEYS = ("end", "end_time")

def find_segments(data):
    """Return (container, key) so the segment list can be replaced in place."""
    if isinstance(data, list):
        return None, None
    for key in ("segments", "transcript"):
        if isinstance(data.get(key), list):
            return data, key
    for key, val in data.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return data, key
    raise SystemExit("Couldn't find a segment list in the JSON. Send me the top-level keys.")

def get_segments(data):
    c, k = find_segments(data)
    return data if c is None else c[k]

def part_warnings(data, part):
    """metadata.warnings of one part, each prefixed with the part it came from."""
    meta = data.get("metadata") if isinstance(data, dict) else None
    warnings = meta.get("warnings") if isinstance(meta, dict) else None
    if isinstance(warnings, str):
        warnings = [warnings]
    if not isinstance(warnings, list):
        return []
    return [f"Part {part}: {w.strip()}" for w in warnings if isinstance(w, str) and w.strip()]

def max_end(segs):
    return max((s[k] for s in segs for k in END_KEYS if k in s), default=0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part1"); ap.add_argument("part2"); ap.add_argument("out")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between parts")
    ap.add_argument("--swap-speakers", nargs=2, metavar=("A", "B"),
                    help="swap two speaker labels in part 2, e.g. SPEAKER_00 SPEAKER_01")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    d1 = json.load(open(a.part1, encoding="utf-8"))
    d2 = json.load(open(a.part2, encoding="utf-8"))
    s1, s2 = get_segments(d1), get_segments(d2)

    offset = max_end(s1) + a.gap
    for s in s2:
        for k in START_KEYS + END_KEYS:
            if k in s:
                s[k] += offset
        if a.swap_speakers and "speaker" in s:
            x, y = a.swap_speakers
            s["speaker"] = y if s["speaker"] == x else x if s["speaker"] == y else s["speaker"]

    print(f"Part 1: {len(s1)} segments, ends at {max_end(s1):.1f}s")
    print(f"Part 2: {len(s2)} segments, offset by {offset:.1f}s")
    print("Speakers in part 1:", sorted({s.get("speaker") for s in s1}))
    print("Speakers in part 2:", sorted({s.get("speaker") for s in s2}))
    print("Sample keys:", list(s1[0].keys()))
    warnings = part_warnings(d1, 1) + part_warnings(d2, 2)
    for w in warnings:
        print("Warning:", w)

    if a.dry_run:
        return

    c, k = find_segments(d1)
    merged = s1 + s2
    if c is None:
        d1 = merged
        if warnings:
            print("Part 1 is a bare segment list with no metadata; warnings not written.")
    else:
        c[k] = merged
        if "segment_count" in c:
            c["segment_count"] = len(merged)
        meta = d1.get("metadata")
        if warnings and not isinstance(meta, dict):
            meta = d1["metadata"] = {}
        if isinstance(meta, dict):
            if warnings:
                meta["warnings"] = warnings
            else:
                meta.pop("warnings", None)
    json.dump(d1, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()