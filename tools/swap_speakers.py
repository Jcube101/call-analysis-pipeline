# swap_speakers.py
"""
Swap two speaker labels throughout a transcript JSON, e.g. to fix label
flipping on a single-mic recording.

    python tools/swap_speakers.py in.json out.json "Speaker A" "Speaker B" [--dry-run]

Swapped wherever a speaker label is stored: every "speaker" string field at
any depth, a "label" field when it holds one of the two names, and every
list of strings under a key containing "speaker" (e.g. metadata.speaker_names).
Everything else, metadata.warnings included, is written back unchanged.
The input file is never overwritten.
"""
import json, os, sys, argparse

LABEL_KEYS = ("speaker", "label")

def swap_value(v, a, b):
    return b if v == a else a if v == b else v

def walk(node, a, b, counts):
    """Swap a <-> b in place; count changed segments and other values."""
    if isinstance(node, list):
        for item in node:
            walk(item, a, b, counts)
        return
    if not isinstance(node, dict):
        return
    for key, val in node.items():
        if key in LABEL_KEYS and isinstance(val, str):
            new = swap_value(val, a, b)
            if new != val:
                node[key] = new
                counts["found"].add(val)
                if key == "speaker":
                    counts["a_to_b" if val == a else "b_to_a"] += 1
                else:
                    counts["other"] += 1
        elif "speaker" in key.lower() and isinstance(val, list) and val \
                and all(isinstance(x, str) for x in val):
            for i, x in enumerate(val):
                new = swap_value(x, a, b)
                if new != x:
                    val[i] = new
                    counts["found"].add(x)
                    counts["other"] += 1
        else:
            walk(val, a, b, counts)

def same_path(p, q):
    norm = lambda x: os.path.normcase(os.path.realpath(x))
    return norm(p) == norm(q)

def main():
    # Names may be non-ASCII; never let a cp1252 console crash the summary.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("out")
    ap.add_argument("a", metavar="A", help='first label, e.g. "Speaker A"')
    ap.add_argument("b", metavar="B", help='second label, e.g. "Speaker B"')
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.a == a.b:
        raise SystemExit(f"The two labels are the same ({a.a!r}); nothing to swap.")
    if same_path(a.input, a.out):
        raise SystemExit("Output path is the input file; choose a different output path.")

    with open(a.input, encoding="utf-8") as f:
        data = json.load(f)

    counts = {"a_to_b": 0, "b_to_a": 0, "other": 0, "found": set()}
    walk(data, a.a, a.b, counts)

    missing = [x for x in (a.a, a.b) if x not in counts["found"]]
    if missing:
        raise SystemExit("Label not found in any speaker field: "
                         + ", ".join(repr(x) for x in missing) + ". Nothing written.")

    print(f"Segments {a.a} -> {a.b}: {counts['a_to_b']}")
    print(f"Segments {a.b} -> {a.a}: {counts['b_to_a']}")
    if counts["other"]:
        print(f"Other speaker values swapped (labels, speaker lists): {counts['other']}")

    if a.dry_run:
        print(f"Dry run; would write {a.out}")
        return

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()
