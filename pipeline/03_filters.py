"""Step 3: deterministic quality filters on every unscreened candidate.

Hard flags (disqualify): lang (output not Romanian), loop (n-gram repetition,
e.g. "ca o sabie, ca o sabie, ..."), empty (missing/truncated output).
Soft flags (information only, QE decides): ratio (suspicious length), num
(digits differ -- legit when numbers get spelled out in Romanian).

Cheap, pure-CPU, resumable (flags IS NULL = not yet screened).
"""

import re
import sys
from collections import Counter

from common import chunked, connect

try:
    from lingua import Language, LanguageDetectorBuilder

    DETECTOR = (
        LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.ROMANIAN)
        .with_low_accuracy_mode()
        .build()
    )
except ImportError:
    DETECTOR = None
    print("[warn] lingua-language-detector neinstalat -- sar flag-ul 'lang'", file=sys.stderr)


def has_loop(text: str) -> bool:
    words = text.split()
    for n, k in ((1, 5), (2, 4), (3, 3), (4, 3)):
        span = n * k
        for i in range(len(words) - span + 1):
            gram = words[i : i + n]
            if all(words[i + j * n : i + (j + 1) * n] == gram for j in range(1, k)):
                return True
    return False


def digit_groups(s: str):
    return sorted(re.findall(r"\d+", s))


def get_flags(en: str, ro: str) -> str:
    en, ro = en.strip(), ro.strip()
    flags = []
    if len(ro) < max(2, 0.25 * len(en)):
        flags.append("empty")
    if has_loop(ro):
        flags.append("loop")
    if DETECTOR is not None and len(ro) > 25:
        lang = DETECTOR.detect_language_of(ro)
        if lang is not None and lang == Language.ENGLISH:
            flags.append("lang")
    r = len(ro) / max(1, len(en))
    if r < 0.5 or r > 2.5:
        flags.append("ratio")
    if digit_groups(en) != digit_groups(ro):
        flags.append("num")
    return ",".join(flags)


def main():
    con = connect()
    rows = con.execute(
        "SELECT t.id, s.text_en, t.text_ro FROM translations t"
        " JOIN segments s ON s.id = t.seg_id WHERE t.flags IS NULL"
    ).fetchall()
    print(f"De verificat: {len(rows)} candidați")
    hist = Counter()
    for chunk in chunked(rows, 20000):
        updates = []
        for tid, en, ro in chunk:
            flags = get_flags(en, ro)
            updates.append((flags, tid))
            for f in flags.split(",") if flags else ["clean"]:
                hist[f] += 1
        con.executemany("UPDATE translations SET flags=? WHERE id=?", updates)
        con.commit()
        print(f"  ...{sum(hist.values())} procesate", end="\r", flush=True)
    print("\n=== Histogramă flag-uri ===")
    for flag, count in hist.most_common():
        print(f"{flag}: {count}")


if __name__ == "__main__":
    main()
