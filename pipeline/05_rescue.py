"""Step 5: mark segments whose best candidate is unusable or below --tau.

A segment needs rescue when it has no clean candidate (hard-flagged only) or
its best clean QE score is under the threshold. Marked segments get
status='rescue'; then run:

    python 02_translate.py --dataset all --model 12b --rescue
    python 03_filters.py && python 04_qe.py

Segments that already have a 12B candidate are left alone (no infinite loop);
whatever scored best wins at export, and the rest land in excluded.jsonl.
"""

import argparse
from collections import defaultdict

from common import connect, is_clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.70, help="prag QE provizoriu; calibrează-l pe distribuția din 04")
    ap.add_argument("--dataset", default="all", choices=["all", "mnli", "snli"])
    args = ap.parse_args()

    con = connect()
    best = defaultdict(lambda: None)  # seg_id -> best clean qe
    has_12b = set()
    unscored = set()
    for seg_id, model, flags, qe in con.execute(
        "SELECT t.seg_id, t.model, t.flags, t.qe FROM translations t"
        " JOIN segments s ON s.id = t.seg_id WHERE (?='all' OR s.dataset=?)",
        (args.dataset, args.dataset),
    ):
        if model.startswith("translategemma-12b"):
            has_12b.add(seg_id)
        if not is_clean(flags):
            continue
        if qe is None:
            unscored.add(seg_id)
            continue
        if best[seg_id] is None or qe > best[seg_id]:
            best[seg_id] = qe

    if unscored - set(k for k, v in best.items() if v is not None and v >= args.tau):
        print(f"[warn] {len(unscored)} segmente au candidați nescorați -- rulează 04_qe.py întâi")

    all_segs = {
        seg_id
        for (seg_id,) in con.execute(
            "SELECT id FROM segments WHERE (?='all' OR dataset=?)",
            (args.dataset, args.dataset),
        )
    }
    no_clean = [s for s in all_segs if best[s] is None and s not in unscored]
    low_qe = [s for s, q in best.items() if q is not None and q < args.tau]
    candidates = [s for s in set(no_clean) | set(low_qe) if s not in has_12b]

    con.executemany(
        "UPDATE segments SET status='rescue' WHERE id=?", [(s,) for s in candidates]
    )
    con.commit()
    print(f"Fără candidat curat: {len(no_clean)} | sub tau={args.tau}: {len(low_qe)}")
    print(f"Marcate pentru rescue cu 12B: {len(candidates)} (excluse {len(set(no_clean+low_qe))-len(candidates)} care au deja 12B)")


if __name__ == "__main__":
    main()
