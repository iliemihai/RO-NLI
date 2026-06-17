"""Step 5: mark segments whose best candidate is unusable or below threshold.

Prag STRATIFICAT pe lungime (motivat empiric pe SNLI 4B):
  n_words 1-4  : tau=0.45  -- CometKiwi nesigur pe fragmente scurte (caption-style);
                              acceptăm orice nu e hard-flagged, prindem doar catastrofe
  n_words 5-9  : tau=0.65  -- zonă intermediară
  n_words 10+  : tau=0.72  -- texte lungi, QE e de încredere

Observație empirică: 98% din SNLI e 1-4 cuvinte; un prag plat de 0.70 ar marca
~490 traduceri corecte pentru rescue (false positives CometKiwi pe segmente scurte).

A segment needs rescue when it has no clean candidate (hard-flagged only) or
its best clean QE score is under the length-stratified threshold. Segments that
already have a 12B candidate are left alone (no infinite loop).
"""

import argparse
from collections import defaultdict

from common import connect, is_clean

# Praguri stratificate pe lungime (cuvinte EN)
TAU_STRATIFIED = [
    (1, 4,  0.45),
    (5, 9,  0.65),
    (10, 9999, 0.72),
]


def tau_for(n_words: int) -> float:
    for lo, hi, tau in TAU_STRATIFIED:
        if lo <= n_words <= hi:
            return tau
    return 0.72


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "mnli", "snli"])
    ap.add_argument("--flat-tau", type=float, default=None,
                    help="depășire: folosește un singur prag plat (pentru comparație/debug)")
    args = ap.parse_args()

    con = connect()

    # încarcă n_words per segment
    n_words = dict(con.execute(
        "SELECT id, n_words FROM segments WHERE (?='all' OR dataset=?)",
        (args.dataset, args.dataset),
    ))

    best = defaultdict(lambda: None)   # seg_id -> best clean qe
    has_12b = set()
    unscored_clean = set()

    for seg_id, model, flags, qe in con.execute(
        "SELECT t.seg_id, t.model, t.flags, t.qe FROM translations t"
        " JOIN segments s ON s.id=t.seg_id WHERE (?='all' OR s.dataset=?)",
        (args.dataset, args.dataset),
    ):
        if model.startswith("translategemma-12b"):
            has_12b.add(seg_id)
        if not is_clean(flags):
            continue
        if qe is None:
            unscored_clean.add(seg_id)
            continue
        if best[seg_id] is None or qe > best[seg_id]:
            best[seg_id] = qe

    # raport stratificat — util ca să validezi că pragurile au sens
    print("=== Distribuție per bandă de lungime ===")
    print(f"{'band':8s} | {'tau':5s} | {'total':7s} | {'sub tau':8s} | {'%':6s}")
    for lo, hi, default_tau in TAU_STRATIFIED:
        tau = args.flat_tau if args.flat_tau else default_tau
        segs_in_band = [s for s, n in n_words.items() if lo <= n <= hi and s in best and best[s] is not None]
        sub = sum(1 for s in segs_in_band if best[s] < tau)
        pct = 100 * sub / len(segs_in_band) if segs_in_band else 0
        band_label = f"{lo}-{min(hi,99)}"
        print(f"{band_label:8s} | {tau:.2f}  | {len(segs_in_band):7d} | {sub:8d} | {pct:.0f}%")
    print()

    if unscored_clean:
        print(f"[warn] {len(unscored_clean)} segmente cu candidat curat dar fără QE — rulează 04_qe.py întâi")

    all_seg_ids = set(n_words.keys())
    no_clean = [s for s in all_seg_ids if best[s] is None and s not in unscored_clean]
    low_qe = [
        s for s, q in best.items()
        if q is not None and q < (args.flat_tau if args.flat_tau else tau_for(n_words.get(s, 10)))
    ]
    candidates = [s for s in set(no_clean) | set(low_qe) if s not in has_12b]

    con.executemany(
        "UPDATE segments SET status='rescue' WHERE id=?", [(s,) for s in candidates]
    )
    con.commit()
    mode = f"flat tau={args.flat_tau}" if args.flat_tau else "stratificat"
    print(f"Fără candidat curat: {len(no_clean)}")
    print(f"Sub prag QE ({mode}): {len(low_qe)}")
    print(f"Marcate rescue cu 12B: {len(candidates)}"
          f" (excluse {len(set(no_clean+low_qe))-len(candidates)} care au deja 12B)")
    print("\nPentru comparație cu prag plat:")
    print(f"  python 05_rescue.py --flat-tau 0.70")


if __name__ == "__main__":
    main()
