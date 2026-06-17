"""Step 6: pick the winning candidate per segment and export final files.

Outputs in pipeline/output/:
  ro_{dataset}_{split}_sentence2id.json   -- valid JSON, [[text_ro, sid], ...]
  {dataset}_{split}_pairs.jsonl           -- premise/hypothesis RO+EN, label, min QE
  excluded.jsonl                          -- segments without an acceptable candidate
  report.md                               -- counts, QE stats per model, flags

Winner = clean candidate (no hard flag) with the highest QE >= --tau.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict

from common import CORPUS, OUT_DIR, connect, is_clean, load_relaxed_json

MAPPINGS = {
    ("mnli", "train"): CORPUS / "mnli_data/train_mapping.json",
    ("mnli", "dev_matched"): CORPUS / "mnli_data/dev_matched_mapping.json",
    ("mnli", "dev_mismatched"): CORPUS / "mnli_data/dev_mismatched_mapping.json",
    ("snli", "train"): CORPUS / "snli_data/train_mapping.json",
    ("snli", "dev"): CORPUS / "snli_data/dev_mapping.json",
    ("snli", "test"): CORPUS / "snli_data/test_mapping.json",
}


def pick_winners(con, tau):
    winners = {}  # seg_id -> (text_ro, qe, model)
    excluded = {}  # seg_id -> reason
    by_seg = defaultdict(list)
    for seg_id, model, text_ro, flags, qe in con.execute(
        "SELECT seg_id, model, text_ro, flags, qe FROM translations"
    ):
        by_seg[seg_id].append((model, text_ro, flags, qe))
    for (seg_id,) in con.execute("SELECT id FROM segments"):
        cands = [
            (qe, text, model)
            for model, text, flags, qe in by_seg.get(seg_id, [])
            if is_clean(flags) and qe is not None
        ]
        good = [c for c in cands if c[0] >= tau]
        if good:
            qe, text, model = max(good)
            winners[seg_id] = (text, qe, model)
        elif cands:
            excluded[seg_id] = f"best_qe_below_tau ({max(cands)[0]:.3f})"
        elif by_seg.get(seg_id):
            excluded[seg_id] = "only_flagged_or_unscored"
        else:
            excluded[seg_id] = "untranslated"
    return winners, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.70)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    con = connect()
    winners, excluded = pick_winners(con, args.tau)
    seg_info = {
        seg_id: (dataset, split, sid, text_en)
        for seg_id, dataset, split, sid, text_en in con.execute(
            "SELECT id, dataset, split, sid, text_en FROM segments"
        )
    }

    # sentence2id files + per-split winner lookup keyed by sid
    by_split = defaultdict(dict)  # (dataset, split) -> sid -> (text_ro, qe)
    for seg_id, (text_ro, qe, _model) in winners.items():
        dataset, split, sid, _ = seg_info[seg_id]
        by_split[(dataset, split)][sid] = (text_ro, qe)
    for (dataset, split), sids in sorted(by_split.items()):
        path = OUT_DIR / f"ro_{dataset}_{split}_sentence2id.json"
        path.write_text(
            json.dumps(
                [[text, sid] for sid, (text, _) in sorted(sids.items())],
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"{path.name}: {len(sids)} propoziții")

    # pair-level exports via the mapping files
    pair_stats = {}
    for (dataset, split), map_path in MAPPINGS.items():
        if not map_path.exists():
            continue
        mapping = load_relaxed_json(map_path)
        lookup = by_split.get((dataset, split), {})
        out_path = OUT_DIR / f"{dataset}_{split}_pairs.jsonl"
        kept = dropped = 0
        with out_path.open("w") as fh:
            for pair_id, (pid, hid, label, *_rest) in mapping.items():
                p, h = lookup.get(pid), lookup.get(hid)
                if p is None or h is None:
                    dropped += 1
                    continue
                fh.write(
                    json.dumps(
                        {
                            "pair_id": pair_id,
                            "premise_ro": p[0],
                            "hypothesis_ro": h[0],
                            "label": label,
                            "qe_min": round(min(p[1], h[1]), 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
        pair_stats[(dataset, split)] = (kept, dropped)
        print(f"{out_path.name}: {kept} perechi (-{dropped} cu segmente excluse)")

    with (OUT_DIR / "excluded.jsonl").open("w") as fh:
        for seg_id, reason in excluded.items():
            dataset, split, sid, text_en = seg_info[seg_id]
            fh.write(
                json.dumps(
                    {"dataset": dataset, "split": split, "sid": sid, "reason": reason, "text_en": text_en},
                    ensure_ascii=False,
                )
                + "\n"
            )

    # report
    model_scores = defaultdict(list)
    for text_ro, qe, model in winners.values():
        model_scores[model].append(qe)
    flag_hist = Counter()
    for (flags,) in con.execute("SELECT flags FROM translations WHERE flags != ''"):
        for f in flags.split(","):
            flag_hist[f] += 1
    lines = [
        "# Raport export RO-NLI",
        f"\ntau = {args.tau}",
        f"\nSegmente finale: {len(winners)} | excluse: {len(excluded)}",
        "\n## Câștigători per model",
    ]
    for model, scores in sorted(model_scores.items()):
        lines.append(
            f"- {model}: {len(scores)} segmente, QE medie {statistics.fmean(scores):.4f}"
        )
    lines.append("\n## Motive excludere")
    for reason, n in Counter(excluded.values()).most_common():
        lines.append(f"- {reason}: {n}")
    lines.append("\n## Flag-uri (toți candidații)")
    for flag, n in flag_hist.most_common():
        lines.append(f"- {flag}: {n}")
    lines.append("\n## Perechi exportate")
    for (dataset, split), (kept, dropped) in sorted(pair_stats.items()):
        lines.append(f"- {dataset}/{split}: {kept} (-{dropped})")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nRaport: {OUT_DIR/'report.md'}")


if __name__ == "__main__":
    main()
