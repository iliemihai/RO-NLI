"""Step 4: reference-free quality estimation with CometKiwi on every candidate.

Needs: pip install unbabel-comet, plus HF login + accepted license for
Unbabel/wmt22-cometkiwi-da (gated model):  hf auth login

Runs on MPS (Apple GPU). Resumable: qe IS NULL = not yet scored; commits per
chunk. Skips candidates already hard-flagged by step 3 (no point scoring them).
"""

import argparse
import os
import statistics

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from common import HARD_FLAGS, chunked, connect


def hard_flagged(flags: str) -> bool:
    return bool(HARD_FLAGS & set(flags.split(","))) if flags else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="doar primii N (smoke test)")
    ap.add_argument("--sample", type=int, default=None, help="N aleși aleator (calibrare prag)")
    args = ap.parse_args()

    con = connect()
    rows = [
        (tid, en, ro)
        for tid, en, ro, flags in con.execute(
            "SELECT t.id, s.text_en, t.text_ro, t.flags FROM translations t"
            " JOIN segments s ON s.id = t.seg_id"
            " WHERE t.qe IS NULL AND t.flags IS NOT NULL"
        )
        if not hard_flagged(flags)
    ]
    if args.sample:
        import random

        random.seed(42)
        rows = random.sample(rows, min(args.sample, len(rows)))
    elif args.limit:
        rows = rows[: args.limit]
    print(f"De scorat: {len(rows)} candidați cu {args.model}")
    if not rows:
        return

    from comet import download_model, load_from_checkpoint

    ckpt = download_model(args.model)
    model = load_from_checkpoint(ckpt)
    model.eval()

    def predict(data):
        kwargs = dict(batch_size=args.batch, num_workers=2, progress_bar=True)
        if args.cpu:
            return model.predict(data, gpus=0, **kwargs)
        try:
            return model.predict(data, gpus=1, accelerator="mps", **kwargs)
        except Exception:
            return model.predict(data, gpus=1, **kwargs)  # let lightning pick

    all_scores = []
    for chunk in chunked(rows, args.chunk):
        data = [{"src": en, "mt": ro} for _, en, ro in chunk]
        out = predict(data)
        scores = list(out.scores if hasattr(out, "scores") else out[0])
        con.executemany(
            "UPDATE translations SET qe=? WHERE id=?",
            [(float(s), tid) for s, (tid, _, _) in zip(scores, chunk)],
        )
        con.commit()
        all_scores.extend(float(s) for s in scores)
        print(f"  comis {len(all_scores)}/{len(rows)}")

    qs = statistics.quantiles(all_scores, n=20)
    print("\n=== Distribuția scorurilor (rulajul curent) ===")
    print(f"medie={statistics.fmean(all_scores):.4f}  p5={qs[0]:.4f}  p25={qs[4]:.4f}  p50={qs[9]:.4f}  p95={qs[18]:.4f}")
    print("Folosește distribuția ca să alegi --tau pentru 05_rescue / 06_export.")


if __name__ == "__main__":
    main()
