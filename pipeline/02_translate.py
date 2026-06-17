"""Step 2: translate pending segments with TranslateGemma via MLX.

Bulk pass:    python 02_translate.py --dataset snli --model 4b
Smoke test:   python 02_translate.py --dataset mnli --model 4b --limit 32
Rescue pass:  python 02_translate.py --dataset all --model 12b --rescue
              (only segments marked status='rescue' by 05_rescue.py)

Resumable: progress is committed per batch; re-running skips anything that
already has a candidate from the same model.
"""

import argparse
import re
import sys
import time

from common import MLX_MODELS, chunked, connect


def build_prompt(tokenizer, text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "ro",
                    "text": text,
                }
            ],
        }
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def clean_output(text: str) -> str:
    t = text.strip()
    for stop in ("<end_of_turn>", "<eos>", "<pad>"):
        t = t.split(stop)[0]
    return re.sub(r"\s*\n+\s*", " ", t).strip()


class Generator:
    """batch_generate if the installed mlx-lm has it, otherwise sequential."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self.generate = generate
        self.sampler = make_sampler(temp=0.0)
        try:
            from mlx_lm import batch_generate

            self.batch_fn = batch_generate
        except ImportError:
            self.batch_fn = None
        self.mode = None  # decided on first call

    def __call__(self, prompts, max_tokens):
        if self.batch_fn is not None and self.mode != "seq":
            try:
                res = self.batch_fn(
                    self.model,
                    self.tokenizer,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    sampler=self.sampler,
                    verbose=False,
                )
                self.mode = "batch"
                return list(res.texts)
            except Exception as e:
                if self.mode == "batch":
                    raise
                print(f"[warn] batch_generate indisponibil ({e}); trec pe secvențial", file=sys.stderr)
                self.mode = "seq"
        return [
            self.generate(
                self.model,
                self.tokenizer,
                prompt=p,
                max_tokens=max_tokens,
                sampler=self.sampler,
                verbose=False,
            )
            for p in prompts
        ]


def fetch_todo(con, dataset: str, model_tag: str, rescue: bool,
               limit: int | None, all_models: bool = False):
    where = "(? = 'all' OR s.dataset = ?)"
    if rescue:
        where += " AND s.status = 'rescue'"
    # nu retraduce ce a produs deja acest model
    where += " AND NOT EXISTS (SELECT 1 FROM translations t WHERE t.seg_id = s.id AND t.model = ?)"
    # fără --all-models: sare și segmentele care au orice traducere (alt model)
    if not rescue and not all_models:
        where += " AND NOT EXISTS (SELECT 1 FROM translations t WHERE t.seg_id = s.id)"
    q = (
        f"SELECT s.id, s.text_en, s.n_words FROM segments s WHERE {where}"
        " ORDER BY s.n_words, s.id"
    )
    if limit:
        q += f" LIMIT {int(limit)}"
    return con.execute(q, (dataset, dataset, model_tag)).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "mnli", "snli"])
    ap.add_argument("--model", default="4b", choices=sorted(MLX_MODELS))
    ap.add_argument("--batch", type=int, default=None, help="implicit: 24 (4b) / 8 (12b)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescue", action="store_true")
    ap.add_argument("--all-models", action="store_true",
                    help="traduce și segmente care au deja o traducere de alt model; "
                         "candidații coexistă în DB, QE alege câștigătorul la export")
    args = ap.parse_args()
    batch_size = args.batch or (24 if args.model == "4b" else 8)
    model_tag = f"translategemma-{args.model}-q4"

    con = connect()
    todo = fetch_todo(con, args.dataset, model_tag, args.rescue, args.limit,
                      all_models=args.all_models)
    if not todo:
        print("Nimic de tradus pentru selecția curentă.")
        return
    total_words = sum(n for _, _, n in todo)
    print(f"De tradus: {len(todo)} segmente (~{total_words/1000:.0f}k cuvinte EN) cu {model_tag}")

    from mlx_lm import load

    model, tokenizer = load(MLX_MODELS[args.model])
    gen = Generator(model, tokenizer)

    done, t0 = 0, time.time()
    for batch in chunked(todo, batch_size):
        max_words = max(n for _, _, n in batch)
        max_tokens = min(512, max(64, int(max_words * 3.2) + 24))
        prompts = [build_prompt(tokenizer, en) for _, en, _ in batch]
        outputs = gen(prompts, max_tokens)
        rows = [
            (seg_id, model_tag, "pipeline", clean_output(out))
            for (seg_id, _, _), out in zip(batch, outputs)
            if clean_output(out)
        ]
        con.executemany(
            "INSERT OR IGNORE INTO translations(seg_id, model, source, text_ro)"
            " VALUES (?,?,?,?)",
            rows,
        )
        new_status = "rescued" if args.rescue else "translated"
        con.executemany(
            "UPDATE segments SET status=? WHERE id=?",
            [(new_status, seg_id) for seg_id, _, _ in batch],
        )
        con.commit()
        done += len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed
        eta_h = (len(todo) - done) / rate / 3600 if rate else 0
        print(
            f"\r{done}/{len(todo)} | {rate:.1f} seg/s | mod={gen.mode} | ETA {eta_h:.1f}h ",
            end="",
            flush=True,
        )
    print(f"\nGata: {done} segmente în {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
