# Pipeline traducere RO-NLI (MNLI + SNLI)

Cascadă: **TranslateGemma 4B (bulk) → filtre deterministe → CometKiwi QE →
rescue cu 12B → export**. Traducerile vechi (Gemma-12B, 2023-2024) sunt
importate ca candidați și concurează pe QE cu cele noi — corecturile tale
manuale câștigă automat dacă sunt mai bune.

## Setup (o singură dată) — DEJA FĂCUT pe acest laptop

Două venv-uri, pentru că unbabel-comet cere transformers<5 iar mlx-lm cere ≥5:

```bash
cd pipeline
python3 -m venv --system-site-packages .venv       # traducere + filtre
.venv/bin/pip install mlx-lm lingua-language-detector tqdm
python3 -m venv --system-site-packages .venv-qe    # CometKiwi
.venv-qe/bin/pip install unbabel-comet
hf auth login   # pentru CometKiwi (model gated: acceptă licența pe
                # huggingface.co/Unbabel/wmt22-cometkiwi-da)
```

## Rulare

```bash
# 1. Ingest (stdlib only, idempotent) — încarcă EN + traducerile existente
.venv/bin/python 01_ingest.py

# 2. Bulk SNLI (~655k propoziții, estimat 1-2 zile; Ctrl-C + reia oricând)
caffeinate -dimsu .venv/bin/python 02_translate.py --dataset snli --model 4b

# 3. Filtre deterministe (CPU, ~5 min / 500k)
.venv/bin/python 03_filters.py

# 4. QE pe MPS (~30-40 seg/s ⇒ ~4-5h / 500k, resumable)
caffeinate -dimsu .venv-qe/bin/python 04_qe.py

# 5. Uită-te la distribuția scorurilor tipărită de 04, alege tau, apoi:
.venv/bin/python 05_rescue.py --tau 0.70
caffeinate -dimsu .venv/bin/python 02_translate.py --dataset all --model 12b --rescue
.venv/bin/python 03_filters.py && .venv-qe/bin/python 04_qe.py

# 6. Export final + raport
.venv/bin/python 06_export.py --tau 0.70
```

Totul e **resumable**: progresul se scrie în `ro_nli.db` (SQLite) la fiecare
batch; Ctrl-C oricând, reia cu aceeași comandă.

## Note

- `--tau 0.70` e provizoriu. Calibrarea corectă: după pasul 4, ia ~300 de
  candidați stratificați pe scor, judecă-i manual și alege tau astfel încât
  P(traducere proastă | scor ≥ tau) < 2%.
- Scorurile QE rămân în DB per candidat → la publicare poți raporta
  distribuția completă, nu doar media.
- `output/excluded.jsonl` = segmentele fără candidat acceptabil; puține la
  număr, se rezolvă manual sau se documentează ca excluse.
