"""Shared helpers for the RO-NLI translation pipeline.

Data model (SQLite, pipeline/ro_nli.db):
  segments      -- one row per unique English sentence (dataset, split, sid)
  translations  -- zero or more Romanian candidates per segment; the legacy
                   Gemma-12B output, the 4B bulk pass and the 12B rescue pass
                   are all just rows here. Export picks the best clean one.

Status values on segments: pending -> translated -> rescue -> rescued.
flags on translations: NULL = filters not run yet, '' = clean,
otherwise comma-separated list. Hard flags disqualify a candidate.
"""

import json
import re
import sqlite3
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent
ROOT = PIPE_DIR.parent          # RO-NLI/
CORPUS = ROOT / "corpus_dev"
OUT_DIR = PIPE_DIR / "output"
DB_PATH = PIPE_DIR / "ro_nli.db"

HARD_FLAGS = {"lang", "loop", "empty"}

MLX_MODELS = {
    "4b": "mlx-community/translategemma-4b-it-4bit",
    "12b": "mlx-community/translategemma-12b-it-4bit",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id      INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL,
    split   TEXT NOT NULL,
    sid     INTEGER NOT NULL,
    text_en TEXT NOT NULL,
    n_words INTEGER NOT NULL,
    status  TEXT NOT NULL DEFAULT 'pending',
    UNIQUE (dataset, split, sid)
);
CREATE TABLE IF NOT EXISTS translations (
    id      INTEGER PRIMARY KEY,
    seg_id  INTEGER NOT NULL REFERENCES segments(id),
    model   TEXT NOT NULL,
    source  TEXT NOT NULL,
    text_ro TEXT NOT NULL,
    flags   TEXT,
    qe      REAL,
    UNIQUE (seg_id, model, text_ro)
);
CREATE INDEX IF NOT EXISTS idx_tr_seg ON translations(seg_id);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def load_relaxed_json(path):
    """Load JSON files written incrementally: trailing commas, possibly an
    unterminated top-level array (the historical RO_* files)."""
    s = Path(path).read_text()
    s = re.sub(r",\s*([\]}])", r"\1", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(s.rstrip().rstrip(",") + "]")


def is_clean(flags: str | None) -> bool:
    """A candidate is usable if filters ran and produced no hard flag."""
    if flags is None:
        return False
    return not (HARD_FLAGS & set(flags.split(",")))


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
