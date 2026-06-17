"""Step 1: ingest English sentences + existing Romanian translations into the DB.

Idempotent: re-running only adds what is missing (INSERT OR IGNORE everywhere).
Stdlib only -- no ML dependencies needed for this step.
"""

import sys

from common import CORPUS, connect, load_relaxed_json

EN_SOURCES = [
    ("mnli", "train", CORPUS / "mnli_data/train_sentence2id.json"),
    ("mnli", "dev_matched", CORPUS / "mnli_data/dev_matched_sentence2id.json"),
    ("mnli", "dev_mismatched", CORPUS / "mnli_data/dev_mismatched_sentence2id.json"),
    ("snli", "train", CORPUS / "snli_data/train_sentence2id.json"),
    ("snli", "dev", CORPUS / "snli_data/dev_sentence2id.json"),
    ("snli", "test", CORPUS / "snli_data/test_sentence2id.json"),
]

# Historical Gemma-12B output, scattered across runs. Exact duplicates collapse
# via the UNIQUE constraint; genuinely different texts for the same sentence
# (e.g. manual corrections) coexist as candidates and QE picks the winner.
RO_LEGACY = [
    ("mnli", "train", CORPUS / "data_100k/ro_new_train_sentence2id.json"),
    ("mnli", "train", CORPUS / "data_100k/ro_train_sentence2id.json"),
    ("mnli", "train", CORPUS / "mnli_data/ro_train_sentence2id.json"),
    ("mnli", "dev_matched", CORPUS / "data_100k/ro_dev_matched_sentence2id.json"),
    ("mnli", "dev_mismatched", CORPUS / "data_100k/ro_dev_mismatched_sentence2id.json"),
    ("mnli", "dev_matched", CORPUS / "mnli_data/partial_data/ro_dev_matched_sentence2id.json"),
    ("mnli", "dev_mismatched", CORPUS / "mnli_data/partial_data/ro_dev_mismatched_sentence2id.json"),
]

LEGACY_MODEL_TAG = "legacy-gemma12b"


def ingest_english(con):
    for dataset, split, path in EN_SOURCES:
        if not path.exists():
            print(f"[warn] lipsește {path}", file=sys.stderr)
            continue
        rows = load_relaxed_json(path)
        data = [
            (dataset, split, sid, text, len(text.split()))
            for text, sid in rows
            if isinstance(text, str) and text.strip()
        ]
        con.executemany(
            "INSERT OR IGNORE INTO segments(dataset, split, sid, text_en, n_words)"
            " VALUES (?,?,?,?,?)",
            data,
        )
        con.commit()
        print(f"EN  {dataset}/{split}: {len(data)} propoziții ({path.name})")


def ingest_legacy_ro(con):
    existing = [(d, s, p) for d, s, p in RO_LEGACY if p.exists()]
    for d, s, p in set(RO_LEGACY) - set(existing):
        print(f"[warn] lipsește {p}", file=sys.stderr)
    # oldest first, so newer files (manual corrections) are inserted last
    existing.sort(key=lambda t: t[2].stat().st_mtime)

    for dataset, split, path in existing:
        sid2seg = dict(
            con.execute(
                "SELECT sid, id FROM segments WHERE dataset=? AND split=?",
                (dataset, split),
            ).fetchall()
        )
        try:
            rows = load_relaxed_json(path)
        except Exception as e:
            print(f"[warn] nu pot citi {path.name}: {e}", file=sys.stderr)
            continue
        data, orphans, skipped = [], 0, 0
        for row in rows:
            if not (isinstance(row, list) and len(row) == 2 and isinstance(row[0], str)):
                skipped += 1
                continue
            text_ro, sid = row
            seg_id = sid2seg.get(sid)
            if seg_id is None:
                orphans += 1
                continue
            if not text_ro.strip():
                skipped += 1
                continue
            data.append((seg_id, LEGACY_MODEL_TAG, path.name, text_ro.strip()))
        con.executemany(
            "INSERT OR IGNORE INTO translations(seg_id, model, source, text_ro)"
            " VALUES (?,?,?,?)",
            data,
        )
        con.commit()
        note = f", {orphans} id-uri orfane" if orphans else ""
        note += f", {skipped} rânduri sărite" if skipped else ""
        print(f"RO  {dataset}/{split}: +{len(data)} candidați din {path.name}{note}")


def report(con):
    print("\n=== Stadiu după ingest ===")
    rows = con.execute(
        """
        SELECT s.dataset, s.split, COUNT(*) AS total,
               SUM(EXISTS(SELECT 1 FROM translations t WHERE t.seg_id = s.id)) AS tradus
        FROM segments s GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()
    for dataset, split, total, tradus in rows:
        print(f"{dataset}/{split}: {total} total, {tradus} cu traducere, {total - tradus} netraduse")
    con.execute(
        "UPDATE segments SET status='translated' WHERE status='pending' AND "
        "EXISTS(SELECT 1 FROM translations t WHERE t.seg_id = segments.id)"
    )
    con.commit()


if __name__ == "__main__":
    con = connect()
    ingest_english(con)
    ingest_legacy_ro(con)
    report(con)
