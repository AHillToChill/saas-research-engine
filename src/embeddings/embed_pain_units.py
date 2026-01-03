from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/db/research.sqlite")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(s: str) -> str:
    if not s:
        return ""
    # Remove code-ish lines and normalize whitespace
    lines: list[str] = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        if any(tok in t for tok in ["Traceback", "Exception", "public static", "SELECT ", "INSERT ", "<div", "</", "{", "}", ";;", "=>"]):
            continue
        if len(t) > 300:
            continue
        lines.append(t)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class PU:
    id: int
    lane: str
    task: str
    friction: str
    consequence: str
    severity: int
    purchase_intent: int
    url: str


def ensure_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pain_unit_embeddings (
          pain_unit_id INTEGER PRIMARY KEY,
          model TEXT NOT NULL,
          dim INTEGER NOT NULL,
          embedding BLOB NOT NULL,
          created_at_utc TEXT NOT NULL,
          FOREIGN KEY (pain_unit_id) REFERENCES pain_units(id)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS ix_pue_model ON pain_unit_embeddings(model);")
    con.commit()


def fetch_pain_units(con: sqlite3.Connection) -> list[PU]:
    rows = con.execute(
        """
        SELECT
          pu.id,
          pu.lane,
          pu.task,
          pu.friction,
          pu.consequence,
          pu.severity,
          pu.purchase_intent,
          rd.url
        FROM pain_units pu
        JOIN raw_documents rd ON rd.id = pu.raw_document_id
        ORDER BY pu.id
        """
    ).fetchall()

    out: list[PU] = []
    for r in rows:
        out.append(
            PU(
                id=int(r[0]),
                lane=r[1] or "",
                task=r[2] or "",
                friction=r[3] or "",
                consequence=r[4] or "",
                severity=int(r[5] or 3),
                purchase_intent=int(r[6] or 0),
                url=r[7] or "",
            )
        )
    return out


def existing_ids(con: sqlite3.Connection, model: str) -> set[int]:
    rows = con.execute(
        "SELECT pain_unit_id FROM pain_unit_embeddings WHERE model = ?",
        (model,),
    ).fetchall()
    return {int(r[0]) for r in rows}


def build_doc(pu: PU) -> str:
    # Key design choice: do NOT use normalized_pain (your template).
    # Use the original-ish content fields.
    parts = [
        clean_text(pu.task),
        clean_text(pu.friction),
        clean_text(pu.consequence),
    ]
    return " ".join([p for p in parts if p]).strip()


def chunked(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--force", action="store_true", help="Recompute embeddings even if already present")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys=ON;")
        ensure_tables(con)

        pus = fetch_pain_units(con)
        if not pus:
            print("No pain_units found.")
            return

        have = set() if args.force else existing_ids(con, args.model)
        todo = [pu for pu in pus if pu.id not in have]

        print(f"pain_units total: {len(pus)}")
        print(f"embeddings present for model='{args.model}': {len(have)}")
        print(f"to embed now: {len(todo)}")

        if not todo:
            print("Nothing to do.")
            return

        # Load model (first run will download weights)
        model = SentenceTransformer(args.model)

        # Embed and store
        inserted = 0
        for batch in chunked(todo, args.batch):
            ids = [pu.id for pu in batch]
            docs = [build_doc(pu) for pu in batch]

            # Some docs may end up empty after cleaning; keep them but mark as empty embedding
            # (we'll filter them out in clustering).
            nonempty_idx = [i for i, d in enumerate(docs) if d]
            if not nonempty_idx:
                continue

            docs_nonempty = [docs[i] for i in nonempty_idx]
            ids_nonempty = [ids[i] for i in nonempty_idx]

            emb = model.encode(
                docs_nonempty,
                batch_size=args.batch,
                normalize_embeddings=True,   # important: cosine similarity becomes dot product
                show_progress_bar=False,
            )
            emb = np.asarray(emb, dtype=np.float32)
            dim = int(emb.shape[1])

            con.executemany(
                """
                INSERT OR REPLACE INTO pain_unit_embeddings(pain_unit_id, model, dim, embedding, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(pid, args.model, dim, emb[i].tobytes(), utcnow_iso()) for i, pid in enumerate(ids_nonempty)],
            )
            inserted += len(ids_nonempty)

            con.commit()
            print(f"Embedded + stored: {inserted}/{len(todo)}")

        print(f"Done. Inserted/updated embeddings: {inserted}")


if __name__ == "__main__":
    main()
