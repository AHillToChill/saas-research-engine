from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DB_PATH = Path("data/db/research.sqlite")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}  # r[1] is column name


def pick_col(cols: set[str], candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


@dataclass(frozen=True)
class SourceKey:
    lane: str
    site: str
    query: str


def ensure_source(con: sqlite3.Connection, lane: str, site: str, query: str) -> int:
    cols = table_columns(con, "sources")

    id_col = pick_col(cols, ["id"])
    lane_col = pick_col(cols, ["lane"])
    site_col = pick_col(cols, ["site", "source", "domain", "host"])
    query_col = pick_col(cols, ["query", "q", "search_query"])
    created_col = pick_col(cols, ["created_at_utc", "created_at", "createdAtUtc"])

    if not (id_col and site_col and query_col):
        raise RuntimeError("sources table does not have expected columns (need at least id, site/source, query).")

    # Reuse an existing source row if present (keeps your DB tidy).
    where_parts = []
    params: list[Any] = []

    if lane_col:
        where_parts.append(f"{lane_col} = ?")
        params.append(lane)
    where_parts.append(f"{site_col} = ?")
    params.append(site)
    where_parts.append(f"{query_col} = ?")
    params.append(query)

    sel = f"SELECT {id_col} FROM sources WHERE " + " AND ".join(where_parts) + f" ORDER BY {id_col} DESC LIMIT 1"
    row = con.execute(sel, params).fetchone()
    if row:
        return int(row[0])

    # Insert new source
    ins_cols: list[str] = []
    ins_vals: list[Any] = []

    if lane_col:
        ins_cols.append(lane_col)
        ins_vals.append(lane)
    ins_cols.append(site_col)
    ins_vals.append(site)
    ins_cols.append(query_col)
    ins_vals.append(query)
    if created_col:
        ins_cols.append(created_col)
        ins_vals.append(utcnow_iso())

    placeholders = ",".join(["?"] * len(ins_cols))
    sql = f"INSERT INTO sources({','.join(ins_cols)}) VALUES ({placeholders})"
    cur = con.execute(sql, ins_vals)
    return int(cur.lastrowid)


def insert_raw_document(
    con: sqlite3.Connection,
    source_id: int,
    url: str,
    title: str,
    text: str,
    extra: dict[str, Any] | None = None,
) -> None:
    cols = table_columns(con, "raw_documents")

    # Identify best-fit columns (supports schema drift).
    source_col = pick_col(cols, ["source_id", "sourceId", "source"])
    url_col = pick_col(cols, ["url", "link"])
    title_col = pick_col(cols, ["title", "doc_title", "name"])
    text_col = pick_col(cols, ["text", "body", "content", "raw_text", "document_text"])
    created_col = pick_col(cols, ["created_at_utc", "created_at", "createdAtUtc", "captured_at_utc"])
    extra_col = pick_col(cols, ["extra_json", "meta_json", "metadata_json", "raw_json"])

    if not url_col:
        raise RuntimeError("raw_documents table must have a url/link column.")

    insert_cols: list[str] = []
    insert_vals: list[Any] = []

    if source_col:
        insert_cols.append(source_col)
        insert_vals.append(source_id)

    insert_cols.append(url_col)
    insert_vals.append(url)

    if title_col:
        insert_cols.append(title_col)
        insert_vals.append(title or "")

    if text_col:
        insert_cols.append(text_col)
        insert_vals.append(text or "")

    if created_col:
        insert_cols.append(created_col)
        insert_vals.append(utcnow_iso())

    if extra_col:
        insert_cols.append(extra_col)
        insert_vals.append(json.dumps(extra or {}, ensure_ascii=False))

    placeholders = ",".join(["?"] * len(insert_cols))
    sql = f"INSERT OR IGNORE INTO raw_documents({','.join(insert_cols)}) VALUES ({placeholders})"
    con.execute(sql, insert_vals)
