from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import feedparser
from bs4 import BeautifulSoup

DB_PATH = Path("data/db/research.sqlite")
FEEDS_PATH = Path("src/collectors/rss_feeds.txt")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_html(s: str) -> str:
    if not s:
        return ""
    soup = BeautifulSoup(s, "html.parser")
    txt = soup.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def pick_col(cols: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


@dataclass(frozen=True)
class FeedSpec:
    lane: str
    site: str
    url: str


def load_feeds(path: Path) -> list[FeedSpec]:
    if not path.exists():
        raise FileNotFoundError(f"RSS feeds file not found at: {path}")

    specs: list[FeedSpec] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Prefer TSV, but allow comma-separated as a fallback.
        parts = [p.strip() for p in re.split(r"\t|,", line) if p.strip()]
        if len(parts) < 3:
            raise ValueError(
                f"Bad line in {path}: expected 3 columns (lane, site, url) but got: {raw!r}"
            )

        lane, site, url = parts[0], parts[1], parts[2]
        specs.append(FeedSpec(lane=lane, site=site, url=url))

    return specs


def get_or_create_source_id(con: sqlite3.Connection, lane: str, site: str, query: str) -> int:
    cols = table_columns(con, "sources")
    id_col = pick_col(cols, ["id"])
    lane_col = pick_col(cols, ["lane"])
    site_col = pick_col(cols, ["site"])
    query_col = pick_col(cols, ["query"])
    ts_col = pick_col(cols, ["collected_at_utc", "created_at_utc"])

    if not (id_col and lane_col and site_col and query_col):
        raise RuntimeError("sources table schema not recognized (missing id/lane/site/query).")

    row = con.execute(
        f"""
        SELECT {id_col}
        FROM sources
        WHERE {lane_col} = ? AND {site_col} = ? AND {query_col} = ?
        ORDER BY {id_col} DESC
        LIMIT 1
        """,
        (lane, site, query),
    ).fetchone()

    if row:
        return int(row[0])

    insert_cols = [lane_col, site_col, query_col]
    insert_vals = [lane, site, query]
    if ts_col:
        insert_cols.append(ts_col)
        insert_vals.append(utcnow_iso())

    q = f"""
    INSERT INTO sources({",".join(insert_cols)})
    VALUES ({",".join(["?"] * len(insert_cols))})
    """
    cur = con.execute(q, insert_vals)
    return int(cur.lastrowid)


def insert_raw_document(con: sqlite3.Connection, source_id: int, title: str, body: str, url: str) -> None:
    cols = table_columns(con, "raw_documents")

    id_col = pick_col(cols, ["id"])
    url_col = pick_col(cols, ["url"])
    title_col = pick_col(cols, ["title"])
    body_col = pick_col(cols, ["body", "text", "content"])
    source_col = pick_col(cols, ["source_id", "sourceId"])
    ts_col = pick_col(cols, ["collected_at_utc", "created_at_utc"])

    if not (id_col and url_col):
        raise RuntimeError("raw_documents table schema not recognized (missing id/url).")

    insert_cols = []
    insert_vals = []

    if source_col:
        insert_cols.append(source_col)
        insert_vals.append(int(source_id))
    if title_col:
        insert_cols.append(title_col)
        insert_vals.append(title)
    if body_col:
        insert_cols.append(body_col)
        insert_vals.append(body)
    insert_cols.append(url_col)
    insert_vals.append(url)

    if ts_col:
        insert_cols.append(ts_col)
        insert_vals.append(utcnow_iso())

    q = f"""
    INSERT OR IGNORE INTO raw_documents({",".join(insert_cols)})
    VALUES ({",".join(["?"] * len(insert_cols))})
    """
    con.execute(q, insert_vals)


def parse_entry(entry) -> tuple[str, str, str]:
    title = strip_html(getattr(entry, "title", "") or "")
    link = getattr(entry, "link", "") or ""
    summary = strip_html(getattr(entry, "summary", "") or getattr(entry, "description", "") or "")

    # Keep body small + “research-safe”: title/summary only.
    body = summary
    if len(body) > 2000:
        body = body[:2000] + "…"

    return title, body, link


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    feeds = load_feeds(FEEDS_PATH)

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys=ON;")

        inserted = 0
        for fs in feeds:
            source_id = get_or_create_source_id(con, fs.lane, fs.site, f"rss:{fs.url}")

            d = feedparser.parse(fs.url)
            entries = getattr(d, "entries", []) or []

            for e in entries:
                title, body, link = parse_entry(e)
                if not link:
                    continue
                if not title and not body:
                    continue

                insert_raw_document(con, source_id, title=title, body=body, url=link)
                inserted += 1

            con.commit()
            print(f"[RSS] {fs.lane} {fs.site} -> {len(entries)} entries (db inserts attempted)")

            # Light throttling to avoid hammering syndicated endpoints
            time.sleep(0.5)

        print(f"Done. Total inserts attempted: {inserted}")


if __name__ == "__main__":
    main()
