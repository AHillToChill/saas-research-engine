from __future__ import annotations

import argparse
import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.collectors._db import DB_PATH, ensure_source, insert_raw_document


HN_BASE = "https://hacker-news.firebaseio.com/v0"


def html_to_text(s: str) -> str:
    if not s:
        return ""
    soup = BeautifulSoup(s, "html.parser")
    txt = soup.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


async def fetch_json(client: httpx.AsyncClient, url: str) -> Any:
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


async def fetch_items(client: httpx.AsyncClient, ids: list[int], concurrency: int = 20) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    out: list[dict[str, Any]] = []

    async def one(item_id: int) -> None:
        async with sem:
            try:
                j = await fetch_json(client, f"{HN_BASE}/item/{item_id}.json")
                if isinstance(j, dict):
                    out.append(j)
            except Exception:
                return

    await asyncio.gather(*[one(i) for i in ids])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="D", help="Lane label to store in sources (default: D)")
    ap.add_argument("--endpoints", default="askstories,showstories", help="Comma list: topstories,newstories,askstories,showstories,jobstories")
    ap.add_argument("--per_endpoint", type=int, default=300, help="How many IDs to ingest per endpoint")
    ap.add_argument("--concurrency", type=int, default=20, help="Polite concurrency for item fetches")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if not endpoints:
        raise ValueError("No endpoints provided.")

    async def run() -> None:
        async with httpx.AsyncClient(headers={"User-Agent": "saas-research-engine/0.1 (HN collector)"}) as client:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("PRAGMA foreign_keys=ON;")

                for ep in endpoints:
                    ids = await fetch_json(client, f"{HN_BASE}/{ep}.json")
                    if not isinstance(ids, list):
                        continue
                    ids = [int(x) for x in ids[: args.per_endpoint] if isinstance(x, int)]
                    if not ids:
                        continue

                    site = "hacker-news.firebaseio.com"
                    query = f"endpoint:{ep}"
                    source_id = ensure_source(con, args.lane, site, query)

                    items = await fetch_items(client, ids, concurrency=args.concurrency)

                    inserted = 0
                    for it in items:
                        if it.get("type") != "story":
                            continue

                        item_id = it.get("id")
                        if not item_id:
                            continue

                        hn_url = f"https://news.ycombinator.com/item?id={item_id}"
                        title = (it.get("title") or "").strip()

                        body_parts = []
                        if title:
                            body_parts.append(title)

                        text = html_to_text(it.get("text") or "")
                        if text:
                            body_parts.append(text)

                        ext = (it.get("url") or "").strip()
                        if ext:
                            body_parts.append(f"external_url: {ext}")

                        body = "\n\n".join(body_parts).strip()
                        if not body:
                            continue

                        insert_raw_document(
                            con=con,
                            source_id=source_id,
                            url=hn_url,
                            title=title or hn_url,
                            text=body,
                            extra={"hn_endpoint": ep, "hn_item_id": item_id},
                        )
                        inserted += 1

                    con.commit()
                    print(f"[HN] {ep}: fetched={len(ids)} stories_inserted={inserted}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
