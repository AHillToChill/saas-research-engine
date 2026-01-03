from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import httpx

DB_PATH = Path("data/db/research.sqlite")
API_BASE = "https://api.stackexchange.com/2.3"

PAGES = 2
PAGESIZE = 50


@dataclass(frozen=True)
class SESpec:
    lane: str
    site: str  # "stackoverflow" or full domain like "or.stackexchange.com"
    mode: Literal["questions_by_tag", "search_advanced"]
    tagged: str | None = None     # semicolon-separated tags
    q: str | None = None          # free text query for search_advanced


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def se_questions_by_tag(
    client: httpx.Client,
    site: str,
    tagged: str,
    pages: int = PAGES,
    pagesize: int = PAGESIZE,
) -> list[dict]:
    results: list[dict] = []
    for page in range(1, pages + 1):
        params = {
            "site": site,
            "tagged": tagged,
            "pagesize": pagesize,
            "page": page,
            "order": "desc",
            "sort": "creation",
            "filter": "withbody",
        }
        r = client.get(f"{API_BASE}/questions", params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        results.extend(payload.get("items", []))
        if not payload.get("has_more"):
            break
    return results


def se_search_advanced(
    client: httpx.Client,
    site: str,
    q: str,
    tagged: str | None,
    pages: int = PAGES,
    pagesize: int = PAGESIZE,
) -> list[dict]:
    results: list[dict] = []
    for page in range(1, pages + 1):
        params = {
            "site": site,
            "q": q,
            "pagesize": pagesize,
            "page": page,
            "order": "desc",
            "sort": "creation",
            "filter": "withbody",
        }
        if tagged:
            params["tagged"] = tagged

        r = client.get(f"{API_BASE}/search/advanced", params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        results.extend(payload.get("items", []))
        if not payload.get("has_more"):
            break
    return results


def insert_source(con: sqlite3.Connection, spec: SESpec) -> int:
    query_repr = ""
    if spec.mode == "questions_by_tag":
        query_repr = f"tagged:{spec.tagged}"
    else:
        query_repr = f"q:{spec.q}" + (f" tagged:{spec.tagged}" if spec.tagged else "")

    cur = con.execute(
        "INSERT INTO sources(lane, source_type, site, query, collected_at_utc) VALUES (?, ?, ?, ?, ?)",
        (spec.lane, "stackexchange", spec.site, query_repr, utcnow_iso()),
    )
    return int(cur.lastrowid)


def upsert_raw_document(con: sqlite3.Connection, source_id: int, item: dict) -> None:
    external_id = str(item.get("question_id"))
    url = item.get("link")
    title = item.get("title")
    body = item.get("body")
    created = item.get("creation_date")
    created_iso = datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else None
    tags = ",".join(item.get("tags", []))
    score = item.get("score")
    num_answers = item.get("answer_count")

    con.execute(
        """
        INSERT OR IGNORE INTO raw_documents
        (source_id, external_id, url, title, body_text, created_at_utc, score, num_answers, tags, raw_json, ingested_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            external_id,
            url,
            title,
            body,
            created_iso,
            score,
            num_answers,
            tags,
            json.dumps(item),
            utcnow_iso(),
        ),
    )


def run(specs: Iterable[SESpec]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as con, httpx.Client(headers={"User-Agent": "saas-research-engine/0.1"}) as client:
        con.execute("PRAGMA foreign_keys=ON;")

        for spec in specs:
            source_id = insert_source(con, spec)

            if spec.mode == "questions_by_tag":
                assert spec.tagged
                items = se_questions_by_tag(client, spec.site, spec.tagged)
                print(f"[{spec.lane}] {spec.site} tagged='{spec.tagged}' -> {len(items)} items")
            else:
                assert spec.q
                items = se_search_advanced(client, spec.site, spec.q, spec.tagged)
                print(f"[{spec.lane}] {spec.site} q='{spec.q}' tagged='{spec.tagged}' -> {len(items)} items")

            for item in items:
                upsert_raw_document(con, source_id, item)

            con.commit()


if __name__ == "__main__":
    specs = [
        # ----------------------------
        # Lane A: Simulation / Digital Twin workflow
        # ----------------------------
        SESpec("A", "stackoverflow", "questions_by_tag", tagged="simpy"),
        SESpec("A", "stackoverflow", "questions_by_tag", tagged="anylogic"),
        SESpec("A", "stackoverflow", "search_advanced", tagged="simulation", q="parameter sweep"),
        SESpec("A", "stackoverflow", "search_advanced", tagged="simulation", q="performance slow"),
        SESpec("A", "cs.stackexchange.com", "search_advanced", q="discrete event simulation"),

        # ----------------------------
        # Lane B: KPI / Reporting automation
        # ----------------------------
        SESpec("B", "stackoverflow", "questions_by_tag", tagged="excel"),
        SESpec("B", "stackoverflow", "questions_by_tag", tagged="powerbi"),
        SESpec("B", "stackoverflow", "search_advanced", tagged="excel", q="automate report"),
        SESpec("B", "stackoverflow", "search_advanced", q="SSIS automated reporting"),

        # ----------------------------
        # Lane C: Routing optimization (VRP)
        # ----------------------------
        SESpec("C", "or.stackexchange.com", "questions_by_tag", tagged="vehicle-routing"),
        SESpec("C", "stackoverflow", "questions_by_tag", tagged="vehicle-routing"),
        SESpec("C", "or.stackexchange.com", "search_advanced", q="time windows vehicle routing"),
    ]
    run(specs)
