from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup

DB_PATH = Path("data/db/research.sqlite")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    # Unescape entities first (&amp;, &#39;, etc.), then strip HTML.
    html = unescape(html)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_purchase_intent(text: str) -> int:
    t = text.lower()
    signals = [
        "best way", "recommend", "tool", "software", "library", "framework",
        "alternative", "which is better", "ss is", "ssrs", "ssis",
        "paid", "pricing", "subscription", "service",
    ]
    score = sum(1 for s in signals if s in t)
    # Map to 0..3
    if score == 0:
        return 0
    if score == 1:
        return 1
    if score == 2:
        return 2
    return 3


def guess_severity(text: str) -> int:
    t = text.lower()
    high = ["crash", "fails", "broken", "urgent", "production", "error", "timeout", "cannot", "can't"]
    mid = ["slow", "performance", "stuck", "problem", "issue", "hard", "difficult"]
    low = ["how do i", "example", "tutorial"]

    if any(w in t for w in high):
        return 4
    if any(w in t for w in mid):
        return 3
    if any(w in t for w in low):
        return 2
    return 3


def guess_stage(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["install", "setup", "configure", "onboarding", "getting started"]):
        return "onboarding"
    if any(w in t for w in ["slow", "performance", "scale", "large", "many", "batch"]):
        return "scaling"
    if any(w in t for w in ["edge case", "corner case", "sometimes", "intermittent"]):
        return "edge-case"
    return "daily"


def build_normalized_pain(title: str, body_text: str) -> tuple[str, str, str, str]:
    """
    Returns (actor, task, friction, consequence) with actor often unknown initially.
    """
    actor = "unknown"

    # Use title as the primary "task/friction" signal.
    task = title.strip()

    # Friction: first ~2 paragraphs of the question body.
    paras = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    friction = paras[0] if len(paras) >= 1 else ""
    if len(paras) >= 2:
        friction = friction + " " + paras[1]

    # Consequence: infer from keywords (very rough v1).
    t = (title + " " + friction).lower()
    if any(w in t for w in ["slow", "performance", "timeout"]):
        consequence = "Performance delays / wasted time"
    elif any(w in t for w in ["error", "fails", "broken", "crash"]):
        consequence = "Failures / instability / rework"
    elif any(w in t for w in ["automate", "manual", "repeat", "report"]):
        consequence = "Manual effort / recurring time cost"
    else:
        consequence = "Workflow friction"

    return actor, task[:500], friction[:1200], consequence


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys=ON;")

        # Pull raw docs that do NOT yet have a pain_unit.
        rows = con.execute(
            """
            SELECT rd.id, s.lane, rd.title, rd.body_text, rd.url
            FROM raw_documents rd
            JOIN sources s ON s.id = rd.source_id
            LEFT JOIN pain_units pu ON pu.raw_document_id = rd.id
            WHERE pu.id IS NULL
            ORDER BY rd.id
            """
        ).fetchall()

        print(f"Raw documents pending normalization: {len(rows)}")

        inserted = 0
        for raw_id, lane, title, body_html, url in rows:
            title_clean = unescape(title or "").strip()
            body_clean = html_to_text(body_html or "")

            # If there's effectively no content, skip (rare, but happens).
            if not title_clean and not body_clean:
                continue

            actor, task, friction, consequence = build_normalized_pain(title_clean, body_clean)

            normalized_pain = f"As a {actor}, I struggle with: {task} because {consequence}."
            workaround = "unknown"

            severity = guess_severity(title_clean + " " + body_clean)
            frequency = 3  # heuristic baseline for now; improved later via clustering counts
            purchase_intent = guess_purchase_intent(title_clean + " " + body_clean)
            stage = guess_stage(title_clean + " " + body_clean)

            con.execute(
                """
                INSERT INTO pain_units
                (raw_document_id, lane, actor, task, friction, consequence, normalized_pain, workaround,
                 severity, frequency, purchase_intent, stage, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw_id, lane, actor, task, friction, consequence, normalized_pain, workaround,
                    severity, frequency, purchase_intent, stage, utcnow_iso()
                ),
            )
            inserted += 1

        con.commit()
        print(f"Inserted pain_units: {inserted}")


if __name__ == "__main__":
    main()
