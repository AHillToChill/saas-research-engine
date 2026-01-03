from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
import os

from src.collectors._db import DB_PATH, ensure_source, insert_raw_document


REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"


@dataclass(frozen=True)
class Job:
    lane: str
    subreddit: str
    query: str
    sort: str  # new | top | relevance
    t: str     # hour | day | week | month | year | all
    pages: int


def get_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def oauth_token(client: httpx.Client, client_id: str, client_secret: str, username: str, password: str, user_agent: str) -> str:
    r = client.post(
        REDDIT_TOKEN_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "password", "username": username, "password": password},
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    tok = j.get("access_token")
    if not tok:
        raise RuntimeError(f"Token response missing access_token: {j}")
    return str(tok)


def polite_sleep_from_headers(resp: httpx.Response) -> None:
    # If present, respect rate-limit headers. Otherwise, light throttle.
    remaining = resp.headers.get("x-ratelimit-remaining")
    reset = resp.headers.get("x-ratelimit-reset")
    if remaining is None or reset is None:
        time.sleep(0.8)
        return
    try:
        rem = float(remaining)
        rst = float(reset)
        if rem < 5:
            time.sleep(min(5.0, max(1.0, rst)))
        else:
            time.sleep(0.3)
    except Exception:
        time.sleep(0.8)


def extract_text(post: dict[str, Any]) -> str:
    title = (post.get("title") or "").strip()
    selftext = (post.get("selftext") or "").strip()
    parts = []
    if title:
        parts.append(title)
    if selftext:
        parts.append(selftext)
    return "\n\n".join(parts).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force_lane", default="", help="If set, override lane for all jobs")
    ap.add_argument("--max_per_page", type=int, default=100, help="Reddit API max is typically 100")
    args = ap.parse_args()

    load_dotenv()

    client_id = get_env("REDDIT_CLIENT_ID")
    client_secret = get_env("REDDIT_CLIENT_SECRET")
    username = get_env("REDDIT_USERNAME")
    password = get_env("REDDIT_PASSWORD")
    user_agent = get_env("REDDIT_USER_AGENT")  # must be descriptive/unique

    # Default buyer-heavy targets + your lanes.
    jobs = [
        Job("A", "simulation", "experiment management OR parameter sweep tool", "new", "year", 3),
        Job("A", "datascience", "digital twin OR discrete event simulation reporting", "new", "year", 3),

        Job("B", "smallbusiness", "reporting automation OR dashboard refresh OR spreadsheet workflow", "new", "year", 4),
        Job("B", "entrepreneur", "monthly reporting automation OR KPI dashboard tool", "new", "year", 4),

        Job("C", "logistics", "route optimization OR dispatch software OR delivery scheduling", "new", "year", 4),
        Job("C", "trucking", "dispatch software OR routing tool OR time windows", "new", "year", 4),
    ]

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    with httpx.Client() as hc:
        token = oauth_token(hc, client_id, client_secret, username, password, user_agent)

        headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}

        with sqlite3.connect(DB_PATH) as con:
            con.execute("PRAGMA foreign_keys=ON;")

            for job in jobs:
                lane = args.force_lane.strip() or job.lane
                site = "reddit.com"
                query = f"r/{job.subreddit} q={job.query} sort={job.sort} t={job.t}"
                source_id = ensure_source(con, lane, site, query)

                after = None
                total = 0

                for _page in range(job.pages):
                    params = {
                        "q": job.query,
                        "restrict_sr": "1",
                        "sort": job.sort,
                        "t": job.t,
                        "limit": str(args.max_per_page),
                    }
                    if after:
                        params["after"] = after

                    url = f"{REDDIT_API_BASE}/r/{job.subreddit}/search"
                    resp = hc.get(url, headers=headers, params=params, timeout=30)
                    if resp.status_code == 429:
                        time.sleep(5)
                        continue
                    resp.raise_for_status()

                    data = resp.json()
                    children = (data.get("data") or {}).get("children") or []
                    after = (data.get("data") or {}).get("after")

                    inserted = 0
                    for ch in children:
                        post = (ch.get("data") or {})
                        permalink = post.get("permalink")
                        if not permalink:
                            continue

                        doc_url = f"https://www.reddit.com{permalink}"
                        title = (post.get("title") or "").strip()
                        text = extract_text(post)
                        if len(text) < 30:
                            continue

                        insert_raw_document(
                            con=con,
                            source_id=source_id,
                            url=doc_url,
                            title=title or doc_url,
                            text=text,
                            extra={
                                "subreddit": job.subreddit,
                                "score": post.get("score"),
                                "num_comments": post.get("num_comments"),
                                "created_utc": post.get("created_utc"),
                            },
                        )
                        inserted += 1

                    con.commit()
                    total += inserted
                    print(f"[REDDIT] lane={lane} r/{job.subreddit} inserted={inserted} after={after}")

                    polite_sleep_from_headers(resp)
                    if not after:
                        break

                print(f"[REDDIT] DONE lane={lane} r/{job.subreddit} total_inserted={total}")

    print("Done.")


if __name__ == "__main__":
    main()
