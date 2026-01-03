from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

DB_PATH = Path("data/db/research.sqlite")
OUT_DIR = Path("artifacts/reports")

# Bump the method name so you can compare old vs refined runs in the DB/reports.
METHOD = "tfidf_kmeans_v2_refined"

STOPWORDS_PATH = Path("src/cluster/stopwords_global.txt")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_text(s: str) -> str:
    """
    Lightweight cleanup:
    - removes very code-like or stack-trace-ish lines
    - removes long log/code lines
    - normalizes whitespace
    """
    if not s:
        return ""

    lines: list[str] = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue

        if any(tok in t for tok in ["Traceback", "Exception", "public static", "SELECT ", "INSERT ", "<div", "</", "{", "}", ";;", "=>"]):
            continue

        if len(t) > 250:
            continue

        lines.append(t)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


LANE_STOPWORDS: dict[str, set[str]] = {
    # Lane A: simulation / digital-twin workflow
    "A": {
        "simpy", "anylogic", "simulation", "simulate", "simulations", "process", "event",
        "model", "models", "discrete", "des", "agent", "agents",
    },
    # Lane B: reporting automation
    "B": {
        "excel", "power", "bi", "powerbi", "vba", "worksheet", "workbook", "pivot", "pivottable",
        "ssrs", "ssis", "ssms", "sql", "server", "table", "query",
    },
    # Lane C: routing optimization
    "C": {
        "vrp", "vrptw", "routing", "route", "routes", "vehicle", "vehicles",
        "ortools", "or-tools", "gurobi", "solver", "solvers", "constraint", "constraints",
        "time", "windows", "window",
    },
}


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


def top_terms_for_cluster(tfidf, labels, cluster_idx: int, vectorizer: TfidfVectorizer, top_n: int = 10) -> list[str]:
    """
    centroid-like terms: average tf-idf per term within the cluster.
    """
    import numpy as np

    idx = np.where(labels == cluster_idx)[0]
    if len(idx) == 0:
        return []

    sub = tfidf[idx]
    mean_vec = sub.mean(axis=0)

    if hasattr(mean_vec, "A1"):
        mean_vec = mean_vec.A1
    else:
        mean_vec = mean_vec.ravel()

    terms = vectorizer.get_feature_names_out()
    top_ids = mean_vec.argsort()[-top_n:][::-1]
    return [terms[i] for i in top_ids if mean_vec[i] > 0]


def compute_score(size: int, mean_sev: float, mean_pi: float) -> float:
    """
    Composite ranking score: prefer larger clusters with higher severity and purchase intent.
    log1p(size) dampens huge clusters.
    """
    return math.log1p(size) * (mean_sev + (0.75 * mean_pi))


def fetch_pain_units(con: sqlite3.Connection, lane: str) -> list[PU]:
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
        WHERE pu.lane = ?
        """,
        (lane,),
    ).fetchall()

    return [
        PU(
            id=int(r[0]),
            lane=r[1],
            task=r[2] or "",
            friction=r[3] or "",
            consequence=r[4] or "",
            severity=int(r[5] or 3),
            purchase_intent=int(r[6] or 0),
            url=r[7] or "",
        )
        for r in rows
    ]


def reset_method(con: sqlite3.Connection) -> None:
    cluster_ids = [r[0] for r in con.execute("SELECT id FROM clusters WHERE method = ?", (METHOD,)).fetchall()]
    if cluster_ids:
        con.executemany("DELETE FROM cluster_members WHERE cluster_id = ?", [(cid,) for cid in cluster_ids])
        con.execute("DELETE FROM clusters WHERE method = ?", (METHOD,))
        con.commit()


def insert_cluster(con: sqlite3.Connection, lane: str, label: str, summary: str, score: float) -> int:
    cur = con.execute(
        """
        INSERT INTO clusters(lane, method, label, summary, score, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (lane, METHOD, label, summary, float(score), utcnow_iso()),
    )
    return int(cur.lastrowid)


def insert_members(con: sqlite3.Connection, cluster_id: int, pain_unit_ids: Iterable[int]) -> None:
    con.executemany(
        "INSERT OR IGNORE INTO cluster_members(cluster_id, pain_unit_id) VALUES (?, ?)",
        [(cluster_id, pid) for pid in pain_unit_ids],
    )


def choose_k_by_silhouette(X, candidate_ks: list[int]) -> tuple[int, float]:
    """
    Picks k using silhouette score.
    Returns (best_k, best_silhouette).
    """
    n = X.shape[0]
    valid_ks = sorted({k for k in candidate_ks if 2 <= k < n})
    if not valid_ks:
        return 2, -1.0

    best_k = valid_ks[0]
    best_score = -1.0

    for k in valid_ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)

        # silhouette_score can be expensive at large n; you're currently fine at ~200-400/lane.
        s = silhouette_score(X, labels, metric="cosine")

        if s > best_score:
            best_score = s
            best_k = k

    return best_k, float(best_score)


def write_report(con: sqlite3.Connection, stamp: str, lane_meta: dict[str, dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"clusters_{METHOD}_{stamp}.md"

    clusters = con.execute(
        """
        SELECT id, lane, label, summary, score
        FROM clusters
        WHERE method = ?
        ORDER BY lane, score DESC
        """,
        (METHOD,),
    ).fetchall()

    def fetch_examples(cluster_id: int, n: int = 6):
        return con.execute(
            """
            SELECT pu.task, pu.friction, rd.url, pu.severity, pu.purchase_intent
            FROM cluster_members cm
            JOIN pain_units pu ON pu.id = cm.pain_unit_id
            JOIN raw_documents rd ON rd.id = pu.raw_document_id
            WHERE cm.cluster_id = ?
            ORDER BY pu.purchase_intent DESC, pu.severity DESC
            LIMIT ?
            """,
            (cluster_id, n),
        ).fetchall()

    lines: list[str] = []
    lines.append(f"# Cluster Report ({METHOD})")
    lines.append("")
    lines.append(f"- Generated: {utcnow_iso()}")
    lines.append(f"- Database: `{DB_PATH.as_posix()}`")
    lines.append(f"- Stopwords file: `{STOPWORDS_PATH.as_posix()}`")
    lines.append("")

    for lane in ["A", "B", "C"]:
        meta = lane_meta.get(lane, {})
        lines.append(f"## Lane {lane} — Ranked clusters")
        lines.append("")
        if meta:
            lines.append(f"- Documents clustered: {meta.get('n_docs')}")
            lines.append(f"- k selected: {meta.get('k')}")
            lines.append(f"- silhouette (cosine): {meta.get('silhouette'):.4f}")
            lines.append("")

        lane_clusters = [c for c in clusters if c[1] == lane]
        if not lane_clusters:
            lines.append("_No clusters found for this lane._")
            lines.append("")
            continue

        lines.append("| Label | Cluster ID | Size | Mean Sev | Mean Purchase | Score | Keywords |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")

        for cid, _lane, label, summary, score in lane_clusters:
            stats = con.execute(
                """
                SELECT COUNT(*), AVG(pu.severity), AVG(pu.purchase_intent)
                FROM cluster_members cm
                JOIN pain_units pu ON pu.id = cm.pain_unit_id
                WHERE cm.cluster_id = ?
                """,
                (cid,),
            ).fetchone()

            size = int(stats[0] or 0)
            mean_sev = float(stats[1] or 0.0)
            mean_pi = float(stats[2] or 0.0)
            lines.append(
                f"| {label} | {cid} | {size} | {mean_sev:.2f} | {mean_pi:.2f} | {float(score):.2f} | {summary} |"
            )

        lines.append("")
        lines.append(f"## Lane {lane} — Examples (top clusters)")
        lines.append("")

        for cid, _lane, label, summary, score in lane_clusters[:5]:
            lines.append(f"### {label} (Cluster {cid})")
            lines.append(f"- Score: {float(score):.2f}")
            lines.append(f"- Keywords: {summary}")
            lines.append("")
            ex = fetch_examples(cid, n=6)
            for task, friction, url, sev, pi in ex:
                task_s = (task or "").strip()
                fr_s = clean_text(friction or "")
                if len(fr_s) > 220:
                    fr_s = fr_s[:220] + "…"
                lines.append(f"- (sev={sev}, buy={pi}) {task_s}")
                if fr_s:
                    lines.append(f"  - Snippet: {fr_s}")
                lines.append(f"  - Source: {url}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    global_stop = load_stopwords(STOPWORDS_PATH)

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys=ON;")
        reset_method(con)

        stamp = now_stamp()
        lane_meta: dict[str, dict] = {}

        for lane in ["A", "B", "C"]:
            pus = fetch_pain_units(con, lane)
            if len(pus) < 20:
                print(f"[{lane}] Not enough pain_units to cluster (n={len(pus)})")
                continue

            docs: list[str] = []
            ids: list[int] = []
            for pu in pus:
                # Key refinement: do NOT cluster on normalized_pain (it contains our boilerplate).
                text = " ".join(
                    [
                        clean_text(pu.task),
                        clean_text(pu.friction),
                        clean_text(pu.consequence),
                    ]
                ).strip()
                if text:
                    docs.append(text)
                    ids.append(pu.id)

            if len(docs) < 20:
                print(f"[{lane}] Not enough usable text after cleaning (n={len(docs)})")
                continue

            lane_stop = global_stop.union(LANE_STOPWORDS.get(lane, set())).union(set(ENGLISH_STOP_WORDS))

            vectorizer = TfidfVectorizer(
                stop_words=list(lane_stop),
                min_df=3,
                max_df=0.90,
                ngram_range=(1, 2),
                max_features=7000,
            )
            X = vectorizer.fit_transform(docs)

            # Candidate k sweep (statistical selection)
            candidate_ks = [8, 10, 12, 15, 18, 22, 26, 30]
            best_k, best_sil = choose_k_by_silhouette(X, candidate_ks)

            print(f"[{lane}] Clustering n={len(docs)} into k={best_k} (silhouette={best_sil:.4f}) ...")

            km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)

            # Insert clusters
            for cidx in range(best_k):
                member_ids = [ids[i] for i, lab in enumerate(labels) if lab == cidx]
                if not member_ids:
                    continue

                q = f"""
                SELECT COUNT(*), AVG(severity), AVG(purchase_intent)
                FROM pain_units
                WHERE id IN ({",".join(["?"] * len(member_ids))})
                """
                size, mean_sev, mean_pi = con.execute(q, member_ids).fetchone()
                size = int(size or 0)
                mean_sev = float(mean_sev or 0.0)
                mean_pi = float(mean_pi or 0.0)

                terms = top_terms_for_cluster(X, labels, cidx, vectorizer, top_n=10)
                summary = ", ".join(terms[:8]) if terms else "n/a"

                score = compute_score(size, mean_sev, mean_pi)
                label = f"{lane}-{cidx:02d}"

                cluster_id = insert_cluster(con, lane, label, summary, score)
                insert_members(con, cluster_id, member_ids)

            con.commit()
            lane_meta[lane] = {"n_docs": len(docs), "k": best_k, "silhouette": best_sil}
            print(f"[{lane}] Inserted clusters for method={METHOD}")

        report_path = write_report(con, stamp, lane_meta)
        print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
