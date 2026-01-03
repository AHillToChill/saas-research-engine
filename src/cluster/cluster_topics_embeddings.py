from __future__ import annotations

import argparse
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

DB_PATH = Path("data/db/research.sqlite")
OUT_DIR = Path("artifacts/reports")

METHOD = "emb_kmeans_v1"  # stored in clusters.method


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_text(s: str) -> str:
    if not s:
        return ""
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


def load_stopwords_txt(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


GLOBAL_STOP_PATH = Path("src/cluster/stopwords_global.txt")


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
    # Assumes init_db created: clusters, cluster_members.
    # embed_pain_units.py creates pain_unit_embeddings.
    con.commit()


def reset_method(con: sqlite3.Connection, method: str) -> None:
    cluster_ids = [r[0] for r in con.execute("SELECT id FROM clusters WHERE method = ?", (method,)).fetchall()]
    if cluster_ids:
        con.executemany("DELETE FROM cluster_members WHERE cluster_id = ?", [(cid,) for cid in cluster_ids])
        con.execute("DELETE FROM clusters WHERE method = ?", (method,))
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


def fetch_embeddings(con: sqlite3.Connection, model: str) -> dict[int, tuple[int, bytes]]:
    rows = con.execute(
        """
        SELECT pain_unit_id, dim, embedding
        FROM pain_unit_embeddings
        WHERE model = ?
        """,
        (model,),
    ).fetchall()
    return {int(r[0]): (int(r[1]), r[2]) for r in rows}


def build_doc(pu: PU) -> str:
    parts = [
        clean_text(pu.task),
        clean_text(pu.friction),
        clean_text(pu.consequence),
    ]
    return " ".join([p for p in parts if p]).strip()


def auto_stopwords_from_df(vectorizer: TfidfVectorizer, X, df_frac_threshold: float = 0.60) -> set[str]:
    # Terms appearing in >60% of docs are likely glue terms in this corpus.
    df = (X > 0).sum(axis=0)
    if hasattr(df, "A1"):
        df = df.A1
    else:
        df = np.asarray(df).ravel()
    frac = df / X.shape[0]
    terms = vectorizer.get_feature_names_out()
    return {terms[i] for i in range(len(terms)) if frac[i] >= df_frac_threshold}


def build_tfidf(docs: list[str], global_stop: set[str]) -> tuple[TfidfVectorizer, any]:
    base_stop = set(ENGLISH_STOP_WORDS).union(global_stop)

    v1 = TfidfVectorizer(
        stop_words=list(base_stop),
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        max_features=20000,
    )
    X1 = v1.fit_transform(docs)
    hi_df = auto_stopwords_from_df(v1, X1, df_frac_threshold=0.60)

    stop2 = base_stop.union({w.lower() for w in hi_df})
    v2 = TfidfVectorizer(
        stop_words=list(stop2),
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        max_features=20000,
    )
    X2 = v2.fit_transform(docs)
    return v2, X2


def top_terms_for_cluster(X_tfidf, labels: np.ndarray, cluster_idx: int, vectorizer: TfidfVectorizer, top_n: int = 10) -> list[str]:
    idx = np.where(labels == cluster_idx)[0]
    if len(idx) == 0:
        return []
    sub = X_tfidf[idx]
    mean_vec = sub.mean(axis=0)
    if hasattr(mean_vec, "A1"):
        mean_vec = mean_vec.A1
    else:
        mean_vec = np.asarray(mean_vec).ravel()
    terms = vectorizer.get_feature_names_out()
    top_ids = mean_vec.argsort()[-top_n:][::-1]
    return [terms[i] for i in top_ids if mean_vec[i] > 0]


def compute_opportunity_score(size: int, mean_sev: float, mean_pi: float, tool_rate: float, buyer_rate: float) -> float:
    base = math.log1p(size) * (mean_sev + 0.75 * mean_pi)
    base *= (1.0 + 0.25 * tool_rate)
    base *= (1.0 + 0.10 * buyer_rate)
    return float(base)


TOOL_WORDS = {
    "tool", "tools", "software", "app", "service", "platform",
    "automate", "automation", "schedule", "scheduler", "monitor", "monitoring",
    "dashboard", "reporting", "pipeline", "integration", "api",
    "subscription", "pricing", "license", "deploy", "deployment",
}

BUYER_WORDS = {
    "business", "company", "customer", "clients", "client", "team",
    "operations", "ops", "production", "workflow", "manager", "finance",
    "sales", "support", "deliver", "delivery", "dispatch",
}


def rate_contains(docs: list[str], wordset: set[str]) -> float:
    if not docs:
        return 0.0
    hits = 0
    for d in docs:
        t = d.lower()
        if any(w in t for w in wordset):
            hits += 1
    return hits / len(docs)


def insert_cluster(con: sqlite3.Connection, lane: str, method: str, label: str, summary: str, score: float) -> int:
    cur = con.execute(
        """
        INSERT INTO clusters(lane, method, label, summary, score, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (lane, method, label, summary, float(score), utcnow_iso()),
    )
    return int(cur.lastrowid)


def insert_members(con: sqlite3.Connection, cluster_id: int, pain_unit_ids: Iterable[int]) -> None:
    con.executemany(
        "INSERT OR IGNORE INTO cluster_members(cluster_id, pain_unit_id) VALUES (?, ?)",
        [(cluster_id, pid) for pid in pain_unit_ids],
    )


def pick_k_embeddings(X: np.ndarray, candidate_ks: list[int]) -> tuple[int, float]:
    n = X.shape[0]
    valid = [k for k in sorted(set(candidate_ks)) if 2 <= k < n]
    if not valid:
        return 10, -1.0

    best_k = valid[0]
    best_s = -1.0

    sample_size = min(2000, n)

    for k in valid:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, metric="cosine", sample_size=sample_size, random_state=42)
        if s > best_s:
            best_s = float(s)
            best_k = int(k)

    return best_k, best_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model to use (must exist in DB)")
    ap.add_argument("--min_chars", type=int, default=30, help="Drop docs shorter than this after cleaning")
    ap.add_argument("--force", action="store_true", help="Overwrite prior clusters for this METHOD")
    ap.add_argument("--noise_quantile", type=float, default=0.05, help="Move lowest-confidence fraction to a NOISE cluster")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys=ON;")
        ensure_tables(con)

        method = f"{METHOD}::{args.model}"

        if args.force:
            reset_method(con, method)

        pus = fetch_pain_units(con)
        emb_map = fetch_embeddings(con, args.model)

        ids: list[int] = []
        lanes: list[str] = []
        docs: list[str] = []
        sev: list[int] = []
        pi: list[int] = []
        urls: list[str] = []
        embs: list[np.ndarray] = []

        for pu in pus:
            if pu.id not in emb_map:
                continue
            doc = build_doc(pu)
            if len(doc) < args.min_chars:
                continue

            dim, blob = emb_map[pu.id]
            v = np.frombuffer(blob, dtype=np.float32)
            if v.size != dim:
                continue

            ids.append(pu.id)
            lanes.append(pu.lane or "")
            docs.append(doc)
            sev.append(pu.severity)
            pi.append(pu.purchase_intent)
            urls.append(pu.url)
            embs.append(v)

        if len(ids) < 50:
            raise RuntimeError(f"Not enough embedded docs to cluster (n={len(ids)}). Run embed_pain_units.py first.")

        X = np.vstack(embs).astype(np.float32)

        n = X.shape[0]
        base = int(max(12, min(80, round(math.sqrt(n) * 1.8))))
        candidate_ks = sorted({
            max(8, base - 20),
            max(10, base - 10),
            base,
            min(80, base + 10),
            min(80, base + 20),
            15, 20, 25, 30, 40, 50, 60
        })
        candidate_ks = [k for k in candidate_ks if 2 <= k < n]

        best_k, best_sil = pick_k_embeddings(X, candidate_ks)
        print(f"Clustering embeddings: n={n}, selected k={best_k}, silhouette(cosine)={best_sil:.4f}")

        km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)

        # Confidence: cosine similarity to assigned centroid (embeddings are normalized)
        centroids = km.cluster_centers_.astype(np.float32)
        cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        sims = np.sum(X * cent_norm[labels], axis=1)

        noise_thresh = float(np.quantile(sims, args.noise_quantile))
        noise_mask = sims <= noise_thresh
        n_noise = int(noise_mask.sum())
        print(f"Noise assignment: {n_noise}/{n} moved to NOISE (threshold={noise_thresh:.4f})")

        # Create adjusted labels once (this also fixes your prior SyntaxError line cleanly)
        labels_adj = np.where(noise_mask, -1, labels).astype(int)

        # Build TF-IDF for keyword summaries (automatic stopwords expansion, global)
        global_stop = load_stopwords_txt(GLOBAL_STOP_PATH)
        vectorizer, X_tfidf = build_tfidf(docs, global_stop)

        stamp = now_stamp()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUT_DIR / f"topics_{method.replace('::','__')}_{stamp}.md"

        cluster_db_ids: dict[int, int] = {}
        cluster_members: dict[int, list[int]] = {}

        for i, pid in enumerate(ids):
            c = int(labels_adj[i])
            cluster_members.setdefault(c, []).append(pid)

        cluster_stats_rows = []

        for cidx, member_ids in cluster_members.items():
            if not member_ids:
                continue

            member_set = set(member_ids)
            member_idx = [i for i, pid in enumerate(ids) if pid in member_set]

            size = len(member_idx)
            mean_sev = float(np.mean([sev[i] for i in member_idx])) if member_idx else 0.0
            mean_pi = float(np.mean([pi[i] for i in member_idx])) if member_idx else 0.0

            member_docs = [docs[i] for i in member_idx]
            tool_rate = rate_contains(member_docs, TOOL_WORDS)
            buyer_rate = rate_contains(member_docs, BUYER_WORDS)

            score = compute_opportunity_score(size, mean_sev, mean_pi, tool_rate, buyer_rate)

            if cidx == -1:
                label = "T-NOISE"
                keywords = "noise, outliers, mixed"
            else:
                label = f"T-{cidx:03d}"
                terms = top_terms_for_cluster(X_tfidf, labels_adj, cidx, vectorizer, top_n=10)
                keywords = ", ".join(terms[:8]) if terms else "n/a"

            db_cluster_id = insert_cluster(con, lane="ALL", method=method, label=label, summary=keywords, score=score)
            insert_members(con, db_cluster_id, member_ids)
            cluster_db_ids[cidx] = db_cluster_id

            lane_counts: dict[str, int] = {}
            for i in member_idx:
                lane_counts[lanes[i]] = lane_counts.get(lanes[i], 0) + 1

            cluster_stats_rows.append(
                (cidx, db_cluster_id, label, size, mean_sev, mean_pi, tool_rate, buyer_rate, score, keywords, lane_counts)
            )

        con.commit()

        # Sort clusters by opportunity score (descending), keep noise at bottom
        cluster_stats_rows.sort(key=lambda r: (r[0] == -1, -r[8]))

        lines: list[str] = []
        lines.append(f"# Topics Report ({method})")
        lines.append("")
        lines.append(f"- Generated: {utcnow_iso()}")
        lines.append(f"- Database: `{DB_PATH.as_posix()}`")
        lines.append(f"- Embedded docs clustered: {n}")
        lines.append(f"- k selected: {best_k}")
        lines.append(f"- silhouette(cosine): {best_sil:.4f}")
        lines.append(f"- Noise quantile: {args.noise_quantile} (n_noise={n_noise})")
        lines.append("")
        lines.append("## Ranked topic clusters")
        lines.append("")
        lines.append("| Label | Cluster ID | Size | Mean Sev | Mean Purchase | Tool-rate | Buyer-rate | Opportunity | Keywords | Lane mix |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")

        for cidx, dbid, label, size, mean_sev, mean_pi, tool_rate, buyer_rate, score, keywords, lane_counts in cluster_stats_rows:
            lane_mix = ", ".join([f"{k}:{v}" for k, v in sorted(lane_counts.items()) if k]) or "n/a"
            lines.append(
                f"| {label} | {dbid} | {size} | {mean_sev:.2f} | {mean_pi:.2f} | {tool_rate:.2f} | {buyer_rate:.2f} | {score:.2f} | {keywords} | {lane_mix} |"
            )

        def examples_for_cluster(cidx: int, n_ex: int = 6):
            member_ids = cluster_members.get(cidx, [])
            if not member_ids:
                return []
            member_set = set(member_ids)
            member_idx = [i for i, pid in enumerate(ids) if pid in member_set]
            member_idx.sort(key=lambda i: (pi[i], sev[i]), reverse=True)

            out = []
            for i in member_idx[:n_ex]:
                snippet = clean_text(docs[i])
                if len(snippet) > 240:
                    snippet = snippet[:240] + "…"
                out.append((lanes[i], sev[i], pi[i], snippet, urls[i]))
            return out

        lines.append("")
        lines.append("## Examples (top clusters)")
        lines.append("")

        top_clusters = [r[0] for r in cluster_stats_rows if r[0] != -1][:10]
        for cidx in top_clusters:
            dbid = cluster_db_ids[cidx]
            row = next(r for r in cluster_stats_rows if r[0] == cidx)
            keywords = row[9]
            label = f"T-{cidx:03d}"

            lines.append(f"### {label} (Cluster {dbid})")
            lines.append(f"- Keywords: {keywords}")
            lines.append("")
            for lane, s, p, snippet, url in examples_for_cluster(cidx, n_ex=6):
                lines.append(f"- (lane={lane}, sev={s}, buy={p}) {snippet}")
                lines.append(f"  - Source: {url}")
            lines.append("")

        if -1 in cluster_members:
            dbid = cluster_db_ids[-1]
            lines.append(f"## Noise / Outliers (Cluster {dbid})")
            lines.append("")
            for lane, s, p, snippet, url in examples_for_cluster(-1, n_ex=10):
                lines.append(f"- (lane={lane}, sev={s}, buy={p}) {snippet}")
                lines.append(f"  - Source: {url}")
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
