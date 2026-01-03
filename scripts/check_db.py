import sqlite3
from pathlib import Path

DB_PATH = Path("data/db/research.sqlite")

def scalar(con, sql):
    return con.execute(sql).fetchone()[0]

def main():
    with sqlite3.connect(DB_PATH) as con:
        sources = scalar(con, "SELECT COUNT(*) FROM sources;")
        docs = scalar(con, "SELECT COUNT(*) FROM raw_documents;")
        pain = scalar(con, "SELECT COUNT(*) FROM pain_units;")

        print(f"sources: {sources}")
        print(f"raw_documents: {docs}")
        print(f"pain_units: {pain}")

        row = con.execute(
            "SELECT lane, site, query, collected_at_utc FROM sources ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        if row:
            print("latest source:", row)

        row2 = con.execute(
            "SELECT title, url FROM raw_documents ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        if row2:
            print("latest document:", row2[0])
            print("url:", row2[1])

if __name__ == "__main__":
    main()
