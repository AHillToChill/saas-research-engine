# scripts/init_db.py
from pathlib import Path
import sqlite3

DB_PATH = Path("data/db/research.sqlite")
SQL_PATH = Path("scripts/init_db.sql")

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sql = SQL_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(sql)
    print(f"Initialized DB at {DB_PATH}")

if __name__ == "__main__":
    main()
