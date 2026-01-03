import sqlite3
from pathlib import Path

DB_PATH = Path("data/db/research.sqlite")

def main():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_documents_url ON raw_documents(url);")
    print("Added/verified unique index: ux_raw_documents_url on raw_documents(url)")

if __name__ == "__main__":
    main()
