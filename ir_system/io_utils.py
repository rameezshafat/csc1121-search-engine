from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=True, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def connect_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_postings_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS postings (
            term TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            tf INTEGER NOT NULL,
            PRIMARY KEY (term, doc_id)
        );

        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            length INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metadata (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            language TEXT,
            bookshelf TEXT
        );
        """
    )
    conn.commit()
