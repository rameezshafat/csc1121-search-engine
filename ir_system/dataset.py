from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


def iter_docs(
    metadata_csv: Path,
    books_dir: Path,
    language_filter: str | None = None,
) -> Iterable[Dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        seen = set()
        for row in reader:
            doc_id = (row.get("gutenberg_id") or "").strip()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)

            language = (row.get("language") or "").strip().lower()
            has_text = (row.get("has_text") or "").strip().lower() == "true"
            if not has_text:
                continue
            if language_filter and language != language_filter.lower():
                continue

            text_path = books_dir / f"{doc_id}.txt"
            if not text_path.exists():
                continue

            yield {
                "doc_id": doc_id,
                "text_path": str(text_path),
                "title": (row.get("title") or "").strip(),
                "author": (row.get("author") or "").strip(),
                "language": language,
                "bookshelf": (row.get("gutenberg_bookshelf") or "").strip(),
            }


def batched(items: List[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_dataset(
    metadata_csv: Path,
    books_dir: Path,
    max_docs: int | None = None,
    language_filter: str | None = None,
) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for item in iter_docs(metadata_csv=metadata_csv, books_dir=books_dir, language_filter=language_filter):
        docs.append(item)
        if max_docs is not None and len(docs) >= max_docs:
            break
    return docs
