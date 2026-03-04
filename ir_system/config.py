from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    books_dir: Path
    metadata_csv: Path
    processed_dir: Path
    partial_indexes_dir: Path
    final_index_dir: Path
    stats_dir: Path
    results_dir: Path
    queries_file: Path


@dataclass(frozen=True)
class IndexConfig:
    batch_size: int = 250
    min_token_len: int = 2
    max_docs: int | None = None
    language_filter: str | None = None
    use_stopwords: bool = False
    use_stemming: bool = False
    resume: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RankConfig:
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    top_k: int = 100


@dataclass(frozen=True)
class StructuredWeights:
    # Title is weighted highest because it is the strongest topical summary for a book.
    title: float = 3.0
    # Author is next: author matches are strong but can be broad for prolific authors.
    author: float = 2.0
    # Bookshelf is coarse metadata, helpful but less specific.
    bookshelf: float = 1.0


def build_paths(project_root: Path) -> Paths:
    data_dir = project_root / "data"
    processed_dir = project_root / "processed"
    return Paths(
        data_dir=data_dir,
        books_dir=data_dir / "books",
        metadata_csv=data_dir / "metadata.csv",
        processed_dir=processed_dir,
        partial_indexes_dir=processed_dir / "partial_indexes",
        final_index_dir=processed_dir / "final_index",
        stats_dir=processed_dir / "stats",
        results_dir=project_root / "results",
        queries_file=project_root / "config" / "queries.txt",
    )
