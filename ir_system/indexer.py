from __future__ import annotations

import math
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .config import IndexConfig, Paths
from .dataset import batched, load_dataset
from .io_utils import atomic_write_json, connect_sqlite, ensure_dirs, init_postings_db
from .preprocess import PreprocessConfig, TextAnalyzer


class ProgressHeartbeat:
    def __init__(self, label: str, total: int | None = None, interval_seconds: float = 5.0) -> None:
        self.label = label
        self.total = total
        self.interval_seconds = interval_seconds
        self.start = time.time()
        self.last = self.start
        self.count = 0

    def update(self, amount: int = 1, force: bool = False) -> None:
        self.count += amount
        now = time.time()
        if force or (now - self.last >= self.interval_seconds):
            elapsed = now - self.start
            if self.total and self.total > 0:
                pct = 100.0 * self.count / self.total
                print(f"[{self.label}] {self.count}/{self.total} ({pct:.2f}%) elapsed={elapsed:.1f}s", flush=True)
            else:
                print(f"[{self.label}] {self.count} elapsed={elapsed:.1f}s", flush=True)
            self.last = now

    def done(self) -> None:
        self.update(0, force=True)


class IndexBuilder:
    def __init__(self, paths: Paths, cfg: IndexConfig) -> None:
        self.paths = paths
        self.cfg = cfg
        self.pre_cfg = PreprocessConfig(
            min_token_len=cfg.min_token_len,
            use_stopwords=cfg.use_stopwords,
            use_stemming=cfg.use_stemming,
        )
        self.analyzer = TextAnalyzer(self.pre_cfg)
        self.ckpt_path = self.paths.stats_dir / "index_checkpoint.json"

    def _checkpoint_config(self) -> Dict[str, object]:
        return {
            "batch_size": self.cfg.batch_size,
            "min_token_len": self.cfg.min_token_len,
            "max_docs": self.cfg.max_docs,
            "language_filter": self.cfg.language_filter,
            "use_stopwords": self.cfg.use_stopwords,
            "use_stemming": self.cfg.use_stemming,
        }

    def _load_checkpoint(self) -> Dict[str, object] | None:
        if not self.ckpt_path.exists():
            return None
        try:
            import json

            return json.loads(self.ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_checkpoint(self, data: Dict[str, object]) -> None:
        atomic_write_json(self.ckpt_path, data)

    def _clear_checkpoint(self) -> None:
        self.ckpt_path.unlink(missing_ok=True)

    def _estimate_dry_run(self, docs: List[Dict[str, str]]) -> Dict[str, int]:
        sample = docs[: min(300, len(docs))]
        if not sample:
            print("Dry-run: no documents found.", flush=True)
            return {"docs": 0, "estimated_terms": 0, "estimated_index_mb": 0}

        total_tokens = 0
        total_unique = 0
        hb = ProgressHeartbeat("dry_run_sample", total=len(sample), interval_seconds=1.0)
        for d in sample:
            text = Path(d["text_path"]).read_text(encoding="utf-8", errors="ignore")
            toks = self.analyzer.tokenize(text)
            total_tokens += len(toks)
            total_unique += len(set(toks))
            hb.update(1)
        hb.done()

        avg_tokens = total_tokens / len(sample)
        avg_unique = total_unique / len(sample)
        est_terms = int(avg_unique * len(docs))
        # rough estimate: 80 bytes per postings/doc metadata footprint average
        est_mb = int((avg_tokens * len(docs) * 80) / (1024 * 1024))
        est_minutes = (len(docs) / 1000) * 0.9  # empirical rough guess

        print(
            "Dry-run estimate: "
            f"docs={len(docs)}, avg_tokens/doc={avg_tokens:.1f}, avg_unique/doc={avg_unique:.1f}, "
            f"estimated_terms~{est_terms}, estimated_index_size~{est_mb}MB, "
            f"estimated_build_time~{est_minutes:.1f}min",
            flush=True,
        )
        return {"docs": len(docs), "estimated_terms": est_terms, "estimated_index_mb": est_mb}

    def build(self) -> Dict[str, int]:
        started = time.time()
        print("Preparing index build...", flush=True)
        print(
            f"Preprocessing options: stopwords={self.cfg.use_stopwords}, stemming={self.cfg.use_stemming}, min_token_len={self.cfg.min_token_len}",
            flush=True,
        )
        print("Loading dataset candidates...", flush=True)
        docs = load_dataset(
            self.paths.metadata_csv,
            self.paths.books_dir,
            max_docs=self.cfg.max_docs,
            language_filter=self.cfg.language_filter,
        )
        total_docs = len(docs)
        total_batches = (total_docs + self.cfg.batch_size - 1) // self.cfg.batch_size if total_docs else 0
        print(f"Loaded {total_docs} documents. batch_size={self.cfg.batch_size}, batches={total_batches}", flush=True)

        if self.cfg.dry_run:
            print("Dry-run mode enabled: no files will be written.", flush=True)
            return self._estimate_dry_run(docs)

        ensure_dirs(
            self.paths.processed_dir,
            self.paths.partial_indexes_dir,
            self.paths.final_index_dir,
            self.paths.stats_dir,
            self.paths.results_dir,
        )

        atomic_write_json(
            self.paths.stats_dir / "preprocessing_config.json",
            {
                "use_stopwords": self.cfg.use_stopwords,
                "use_stemming": self.cfg.use_stemming,
                "min_token_len": self.cfg.min_token_len,
            },
        )

        if not self.cfg.resume:
            print("Cleaning old partial index artifacts...", flush=True)
            for old_part in self.paths.partial_indexes_dir.glob("part_*.sqlite3*"):
                old_part.unlink(missing_ok=True)
            self._clear_checkpoint()

        part_files = self._build_partial_indexes(docs)

        final_db = self.paths.final_index_dir / "index.sqlite3"
        print("Merging partial indexes...", flush=True)
        self._merge_partial_indexes(part_files, final_db)
        print("Merge complete. Cleaning partial indexes to free disk space...", flush=True)
        for part in part_files:
            part.unlink(missing_ok=True)
            (part.parent / f"{part.name}-wal").unlink(missing_ok=True)
            (part.parent / f"{part.name}-shm").unlink(missing_ok=True)

        print("Computing collection statistics...", flush=True)
        stats = self._compute_and_store_stats(final_db)
        stats["partial_files"] = len(part_files)

        self._clear_checkpoint()
        total_elapsed = time.time() - started
        print(f"Index build complete in {total_elapsed:.1f}s: {stats}", flush=True)
        return stats

    def _build_partial_indexes(self, docs: List[Dict[str, str]]) -> List[Path]:
        part_files: List[Path] = []
        batches = list(batched(docs, self.cfg.batch_size))
        overall = ProgressHeartbeat("index_docs", total=len(docs), interval_seconds=5.0)

        done_batches = set()
        if self.cfg.resume:
            ckpt = self._load_checkpoint()
            if ckpt and ckpt.get("config") == self._checkpoint_config():
                done_batches = set(int(x) for x in ckpt.get("done_batches", []))
                print(f"Resuming index build: {len(done_batches)} batches already completed.", flush=True)
            elif ckpt:
                print("Checkpoint config mismatch. Starting fresh batches for current configuration.", flush=True)

        for batch_num, batch_docs in enumerate(batches, start=1):
            free_bytes = shutil.disk_usage(self.paths.partial_indexes_dir).free
            if free_bytes < 2 * 1024 * 1024 * 1024:
                free_mb = free_bytes / (1024 * 1024)
                raise RuntimeError(
                    f"Low disk space: only {free_mb:.1f} MiB free. "
                    "Need at least 2048 MiB to continue indexing. "
                    "Delete old files under processed/partial_indexes and rerun."
                )

            part_path = self.paths.partial_indexes_dir / f"part_{batch_num:05d}.sqlite3"
            if batch_num in done_batches and part_path.exists():
                overall.update(len(batch_docs), force=True)
                part_files.append(part_path)
                print(f"Skipping batch {batch_num}/{len(batches)} (already checkpointed).", flush=True)
                continue

            batch_start = time.time()
            print(f"Starting batch {batch_num}/{len(batches)} with {len(batch_docs)} docs...", flush=True)
            term_postings: Dict[str, Dict[str, int]] = defaultdict(dict)
            doc_lengths: Dict[str, int] = {}
            metadata_rows: List[tuple[str, str, str, str, str]] = []

            for doc in batch_docs:
                doc_id = doc["doc_id"]
                text = Path(doc["text_path"]).read_text(encoding="utf-8", errors="ignore")
                tokens = self.analyzer.tokenize(text)
                tf = Counter(tokens)

                doc_lengths[doc_id] = len(tokens)
                metadata_rows.append(
                    (
                        doc_id,
                        doc.get("title", ""),
                        doc.get("author", ""),
                        doc.get("language", ""),
                        doc.get("bookshelf", ""),
                    )
                )

                for term, freq in tf.items():
                    term_postings[term][doc_id] = int(freq)

                overall.update(1)

            part_path.unlink(missing_ok=True)
            conn = connect_sqlite(part_path)
            init_postings_db(conn)
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO documents(doc_id, length) VALUES(?, ?)",
                    ((doc_id, length) for doc_id, length in doc_lengths.items()),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO metadata(doc_id, title, author, language, bookshelf)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    metadata_rows,
                )
                for term, postings in term_postings.items():
                    conn.executemany(
                        "INSERT OR REPLACE INTO postings(term, doc_id, tf) VALUES(?, ?, ?)",
                        ((term, doc_id, tf) for doc_id, tf in postings.items()),
                    )
            conn.close()

            part_files.append(part_path)
            batch_elapsed = time.time() - batch_start
            print(f"Finished batch {batch_num}/{len(batches)} in {batch_elapsed:.1f}s -> {part_path.name}", flush=True)

            done_batches.add(batch_num)
            self._save_checkpoint(
                {
                    "config": self._checkpoint_config(),
                    "done_batches": sorted(done_batches),
                    "total_batches": len(batches),
                    "total_docs": len(docs),
                }
            )

        overall.done()
        return part_files

    def _merge_partial_indexes(self, part_files: List[Path], final_db: Path) -> None:
        final_db.unlink(missing_ok=True)
        conn = connect_sqlite(final_db)
        init_postings_db(conn)
        with conn:
            conn.execute("DROP INDEX IF EXISTS idx_postings_term")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-200000")

        hb = ProgressHeartbeat("merge_parts", total=len(part_files), interval_seconds=2.0)
        for part_path in part_files:
            conn.execute("ATTACH DATABASE ? AS partdb", (str(part_path),))
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents(doc_id, length)
                    SELECT doc_id, length FROM partdb.documents
                    """
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata(doc_id, title, author, language, bookshelf)
                    SELECT doc_id, title, author, language, bookshelf FROM partdb.metadata
                    """
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO postings(term, doc_id, tf)
                    SELECT term, doc_id, tf FROM partdb.postings
                    """
                )
            conn.execute("DETACH DATABASE partdb")
            hb.update(1)

        hb.done()
        with conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term)")
        conn.close()

    def _compute_and_store_stats(self, final_db: Path) -> Dict[str, int]:
        conn = sqlite3.connect(final_db)
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        avgdl = conn.execute("SELECT AVG(length) FROM documents").fetchone()[0] or 0.0

        total_terms = conn.execute("SELECT COUNT(*) FROM (SELECT term FROM postings GROUP BY term)").fetchone()[0]

        df: Dict[str, int] = {}
        hb_df = ProgressHeartbeat("df_terms", total=total_terms, interval_seconds=2.0)
        for term, count in conn.execute("SELECT term, COUNT(*) as df FROM postings GROUP BY term ORDER BY term"):
            df[term] = int(count)
            hb_df.update(1)
        hb_df.done()

        doc_lengths: Dict[str, int] = {}
        for doc_id, length in conn.execute("SELECT doc_id, length FROM documents"):
            doc_lengths[doc_id] = int(length)

        doc_norm_acc = defaultdict(float)
        hb_norm = ProgressHeartbeat("doc_norm_terms", total=total_terms, interval_seconds=2.0)
        for term, dfi in df.items():
            idf = math.log((total_docs + 1) / (dfi + 1)) + 1.0
            for doc_id, tf in conn.execute("SELECT doc_id, tf FROM postings WHERE term = ?", (term,)):
                w = (1.0 + math.log(tf)) * idf
                doc_norm_acc[doc_id] += w * w
            hb_norm.update(1)
        hb_norm.done()

        doc_norms = {doc_id: math.sqrt(v) for doc_id, v in doc_norm_acc.items()}
        conn.close()

        atomic_write_json(self.paths.stats_dir / "df.json", df)
        atomic_write_json(self.paths.stats_dir / "doc_lengths.json", doc_lengths)
        atomic_write_json(self.paths.stats_dir / "avgdl.json", {"avgdl": avgdl})
        atomic_write_json(self.paths.stats_dir / "doc_norms.json", doc_norms)

        return {"docs": total_docs, "terms": len(df), "partial_files": 0}


def save_run_manifest(paths: Paths, cfg: IndexConfig, stats: Dict[str, int]) -> None:
    manifest = {
        "dataset": "filtered" if cfg.language_filter else "all_languages",
        "language_filter": cfg.language_filter,
        "batch_size": cfg.batch_size,
        "preprocessing": {
            "use_stopwords": cfg.use_stopwords,
            "use_stemming": cfg.use_stemming,
            "min_token_len": cfg.min_token_len,
        },
        "stats": stats,
    }
    atomic_write_json(paths.stats_dir / "run_manifest.json", manifest)
