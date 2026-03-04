from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .config import RankConfig, StructuredWeights
from .preprocess import PreprocessConfig, TextAnalyzer


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SearchEngine:
    def __init__(
        self,
        index_db: Path,
        stats_dir: Path,
        rank_cfg: RankConfig,
        structured_weights: StructuredWeights | None = None,
        books_dir: Path | None = None,
    ) -> None:
        self.index_db = index_db
        self.stats_dir = stats_dir
        self.rank_cfg = rank_cfg
        self.weights = structured_weights or StructuredWeights()
        self.books_dir = books_dir

        self.df: Dict[str, int] = load_json(stats_dir / "df.json")
        self.doc_lengths: Dict[str, int] = load_json(stats_dir / "doc_lengths.json")
        self.avgdl = float(load_json(stats_dir / "avgdl.json").get("avgdl", 0.0))
        self.doc_norms: Dict[str, float] = load_json(stats_dir / "doc_norms.json")
        self.N = len(self.doc_lengths)

        pre_cfg_path = stats_dir / "preprocessing_config.json"
        if pre_cfg_path.exists():
            pre = load_json(pre_cfg_path)
            self.pre_cfg = PreprocessConfig(
                min_token_len=int(pre.get("min_token_len", 2)),
                use_stopwords=bool(pre.get("use_stopwords", False)),
                use_stemming=bool(pre.get("use_stemming", False)),
            )
        else:
            self.pre_cfg = PreprocessConfig(min_token_len=2, use_stopwords=False, use_stemming=False)
        self.analyzer = TextAnalyzer(self.pre_cfg)

    def _idf_bm25(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _idf_tfidf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.N + 1) / (df + 1)) + 1.0

    def search_bm25(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        terms = self.analyzer.tokenize(query)
        scores = defaultdict(float)
        k1 = self.rank_cfg.bm25_k1
        b = self.rank_cfg.bm25_b

        conn = sqlite3.connect(self.index_db)
        for term in terms:
            idf = self._idf_bm25(term)
            if idf == 0.0:
                continue
            for doc_id, tf in conn.execute("SELECT doc_id, tf FROM postings WHERE term = ?", (term,)):
                dl = float(self.doc_lengths.get(doc_id, 0))
                denom = tf + k1 * (1.0 - b + b * (dl / self.avgdl if self.avgdl else 0.0))
                score = idf * (tf * (k1 + 1.0) / denom)
                scores[doc_id] += score
        conn.close()

        k = top_k or self.rank_cfg.top_k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def search_tfidf_cosine(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        terms = self.analyzer.tokenize(query)
        q_tf = defaultdict(int)
        for t in terms:
            q_tf[t] += 1

        q_weights: Dict[str, float] = {}
        q_norm_sq = 0.0
        for term, tf in q_tf.items():
            w = (1.0 + math.log(tf)) * self._idf_tfidf(term)
            q_weights[term] = w
            q_norm_sq += w * w

        q_norm = math.sqrt(q_norm_sq) or 1.0
        dot = defaultdict(float)

        conn = sqlite3.connect(self.index_db)
        for term, q_w in q_weights.items():
            if q_w == 0.0:
                continue
            idf = self._idf_tfidf(term)
            for doc_id, tf in conn.execute("SELECT doc_id, tf FROM postings WHERE term = ?", (term,)):
                d_w = (1.0 + math.log(tf)) * idf
                dot[doc_id] += q_w * d_w
        conn.close()

        scores = {}
        for doc_id, num in dot.items():
            d_norm = float(self.doc_norms.get(doc_id, 0.0)) or 1.0
            scores[doc_id] = num / (q_norm * d_norm)

        k = top_k or self.rank_cfg.top_k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def search_structured(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        terms = self.analyzer.tokenize(query, min_len=1)
        if not terms:
            return []

        scores = defaultdict(float)
        conn = sqlite3.connect(self.index_db)
        for term in terms:
            pattern = f"%{term}%"
            for doc_id, title, author, bookshelf in conn.execute(
                """
                SELECT doc_id, COALESCE(title, ''), COALESCE(author, ''), COALESCE(bookshelf, '')
                FROM metadata
                WHERE lower(title) LIKE ? OR lower(author) LIKE ? OR lower(bookshelf) LIKE ?
                """,
                (pattern, pattern, pattern),
            ):
                t = title.lower()
                a = author.lower()
                b = bookshelf.lower()
                if term in t:
                    scores[doc_id] += self.weights.title
                if term in a:
                    scores[doc_id] += self.weights.author
                if term in b:
                    scores[doc_id] += self.weights.bookshelf
        conn.close()

        k = top_k or self.rank_cfg.top_k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def search(self, model: str, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        if model == "bm25":
            return self.search_bm25(query, top_k=top_k)
        if model == "tfidf":
            return self.search_tfidf_cosine(query, top_k=top_k)
        if model == "structured":
            return self.search_structured(query, top_k=top_k)
        raise ValueError(f"Unknown model: {model}")

    def get_metadata(self, doc_id: str) -> Tuple[str, str]:
        conn = sqlite3.connect(self.index_db)
        row = conn.execute(
            "SELECT COALESCE(title,''), COALESCE(author,'') FROM metadata WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        conn.close()
        if not row:
            return "", ""
        return row[0], row[1]

    def get_preview(self, doc_id: str, query: str, max_len: int = 150) -> str:
        preview, _ = self.get_preview_with_line(doc_id, query, max_len=max_len)
        return preview

    def get_preview_with_line(self, doc_id: str, query: str, max_len: int = 150) -> Tuple[str, int]:
        if self.books_dir is None:
            return "", 0
        text_path = self.books_dir / f"{doc_id}.txt"
        if not text_path.exists():
            return "", 0
        terms = self.analyzer.tokenize(query, min_len=1)
        try:
            with text_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    low = line.lower()
                    if any(t in low for t in terms):
                        s = " ".join(line.strip().split())
                        return s[:max_len], line_no
        except OSError:
            return "", 0
        return "", 0
