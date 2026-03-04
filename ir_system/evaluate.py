from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from .config import RankConfig, StructuredWeights
from .search import SearchEngine

MODELS = ["structured", "tfidf", "bm25"]
ASSIGNMENT_QUERIES = [
    "to be, or not to be",
    "English Grammar",
    "Philip K Dick",
    "Jabberwocky",
    "Gutenberg",
    "Dornröschen",
]


def load_queries(path: Path | None = None) -> List[str]:
    if path is None:
        return ASSIGNMENT_QUERIES

    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                queries.append(q)

    if len(queries) != 6:
        raise ValueError(f"Expected exactly 6 queries, found {len(queries)} in {path}")
    return queries


def intersection_at_k(a: List[str], b: List[str], k: int) -> int:
    return len(set(a[:k]).intersection(b[:k]))


def ranking_displacement(a: List[str], b: List[str], k: int) -> float:
    pos_a = {doc_id: i for i, doc_id in enumerate(a[:k])}
    pos_b = {doc_id: i for i, doc_id in enumerate(b[:k])}
    common = set(pos_a).intersection(pos_b)
    if not common:
        return float(k)
    return mean(abs(pos_a[d] - pos_b[d]) for d in common)


def avg_doc_len(doc_ids: List[str], doc_lengths: Dict[str, int]) -> float:
    if not doc_ids:
        return 0.0
    vals = [doc_lengths.get(doc_id, 0) for doc_id in doc_ids]
    return float(mean(vals)) if vals else 0.0


def write_query_model_tsv(
    results_dir: Path,
    query_num: int,
    model: str,
    query: str,
    ranked: List[Tuple[str, float]],
    engine: SearchEngine,
) -> Path:
    out_path = results_dir / f"{query_num}_{model}.tsv"
    results_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["rank", "book_id", "score", "preview", "start_line"])
        for rank, (doc_id, score) in enumerate(ranked[:100], start=1):
            preview, start_line = engine.get_preview_with_line(doc_id, query, max_len=180)
            writer.writerow([rank, doc_id, f"{score:.6f}", preview, start_line])

    return out_path


def run_experiment(
    index_db: Path,
    stats_dir: Path,
    books_dir: Path,
    queries_file: Path | None,
    results_file: Path,
    tsv_dir: Path,
    structured_weights: StructuredWeights,
) -> None:
    rank_cfg = RankConfig(top_k=100)
    engine = SearchEngine(
        index_db=index_db,
        stats_dir=stats_dir,
        rank_cfg=rank_cfg,
        structured_weights=structured_weights,
        books_dir=books_dir,
    )
    queries = load_queries(queries_file)

    lines: List[str] = []
    lines.append("Evaluation Summary")
    lines.append("Models: structured vs TF-IDF cosine vs BM25")
    lines.append("Top-k evaluated: 10 and 100")
    lines.append("")

    pairwise_keys = [("structured", "tfidf"), ("structured", "bm25"), ("tfidf", "bm25")]
    all_intersections_10: Dict[Tuple[str, str], List[int]] = {k: [] for k in pairwise_keys}
    all_intersections_100: Dict[Tuple[str, str], List[int]] = {k: [] for k in pairwise_keys}
    all_displacements: Dict[Tuple[str, str], List[float]] = {k: [] for k in pairwise_keys}
    length_bias: Dict[str, List[float]] = {m: [] for m in MODELS}

    for idx, q in enumerate(queries, start=1):
        lines.append(f"Query {idx}: {q}")
        ranked_by_model: Dict[str, List[Tuple[str, float]]] = {}

        for model in MODELS:
            ranked = engine.search(model, q, top_k=100)
            ranked_by_model[model] = ranked
            write_query_model_tsv(tsv_dir, idx, model, q, ranked, engine)
            docs = [doc_id for doc_id, _ in ranked]
            length_bias[model].append(avg_doc_len(docs[:10], engine.doc_lengths))

        for left, right in pairwise_keys:
            left_docs = [doc_id for doc_id, _ in ranked_by_model[left]]
            right_docs = [doc_id for doc_id, _ in ranked_by_model[right]]
            i10 = intersection_at_k(left_docs, right_docs, 10)
            i100 = intersection_at_k(left_docs, right_docs, 100)
            disp = ranking_displacement(left_docs, right_docs, 100)
            all_intersections_10[(left, right)].append(i10)
            all_intersections_100[(left, right)].append(i100)
            all_displacements[(left, right)].append(disp)
            lines.append(
                f"  {left} vs {right} -> Intersection@10={i10}, Intersection@100={i100}, displacement@100={disp:.3f}"
            )

        lines.append("  Avg top10 doc length -> " + ", ".join(f"{m}:{length_bias[m][-1]:.2f}" for m in MODELS))
        lines.append("")

    lines.append("Aggregate")
    lines.append("| Pair | Mean Intersection@10 | Mean Intersection@100 | Mean displacement@100 |")
    lines.append("|---|---:|---:|---:|")
    for left, right in pairwise_keys:
        mi10 = mean(all_intersections_10[(left, right)])
        mi100 = mean(all_intersections_100[(left, right)])
        md = mean(all_displacements[(left, right)])
        lines.append(f"| {left} vs {right} | {mi10:.3f} | {mi100:.3f} | {md:.3f} |")

    lines.append(
        "Mean top10 doc length by model: "
        + ", ".join(f"{m}={mean(length_bias[m]):.2f}" for m in MODELS)
    )

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
