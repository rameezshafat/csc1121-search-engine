from __future__ import annotations

import argparse
from pathlib import Path

from ir_system.config import IndexConfig, RankConfig, StructuredWeights, build_paths
from ir_system.evaluate import run_experiment
from ir_system.indexer import IndexBuilder, save_run_manifest
from ir_system.search import SearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disk-backed search engine for Assignment 1")
    parser.add_argument("--project-root", default=".", help="Project root path")

    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Build batch index and stats")
    b.add_argument("--batch-size", type=int, default=250)
    b.add_argument("--min-token-len", type=int, default=2)
    b.add_argument("--max-docs", type=int, default=None)
    b.add_argument("--language-filter", default=None, help="Optional language code (e.g., en). Default is all languages.")
    b.add_argument("--use-stopwords", action="store_true", help="Enable stopword filtering during indexing and querying")
    b.add_argument("--use-stemming", action="store_true", help="Enable Porter stemming during indexing and querying")
    b.add_argument("--resume", action="store_true", help="Resume indexing from checkpointed completed batches")
    b.add_argument("--dry-run", action="store_true", help="Estimate index size/time without writing index files")

    s = sub.add_parser("search", help="Run a single query")
    s.add_argument("--model", choices=["structured", "bm25", "tfidf"], default="bm25")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--structured-title-weight", type=float, default=3.0)
    s.add_argument("--structured-author-weight", type=float, default=2.0)
    s.add_argument("--structured-bookshelf-weight", type=float, default=1.0)

    e = sub.add_parser("run-experiments", help="Run assignment queries and export outputs")
    e.add_argument("--queries-file", default=None, help="Optional path to 6 assignment queries (defaults to built-in assignment queries)")
    e.add_argument("--output", default=None, help="Path to evaluation summary")
    e.add_argument("--tsv-dir", default=None, help="Directory for TSV result files")
    e.add_argument("--structured-title-weight", type=float, default=3.0)
    e.add_argument("--structured-author-weight", type=float, default=2.0)
    e.add_argument("--structured-bookshelf-weight", type=float, default=1.0)

    return parser.parse_args()


def _print_query_table(engine: SearchEngine, query: str, results: list[tuple[str, float]], top_k: int) -> None:
    print("\nTop Results")
    print("-" * 240)
    print(f"{'Rank':<5} {'BookID':<8} {'Title':<45} {'Author':<25} {'Score':>10}  {'Preview (first 150 chars of match)':<140}")
    print("-" * 240)
    for rank, (doc_id, score) in enumerate(results[:top_k], start=1):
        title, author = engine.get_metadata(doc_id)
        preview = engine.get_preview(doc_id, query, max_len=150)
        title = (title[:42] + "...") if len(title) > 45 else title
        author = (author[:22] + "...") if len(author) > 25 else author
        print(f"{rank:<5} {doc_id:<8} {title:<45} {author:<25} {score:>10.6f}  {preview:<140}")
    print("-" * 240)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    paths = build_paths(project_root)

    if args.cmd == "build-index":
        language_filter = args.language_filter
        if language_filter and language_filter.lower() in {"all", "*"}:
            language_filter = None
        cfg = IndexConfig(
            batch_size=args.batch_size,
            min_token_len=args.min_token_len,
            max_docs=args.max_docs,
            language_filter=language_filter,
            use_stopwords=args.use_stopwords,
            use_stemming=args.use_stemming,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        builder = IndexBuilder(paths=paths, cfg=cfg)
        stats = builder.build()
        if not args.dry_run:
            save_run_manifest(paths, cfg, stats)
        print(f"Index build complete: {stats}")
        return

    weights = StructuredWeights(
        title=args.structured_title_weight,
        author=args.structured_author_weight,
        bookshelf=args.structured_bookshelf_weight,
    )

    rank_cfg = RankConfig(top_k=getattr(args, "top_k", 100))
    engine = SearchEngine(
        index_db=paths.final_index_dir / "index.sqlite3",
        stats_dir=paths.stats_dir,
        rank_cfg=rank_cfg,
        structured_weights=weights,
        books_dir=paths.books_dir,
    )

    if args.cmd == "search":
        results = engine.search(args.model, args.query, top_k=args.top_k)
        _print_query_table(engine, args.query, results, top_k=min(args.top_k, 10))
        return

    if args.cmd == "run-experiments":
        queries_file = Path(args.queries_file).resolve() if args.queries_file else None
        output_path = Path(args.output).resolve() if args.output else paths.results_dir / "evaluation_summary.txt"
        tsv_dir = Path(args.tsv_dir).resolve() if args.tsv_dir else paths.results_dir
        run_experiment(
            index_db=paths.final_index_dir / "index.sqlite3",
            stats_dir=paths.stats_dir,
            books_dir=paths.books_dir,
            queries_file=queries_file,
            results_file=output_path,
            tsv_dir=tsv_dir,
            structured_weights=weights,
        )
        print(f"Wrote evaluation summary to {output_path}")
        print(f"Wrote TSV outputs to {tsv_dir}")
        return


if __name__ == "__main__":
    main()
