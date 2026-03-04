# CSC1121 Assignment 1 Search Engine

This implementation supports the three required retrieval models:
- `structured` (metadata/title-author-bookshelf search)
- `tfidf` (Vector Space Model with cosine similarity)
- `bm25`

It builds a disk-backed inverted index in batches (SQLite) and stores derived artifacts in:
- `processed/partial_indexes/`
- `processed/final_index/`
- `processed/stats/`
- `results/`

Source data under `data/` is read-only.

## Build index

```bash
# all languages by default
python3 main.py --project-root . build-index --batch-size 250

# English-only mode (optional)
python3 main.py --project-root . build-index --batch-size 250 --language-filter en
```

Optional preprocessing flags:

```bash
python3 main.py --project-root . build-index --batch-size 250 --use-stopwords --use-stemming
```

Resumable indexing after interruption:

```bash
python3 main.py --project-root . build-index --batch-size 250 --resume
```

Dry run (estimate size/time, no index files written):

```bash
python3 main.py --project-root . build-index --dry-run --max-docs 5000
```

## Search

```bash
python3 main.py --project-root . search --model structured --query "Philip K Dick" --top-k 10
python3 main.py --project-root . search --model tfidf --query "English Grammar" --top-k 10
python3 main.py --project-root . search --model bm25 --query "to be, or not to be" --top-k 10
```

Structured field weights are configurable:

```bash
python3 main.py --project-root . search --model structured --query "English Grammar" \
  --structured-title-weight 3.0 --structured-author-weight 2.0 --structured-bookshelf-weight 1.0
```

Search output prints a top-10 table with:
- rank
- book_id
- title
- author
- score
- preview (first 150 chars of matching line)

## Run assignment evaluation + TSV export

```bash
python3 main.py --project-root . run-experiments --output results/evaluation_summary.txt --tsv-dir results
```

`run-experiments` automatically executes these 6 assignment queries:
1. `to be, or not to be`
2. `English Grammar`
3. `Philip K Dick`
4. `Jabberwocky`
5. `Gutenberg`
6. `Dornröschen`

Optional override:

```bash
python3 main.py --project-root . run-experiments --queries-file config/queries.txt
```

This writes:
- summary report: `results/evaluation_summary.txt`
- required TSV files: `results/<query_nr>_<model_name>.tsv` (6 queries x 3 models = 18 files)

TSV columns:
- `rank`
- `book_id`
- `score`
- `preview`
- `start_line`

`evaluation_summary.txt` includes:
- Intersection@10
- Intersection@100
- ranking displacement@100
- mean top-10 doc length
