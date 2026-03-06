# Assignment 1 Validation Report

Project: `csc1121-search-engine`  
Validated against: `data/Assignment_1.pdf`  
Date: 2026-02-26

## Overall status
**Not submission-ready yet** (core IR pipeline works, but several assignment-required deliverables are missing).

## Requirement Checklist

1. Implement indexing into an inverted index  
Status: **PASS**  
Evidence: `ir_system/indexer.py` builds postings + documents tables in SQLite.

2. Implement three retrieval models: structured data search, Vector Space Model, and BM25  
Status: **PARTIAL / FAIL**  
Evidence:
- BM25: implemented (`search_bm25`)
- Vector Space (TF-IDF cosine): implemented (`search_tfidf_cosine`)
- Structured-data search: **not implemented** (no dedicated structured-field retrieval model in CLI/code)

3. Evaluate the 6 assignment queries for each of the 3 models  
Status: **FAIL**  
Evidence:
- Assignment queries are:
  1) `to be, or not to be`
  2) `English Grammar`
  3) `Philip K Dick`
  4) `Jabberwocky`
  5) `Gutenberg`
  6) `Dornröschen`
- Current `config/queries.txt` contains different placeholder queries.

4. Submit top results (max 100/query/model) as TSV files named `<query-nr>_<model-name>.tsv`  
Status: **FAIL**  
Evidence:
- No TSV files found in `results/`.
- Current evaluation produces one text summary file: `results/evaluation_summary.txt`.

5. TSV columns required: rank, book id, score, preview(optional), start line(optional)  
Status: **FAIL**  
Evidence: TSV outputs are not generated.

6. Retrieval mechanisms implemented by you (no external IR engines like Lucene/Solr/Elastic/Terrier etc.)  
Status: **PASS**  
Evidence: code uses Python stdlib + SQLite only; no prohibited IR framework found.

7. Report submission requirements (ACM template, 5-8 pages, required sections)  
Status: **NOT VALIDATED (artifact missing)**  
Evidence:
- No report PDF found in repo except `data/Assignment_1.pdf` (the assignment sheet).

8. Query results + implementation submitted separately on Loop  
Status: **NOT VALIDATED (outside repo scope)**

## Additional technical validation notes

- Data immutability design is good: output goes to `processed/` and `results/`; no code writes to `data/`.
- Memory-safe indexing is implemented via batching + partial indexes + merge.
- Current run manifest shows a smoke test subset only:
  - `processed/stats/run_manifest.json` -> `docs: 300`, `batch_size: 100`
- For assignment-grade output, you likely need a full-corpus run (or explicitly justify sampling in report).

## Required actions before submission

1. Add third retrieval model (structured-data search).
2. Replace `config/queries.txt` with the exact 6 assignment queries.
3. Generate 18 TSV files (`6 queries x 3 models`) with required naming format.
4. Ensure each TSV has at most 100 rows and required columns.
5. Run on intended dataset scope and document settings/results.
6. Prepare final ACM-format report (5-8 pages) with required sections.

## Suggested TSV naming examples

- `1_bm25.tsv`
- `1_tfidf.tsv`
- `1_structured.tsv`
- ...
- `6_structured.tsv`

