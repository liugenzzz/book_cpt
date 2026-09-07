# Cache-only Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate cached tasks, especially `domain_knowledge_corpus`, from `--output-root` without any source PDF.

**Architecture:** Add cache-only state to the CLI and `PipelineOptions`, discover `BookRecord` values from per-book manifests, and reuse the existing processing pipeline with render/crop disabled. Make MinerU reuse strict in this mode so missing or invalid cache raises instead of falling back to the PDF-backed service.

**Tech Stack:** Python 3, `argparse`, dataclasses, pathlib, unittest.

---

### Task 1: Cached-book discovery

**Files:**
- Modify: `processing/ingest.py`
- Create: `tests/test_pipeline_cache_only.py`

- [ ] **Step 1: Write the failing discovery tests**

Add tests that create `outputs/<book>/manifest.jsonl` with a nonexistent `source_pdf`, call `scan_cached_books`, and assert that the `BookRecord` is restored while `output_dir` is rebound to the current cache directory. Add tests for no manifests and a manifest without a complete book record.

```python
books = scan_cached_books(output_root, cfg)
self.assertEqual(books[0].book_id, "cached-book")
self.assertEqual(Path(books[0].output_dir), book_dir)
self.assertFalse(Path(books[0].source_pdf).exists())
```

- [ ] **Step 2: Run the discovery tests and verify RED**

Run from the project parent:

```powershell
python -m unittest book_cpt.tests.test_pipeline_cache_only.CacheOnlyDiscoveryTests -v
```

Expected: import failure because `scan_cached_books` does not exist.

- [ ] **Step 3: Implement minimal discovery**

Add `scan_cached_books(output_root, cfg)` that scans immediate directories containing the configured manifest path, selects the last JSON object containing all `BookRecord` fields, reconstructs the dataclass, and overrides `output_dir` with the current directory. Raise path-specific `FileNotFoundError`/`ValueError` rather than return an unexplained empty list.

- [ ] **Step 4: Run the discovery tests and verify GREEN**

Run the command from Step 2. Expected: all discovery tests pass.

### Task 2: Option and CLI contract

**Files:**
- Modify: `core/models.py`
- Modify: `config.py`
- Modify: `app/cli.py`
- Modify: `app/pipeline.py`
- Modify: `tests/test_pipeline_cache_only.py`

- [ ] **Step 1: Write failing option tests**

Test that `build_parser()` accepts `--cache-only`, `_apply_options` forces MinerU/normalized reuse and disables render/crop, and cache-only rejects input/PDF and force-rebuild options.

```python
args = build_parser().parse_args(["--output-root", "outputs", "--cache-only"])
self.assertTrue(args.cache_only)
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest book_cpt.tests.test_pipeline_cache_only.CacheOnlyOptionTests -v
```

Expected: parser/model failures because `cache_only` is not defined.

- [ ] **Step 3: Implement the option**

Add `cache_only: bool | None = None` to `PipelineOptions`, a false runtime default, and `--cache-only` to the parser. Pass it into `PipelineOptions`; validate contradictory options before loading books. When enabled, apply:

```python
cfg["runtime"].update(
    cache_only=True,
    reuse_mineru=True,
    reuse_normalized=True,
    render_pages=False,
    crop_blocks=False,
)
```

In `run_pipeline`, call `scan_cached_books(output_root, cfg)` before the existing PDF/input-book branches.

- [ ] **Step 4: Run and verify GREEN**

Run the command from Step 2. Expected: all option tests pass.

### Task 3: Strict MinerU cache mode

**Files:**
- Modify: `services/mineru.py`
- Modify: `tests/test_pipeline_cache_only.py`

- [ ] **Step 1: Write failing strict-cache tests**

Construct a cached `BookRecord` with a missing raw/parsed/map member, patch `MinerUClient.parse_pdf`, call `parse_book_with_mineru`, and assert a cache-specific exception plus zero client calls. Add a valid-cache test with a nonexistent source PDF.

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest book_cpt.tests.test_pipeline_cache_only.CacheOnlyMinerUTests -v
```

Expected: the current implementation falls through to `MinerUClient.parse_pdf`.

- [ ] **Step 3: Implement strict cache handling**

Reuse the existing cache loading and validation branch. If runtime `cache_only` is true, report all missing cache paths before any client construction; if JSON/content validation fails, raise `MinerUParseIncompleteError` with the book ID and preserve the original exception as the cause. Do not change fallback behavior for normal runs.

- [ ] **Step 4: Run and verify GREEN**

Run the command from Step 2. Expected: all strict-cache tests pass and the MinerU mock has no calls.

### Task 4: End-to-end no-PDF generation

**Files:**
- Modify: `tests/test_pipeline_cache_only.py`

- [ ] **Step 1: Write the failing integration test**

Create only a manifest, `mineru/raw/<book_id>.json`, `mineru/parsed/<book_id>.json`, and `mineru/image_map.json`. Use a nonexistent `source_pdf`, sufficiently long parsed page text, and run:

```python
results = run_pipeline(
    PipelineOptions(
        output_root=output_root,
        cache_only=True,
        skip_vlm=True,
        book_workers=1,
        max_workers=1,
        task_types=["domain_knowledge_corpus"],
    )
)
```

Assert that the result contains no error and `exports/pt/domain_knowledge_corpus.jsonl` contains at least one `{ "text": ... }` row.

- [ ] **Step 2: Run and verify RED**

Run the integration test alone. Expected: failure until cache discovery, option propagation, and strict reuse are connected.

- [ ] **Step 3: Make the smallest connection fixes**

Ensure cache-only discovery happens before PDF scanning and that render/crop observe their disabled runtime flags. Do not add a second generation implementation.

- [ ] **Step 4: Run and verify GREEN**

Run the integration test alone. Expected: non-empty PT JSONL with no PDF access.

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the command and cache requirements**

Add a cache-only subsection with:

```bash
python app/cli.py --output-root outputs_test --task domain_knowledge_corpus --cache-only
```

Document `manifest.jsonl`, the three MinerU files, the absence of PDF requirements, strict no-fallback behavior, and incompatible options.

- [ ] **Step 2: Run focused tests**

```powershell
python -m unittest book_cpt.tests.test_pipeline_cache_only -v
```

Expected: all cache-only tests pass.

- [ ] **Step 3: Run the complete suite**

```powershell
python -m unittest discover -s book_cpt/tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 4: Review the final diff and requirements**

Confirm the command needs no input directory, normal PDF behavior remains unchanged, cache-only never calls MinerU, errors identify missing cache files, and README matches implementation.

No commit step is included because the supplied workspace is not a Git repository.
