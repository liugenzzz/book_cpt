# Cache-only Domain Corpus Design

## Goal

Allow the pipeline to generate `domain_knowledge_corpus` from an existing output tree when the source PDFs no longer exist. The command must require only `--output-root`, select cached books from their per-book manifests, and never contact MinerU when cache-only mode is active.

## Chosen approach

Add a first-class `--cache-only` pipeline option. This was chosen over the two alternatives discussed with the user: recreating placeholder PDFs, which is brittle and writes misleading source metadata, and a standalone recovery script, which would duplicate pipeline behavior and be harder to maintain.

## Command contract

```bash
python app/cli.py \
  --output-root outputs_test \
  --task domain_knowledge_corpus \
  --cache-only
```

`--cache-only` implies:

- `reuse_mineru = true`
- `reuse_normalized = true`
- `render_pages = false`
- `crop_blocks = false`

It does not force `skip_vlm`, because cache-only mode can remain compatible with other task types whose required image files are already cached. `domain_knowledge_corpus` itself is rule-based and does not call the VLM.

`--cache-only` rejects `--input-dir`, `--book`, and `--force-rebuild` because those combinations have contradictory meanings.

## Cached-book discovery

The pipeline scans immediate child directories of `--output-root` for `manifest.jsonl`. For each manifest it selects the last valid object, restores `BookRecord`, and replaces the stored `output_dir` with the directory currently being scanned. The stale `source_pdf` value is retained only for provenance and is never opened in cache-only mode.

If no cached books are found, or a discovered manifest has no valid `BookRecord`, the command raises a path-specific error instead of returning an unexplained empty list.

## Strict MinerU behavior

Each cached book must contain:

```text
mineru/raw/<book_id>.json
mineru/parsed/<book_id>.json
mineru/image_map.json
```

The existing structural and coverage validation still applies. If any file is missing or invalid, cache-only mode stops that book with a clear error. It must never fall through to `MinerUClient.parse_pdf`, because the PDF intentionally does not exist.

## Data flow

```text
output-root/*/manifest.jsonl
  -> cached BookRecord
  -> validated MinerU cache
  -> reuse/regenerate normalized pages
  -> build domain corpus jobs
  -> rule-based PT cleaning
  -> validation/deduplication
  -> exports/pt/domain_knowledge_corpus.jsonl
```

Page rendering and block cropping remain in the normal pipeline call sequence but return immediately because their runtime flags are disabled. This keeps the implementation small and preserves existing behavior outside cache-only mode.

## Error handling

- No `manifest.jsonl` below the output root: raise `FileNotFoundError` describing the expected layout.
- Malformed or incomplete manifest: raise `ValueError` naming the manifest.
- Missing MinerU cache member: raise `FileNotFoundError` listing missing paths.
- Invalid MinerU JSON/content: raise `MinerUParseIncompleteError` and do not call MinerU.
- Per-book processing retains the pipeline's existing result-object error reporting.

## Tests

Tests cover:

1. CLI parsing and incompatible argument rejection.
2. Restoring a cached book whose PDF and historic output path do not exist.
3. Cache-only runtime overrides.
4. Strict failure without a complete MinerU cache and proof that MinerU is not called.
5. An integration run that starts with only manifest/MinerU cache, has no PDF, and produces a non-empty PT export.
6. The existing full test suite to protect normal PDF-based operation.

## Scope exclusions

- Recovering metadata when `manifest.jsonl` is missing.
- Reconstructing missing MinerU files.
- Changing normal PDF ingestion behavior.
- Adding recursive cache discovery beneath per-book directories.
