from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..core.io_utils import file_sha256, safe_name, stable_json_hash, utc_now
from ..core.models import BookRecord


def page_count(pdf_path: Path, cfg: dict[str, Any]) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as document:
                return int(document.page_count)
        except Exception:
            pattern = str(cfg["pdf_page_regex"]).encode("ascii")
            return len(re.findall(pattern, pdf_path.read_bytes()))


def _book_id_for(pdf_path: Path, book_name: str, cfg: dict[str, Any]) -> str:
    configured_ids = cfg.get("book_ids") if isinstance(cfg.get("book_ids"), dict) else {}
    configured = configured_ids.get(pdf_path.name) or configured_ids.get(book_name) or configured_ids.get(f"{book_name}.pdf")
    return configured or safe_name(book_name, str(cfg.get("unknown_book_id", "untitled_book")))


def build_book_record(
    *,
    pdf_path: Path,
    output_root: Path,
    cfg: dict[str, Any],
    book_name: str | None = None,
    book_id: str | None = None,
) -> BookRecord:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    resolved_name = (book_name or pdf_path.stem).strip() or pdf_path.stem
    resolved_id = (book_id or _book_id_for(pdf_path, resolved_name, cfg)).strip()
    book_name_safe = safe_name(resolved_name, resolved_id)
    output_dir = output_root / book_name_safe
    return BookRecord(
        book_id=resolved_id,
        book_name=resolved_name,
        source_pdf=str(pdf_path),
        output_dir=str(output_dir),
        page_count=page_count(pdf_path, cfg),
        file_size=pdf_path.stat().st_size,
        file_hash=file_sha256(pdf_path, int(cfg["hash_chunk_size"])),
        status=cfg["statuses"]["ready"],
        created_at=utc_now(cfg["timestamp_format"]),
        pipeline_version=cfg["pipeline_version"],
    )


def scan_books(input_dir: Path, output_root: Path, cfg: dict[str, Any]) -> list[BookRecord]:
    pattern = f"**/*{cfg['pdf_suffix']}" if bool(cfg["runtime"]["recursive"]) else f"*{cfg['pdf_suffix']}"
    books: list[BookRecord] = []
    for pdf_path in sorted(input_dir.glob(pattern)):
        if not pdf_path.is_file():
            continue
        books.append(build_book_record(pdf_path=pdf_path, output_root=output_root, cfg=cfg))
    return books


def scan_input_books(input_books: list[dict[str, str]], output_root: Path, cfg: dict[str, Any]) -> list[BookRecord]:
    books: list[BookRecord] = []
    for item in input_books:
        raw_path = item.get("path") or item.get("source_pdf") or item.get("address") or item.get("url")
        if not raw_path:
            raise ValueError(f"Missing book path in input book item: {item}")
        books.append(
            build_book_record(
                pdf_path=Path(raw_path),
                output_root=output_root,
                cfg=cfg,
                book_name=item.get("book_name") or item.get("name"),
                book_id=item.get("book_id"),
            )
        )
    return books


def _cached_book_from_manifest(manifest_path: Path) -> BookRecord:
    required = tuple(field.name for field in fields(BookRecord))
    latest: dict[str, Any] | None = None
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and all(name in payload for name in required):
            latest = payload
    if latest is None:
        raise ValueError(f'No complete cached BookRecord found in manifest: {manifest_path}')
    values = {name: latest[name] for name in required}
    values['output_dir'] = str(manifest_path.parent)
    return BookRecord(**values)


def _book_page_from_mineru_item(item: dict[str, Any], fallback: int) -> int:
    for key in ("page_idx", "page_index", "page", "page_no"):
        value = item.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_idx" else value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            return number + 1 if key == "page_idx" else number
    return fallback


def _infer_mineru_page_count(content: list[dict[str, Any]]) -> int:
    fallback = 1
    page_count = 0
    for item in content:
        page_no = _book_page_from_mineru_item(item, fallback)
        fallback = page_no
        page_count = max(page_count, page_no)
    return page_count


def _path_template_parent(cfg: dict[str, Any], key: str, default: str) -> Path:
    template = str(cfg.get('paths', {}).get(key, default))
    return Path(template.format(book_id='__book_id__')).parent


def _cached_book_from_mineru_dir(book_dir: Path, cfg: dict[str, Any]) -> BookRecord:
    paths_cfg = cfg.get('paths', {})
    raw_dir = book_dir / _path_template_parent(cfg, 'mineru_raw', 'mineru/raw/{book_id}.json')
    parsed_dir = book_dir / _path_template_parent(cfg, 'mineru_parsed', 'mineru/parsed/{book_id}.json')
    image_map_path = book_dir / str(paths_cfg.get('image_map', 'mineru/image_map.json'))

    missing = [path for path in (raw_dir, parsed_dir, image_map_path) if not path.exists()]
    if missing:
        missing_text = ', '.join(str(path) for path in missing)
        raise FileNotFoundError(f'Incomplete MinerU cache in {book_dir}; missing: {missing_text}')

    raw_files = {path.stem: path for path in raw_dir.glob('*.json') if path.is_file()}
    parsed_files = {path.stem: path for path in parsed_dir.glob('*.json') if path.is_file()}
    book_ids = sorted(set(raw_files) & set(parsed_files))
    if not book_ids:
        raise FileNotFoundError(
            f'Incomplete MinerU cache in {book_dir}; expected matching raw and parsed JSON files'
        )
    if len(book_ids) > 1:
        raise ValueError(
            f'Ambiguous MinerU cache in {book_dir}; expected one book id, found: {", ".join(book_ids)}'
        )

    book_id = book_ids[0]
    try:
        content = json.loads(parsed_files[book_id].read_text(encoding='utf-8'))
    except Exception as exc:
        raise ValueError(
            f'cache-only MinerU parsed cache invalid: {parsed_files[book_id]}: {exc}'
        ) from exc
    if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
        raise ValueError(f'MinerU parsed cache is not a list of objects: {parsed_files[book_id]}')

    source_pdf = book_dir / f'{book_id}{cfg.get("pdf_suffix", ".pdf")}'
    file_size = raw_files[book_id].stat().st_size + parsed_files[book_id].stat().st_size + image_map_path.stat().st_size
    return BookRecord(
        book_id=book_id,
        book_name=book_dir.name,
        source_pdf=str(source_pdf),
        output_dir=str(book_dir),
        page_count=_infer_mineru_page_count(content),
        file_size=file_size,
        file_hash=stable_json_hash(content),
        status=str(cfg.get('statuses', {}).get('ready', 'ready')),
        created_at=utc_now(str(cfg.get('timestamp_format', '%Y-%m-%dT%H:%M:%SZ'))),
        pipeline_version=str(cfg.get('pipeline_version', '')),
    )


def scan_cached_books(output_root: Path, cfg: dict[str, Any]) -> list[BookRecord]:
    manifest_relative = Path(cfg['paths']['manifest'])
    books: list[BookRecord] = []
    if output_root.exists():
        for book_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
            manifest_path = book_dir / manifest_relative
            if manifest_path.is_file():
                books.append(_cached_book_from_manifest(manifest_path))
                continue
            mineru_dir = book_dir / 'mineru'
            if mineru_dir.exists():
                books.append(_cached_book_from_mineru_dir(book_dir, cfg))
    if not books:
        expected_manifest = output_root / '*' / manifest_relative
        expected_mineru = output_root / '*' / 'mineru'
        raise FileNotFoundError(
            f'No cached books found; expected manifests matching: {expected_manifest} '
            f'or MinerU caches under: {expected_mineru}'
        )
    return books
