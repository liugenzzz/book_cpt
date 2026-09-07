from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BookRecord:
    book_id: str
    book_name: str
    source_pdf: str
    output_dir: str
    page_count: int
    file_size: int
    file_hash: str
    status: str
    created_at: str
    pipeline_version: str


@dataclass
class BlockRecord:
    block_id: str
    block_type: str
    bbox: list[float] = field(default_factory=list)
    text: str = ""
    markdown: str = ""
    reading_order: int = 0
    confidence: float = 0.0
    image_path: str = ""
    extracted_image_path: str = ""


@dataclass
class PageRecord:
    book_id: str
    book_name: str
    source_pdf: str
    chapter_no: str
    chapter_title: str
    page_index: int
    page_label: str
    page_image: str
    width: int
    height: int
    blocks: list[BlockRecord] = field(default_factory=list)
    full_text: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    figure_links: list[dict[str, Any]] = field(default_factory=list)
    prev_page: int | None = None
    next_page: int | None = None
    mineru_parse_path: str = ""
    mineru_content_hash: str = ""
    normalized_page_path: str = ""


@dataclass(frozen=True)
class SampleJob:
    task_type: str
    book: BookRecord
    page: PageRecord
    block: BlockRecord | None = None
    page_window: list[PageRecord] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineOptions:
    config_path: Path | None = None
    input_dir: Path | None = None
    input_books: list[dict[str, str]] | None = None
    output_root: Path | None = None
    cache_only: bool | None = None
    reuse_mineru: bool | None = None
    skip_vlm: bool | None = None
    book_workers: int | None = None
    max_workers: int | None = None
    page_workers: int | None = None
    crop_workers: int | None = None
    mineru_workers: int | None = None
    mineru_retry_count: int | None = None
    mineru_min_page_coverage: float | None = None
    reuse_normalized: bool | None = None
    reuse_samples: bool | None = None
    reuse_exports: bool | None = None
    force_rebuild: bool | None = None
    min_page_text_chars: int | None = None
    min_block_text_chars: int | None = None
    task_types: list[str] | None = None
    limit_books: int | None = None
