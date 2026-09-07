from __future__ import annotations

import re
from dataclasses import asdict

from ..core.io_utils import clean_text
from ..core.models import BlockRecord, BookRecord, PageRecord, SampleJob


def _enabled(enabled_tasks: dict[str, bool], task_type: str) -> bool:
    return bool(enabled_tasks.get(task_type, False))


def _page_window(pages: list[PageRecord], start_index: int, window_size: int) -> list[PageRecord]:
    end = min(len(pages), start_index + max(1, window_size))
    return pages[start_index:end]


def _nearby_blocks(page: PageRecord, block: BlockRecord, radius: int = 4) -> list[BlockRecord]:
    try:
        pos = page.blocks.index(block)
    except ValueError:
        return []
    end = min(len(page.blocks), pos + radius + 1)
    return [item for item in page.blocks[pos + 1 : end] if clean_text(item.text)]


def _block_payload(block: BlockRecord) -> dict[str, object]:
    return asdict(block)


def _page_payload(page: PageRecord, *, include_blocks: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "page_index": page.page_index,
        "page_label": page.page_label,
        "chapter_no": page.chapter_no,
        "chapter_title": page.chapter_title,
        "page_image": page.page_image,
        "full_text": page.full_text,
    }
    if include_blocks:
        payload["blocks"] = [asdict(block) for block in page.blocks]
    return payload


def _text_blocks(page: PageRecord, text_types: set[str], min_chars: int) -> list[BlockRecord]:
    return [block for block in page.blocks if block.block_type in text_types and len(clean_text(block.text)) >= min_chars]


def _visual_blocks(page: PageRecord, visual_types: set[str]) -> list[BlockRecord]:
    return [block for block in page.blocks if block.block_type in visual_types]


def _window_text(page_window: list[PageRecord]) -> str:
    return "\n\n".join(clean_text(page.full_text) for page in page_window if clean_text(page.full_text))


def _chapter_path(page_window: list[PageRecord]) -> str:
    titles: list[str] = []
    for page in page_window:
        title = clean_text(page.chapter_title)
        if title and title not in titles:
            titles.append(title)
    if titles:
        return " / ".join(titles)
    if not page_window:
        return ""
    start = clean_text(page_window[0].page_label) or str(page_window[0].page_index)
    end = clean_text(page_window[-1].page_label) or str(page_window[-1].page_index)
    return f"页码 {start}" if start == end else f"页码 {start}-{end}"


def _looks_like_low_value_text(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return True
    if re.fullmatch(r"(第\s*)?\d{1,5}\s*(页)?", text):
        return True
    if re.fullmatch(r"[-—_=·•●○\s]{3,}", text):
        return True
    if re.match(r"^(目录|目\s*录|Contents|版权|版权页|出版说明|前言|索引|Index|参考文献|References)\b", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^(ISBN|版权所有|著作权|责任编辑|出版社|印刷|开本|版次|印次|定价|CIP)", text, flags=re.IGNORECASE):
        return True
    return False


def _low_value_page(page: PageRecord) -> bool:
    text = clean_text(page.full_text)
    if not text:
        return True
    if len(text) < 80 and _looks_like_low_value_text(text):
        return True
    low_value_hits = re.findall(r"(目录|目\s*录|Contents|版权|索引|Index|参考文献|References)", text, flags=re.IGNORECASE)
    return bool(low_value_hits) and len(text) < 1200


def _domain_corpus_raw_text(page_window: list[PageRecord], skip_types: set[str]) -> str:
    pages_text: list[str] = []
    for page in page_window:
        if _low_value_page(page):
            continue
        lines: list[str] = []
        page_label = clean_text(page.page_label) or str(page.page_index)
        lines.append(f"[页码：{page_label}]")
        for block in page.blocks:
            if block.block_type in skip_types:
                continue
            text = clean_text(block.markdown or block.text)
            if _looks_like_low_value_text(text):
                continue
            if text:
                lines.append(text)
        text = "\n".join(lines).strip()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _build_domain_corpus_jobs(book: BookRecord, pages: list[PageRecord], cfg: dict) -> list[SampleJob]:
    routing = cfg["routing"]
    min_chars = int(routing.get("min_domain_corpus_text_chars", routing.get("min_page_text_chars", 200)))
    target_chars = int(routing.get("domain_corpus_target_input_chars", 3600))
    max_pages = max(1, int(routing.get("domain_corpus_window", 2)))
    skip_types = {"header", "footer", "page_number", "unknown"}

    jobs: list[SampleJob] = []
    window: list[PageRecord] = []
    window_chars = 0

    def emit() -> None:
        nonlocal window, window_chars
        if not window:
            return
        raw_text = _domain_corpus_raw_text(window, skip_types)
        if len(clean_text(raw_text)) >= min_chars:
            jobs.append(
                SampleJob(
                    "domain_knowledge_corpus",
                    book,
                    window[0],
                    page_window=list(window),
                    source={
                        "book_name": book.book_name,
                        "chapter_path": _chapter_path(window),
                        "raw_text": raw_text,
                        "page_window": [_page_payload(page) for page in window],
                    },
                )
            )
        window = []
        window_chars = 0

    for page in pages:
        if _low_value_page(page):
            continue
        page_text = clean_text(page.full_text)
        if not page_text:
            continue
        window.append(page)
        window_chars += len(page_text)
        if window_chars >= target_chars or len(window) >= max_pages:
            emit()
    emit()
    return jobs


def build_sample_jobs(book: BookRecord, pages: list[PageRecord], cfg: dict) -> list[SampleJob]:
    jobs: list[SampleJob] = []
    routing = cfg["routing"]
    enabled = routing["enabled_tasks"]
    text_types = set(routing.get("text_block_types", ["text", "paragraph", "list"]))
    title_types = set(routing.get("title_block_types", ["title", "section_title"]))
    visual_types = set(routing.get("visual_block_types", ["figure", "table", "formula"]))
    min_page_text = int(routing["min_page_text_chars"])
    min_block_text = int(routing["min_block_text_chars"])
    min_chapter_text = int(routing.get("min_chapter_text_chars", 700))
    cross_window = int(routing.get("cross_page_window", 3))
    chapter_window = int(routing.get("chapter_window", 6))

    for index, page in enumerate(pages):
        page_text = clean_text(page.full_text)
        has_page_text = len(page_text) >= min_page_text
        text_blocks = _text_blocks(page, text_types, min_block_text)
        visual_blocks = _visual_blocks(page, visual_types)

        if has_page_text and _enabled(enabled, "page_to_structured_description"):
            jobs.append(
                SampleJob(
                    "page_to_structured_description",
                    book,
                    page,
                    images=[page.page_image],
                    source={"page": _page_payload(page)},
                )
            )

        if has_page_text and _enabled(enabled, "page_content_restatement"):
            jobs.append(
                SampleJob(
                    "page_content_restatement",
                    book,
                    page,
                    images=[page.page_image],
                    source={"page": _page_payload(page)},
                )
            )

        if has_page_text and _enabled(enabled, "evidence_to_conclusion_chain"):
            jobs.append(
                SampleJob(
                    "evidence_to_conclusion_chain",
                    book,
                    page,
                    images=[page.page_image],
                    source={"page": _page_payload(page)},
                )
            )

        if _enabled(enabled, "paragraph_summary"):
            for block in text_blocks:
                jobs.append(
                    SampleJob(
                        "paragraph_summary",
                        book,
                        page,
                        block=block,
                        source={"paragraph": _block_payload(block), "page": _page_payload(page, include_blocks=False)},
                    )
                )

        if _enabled(enabled, "title_body_alignment"):
            for block in page.blocks:
                if block.block_type not in title_types or not clean_text(block.text):
                    continue
                body_blocks = _nearby_blocks(page, block)
                if not body_blocks:
                    continue
                jobs.append(
                    SampleJob(
                        "title_body_alignment",
                        book,
                        page,
                        block=block,
                        source={
                            "title": _block_payload(block),
                            "body_blocks": [_block_payload(item) for item in body_blocks],
                            "page": _page_payload(page, include_blocks=False),
                        },
                    )
                )

        if _enabled(enabled, "chart_table_to_text"):
            for block in visual_blocks:
                image = block.extracted_image_path
                if not image:
                    continue
                nearby = _nearby_blocks(page, block)
                jobs.append(
                    SampleJob(
                        "chart_table_to_text",
                        book,
                        page,
                        block=block,
                        images=[image] if image else [],
                        source={
                            "target_block": _block_payload(block),
                            "nearby_context_blocks": [_block_payload(item) for item in nearby],
                            "page": _page_payload(page, include_blocks=False),
                        },
                    )
                )

        if _enabled(enabled, "cross_page_synthesis"):
            page_window = _page_window(pages, index, cross_window)
            if len(page_window) >= 2 and len(_window_text(page_window)) >= min_page_text:
                jobs.append(
                    SampleJob(
                        "cross_page_synthesis",
                        book,
                        page,
                        page_window=page_window,
                        images=[item.page_image for item in page_window],
                        source={"page_window": [_page_payload(item) for item in page_window]},
                    )
                )

        if _enabled(enabled, "chapter_key_conclusions"):
            page_window = _page_window(pages, index, chapter_window)
            if len(_window_text(page_window)) >= min_chapter_text:
                jobs.append(
                    SampleJob(
                        "chapter_key_conclusions",
                        book,
                        page,
                        page_window=page_window,
                        source={"chapter_fragment_pages": [_page_payload(item) for item in page_window]},
                    )
                )

    if _enabled(enabled, "domain_knowledge_corpus"):
        jobs.extend(_build_domain_corpus_jobs(book, pages, cfg))

    return jobs
