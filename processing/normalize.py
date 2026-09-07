from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text, read_json, stable_json_hash, write_json
from ..core.models import BlockRecord, BookRecord, PageRecord


def _page_from_dict(payload: dict[str, Any]) -> PageRecord:
    blocks = [BlockRecord(**block) for block in payload.get("blocks", [])]
    payload = dict(payload)
    payload["blocks"] = blocks
    return PageRecord(**payload)


def _item_page(item: dict[str, Any], fallback: int) -> int:
    for key in ("page_idx", "page_index", "page", "page_no"):
        value = item.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_idx" else value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            return number + 1 if key == "page_idx" else number
    return fallback


def _bbox(item: dict[str, Any]) -> list[float]:
    value = item.get("bbox") or item.get("box") or item.get("poly")
    if isinstance(value, list):
        flat: list[float] = []
        for part in value:
            if isinstance(part, list):
                flat.extend(float(x) for x in part if isinstance(x, (int, float)))
            elif isinstance(part, (int, float)):
                flat.append(float(part))
        return flat[:4]
    return []


def _block_type(raw: str, cfg: dict[str, Any]) -> str:
    return str(cfg["block_types"].get(raw.lower(), cfg["block_types"]["unknown"]))


def _item_text(item: dict[str, Any]) -> str:
    keys = (
        "text",
        "content",
        "md_content",
        "markdown",
        "caption",
        "img_caption",
        "image_caption",
        "table_caption",
        "image_footnote",
        "latex",
        "html",
    )
    return clean_text(" ".join(clean_text(item.get(key)) for key in keys))


def _iter_image_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"img_path", "image_path", "table_img_path", "figure_path", "path", "file_path"}:
                text = clean_text(child)
                if text:
                    refs.append(text)
            else:
                refs.extend(_iter_image_refs(child))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_iter_image_refs(item))
    return refs


def _item_image_ref(item: dict[str, Any]) -> str:
    refs = _iter_image_refs(item)
    return refs[0] if refs else ""


def _has_caption_text(item: dict[str, Any]) -> bool:
    return bool(clean_text(item.get("image_caption") or item.get("img_caption") or item.get("caption") or item.get("table_caption")))


def _resolve_extracted_image(image_ref: str, image_map: dict[str, str]) -> str:
    if not image_ref:
        return ""
    if image_ref in image_map:
        return image_map[image_ref]
    normalized = image_ref.replace("\\", "/")
    for key, value in image_map.items():
        key_normalized = key.replace("\\", "/")
        if key_normalized == normalized or Path(key_normalized).name == Path(normalized).name:
            return value
    return ""


def _can_reuse_normalized(
    page: PageRecord,
    image_map: dict[str, str],
    cfg: dict[str, Any],
    mineru_content_hash: str,
) -> bool:
    if page.mineru_content_hash != mineru_content_hash:
        return False
    if not image_map:
        return True
    visual_types = set(cfg["routing"]["visual_block_types"])
    visual_blocks = [block for block in page.blocks if block.block_type in visual_types]
    if not visual_blocks:
        return True
    return any(block.extracted_image_path and block.text for block in visual_blocks)


def normalize_pages(
    book: BookRecord,
    content: list[dict[str, Any]],
    mineru_parse_path: str,
    image_map: dict[str, str],
    cfg: dict[str, Any],
) -> list[PageRecord]:
    output_dir = Path(book.output_dir)
    grouped: dict[int, list[dict[str, Any]]] = {}
    mineru_content_hash = stable_json_hash(content)
    fallback_page = 1
    for item in content:
        page_no = _item_page(item, fallback_page)
        grouped.setdefault(page_no, []).append(item)
        fallback_page = page_no

    total_pages = max(book.page_count, max(grouped.keys(), default=0))
    pages: list[PageRecord] = []
    for page_no in range(1, total_pages + 1):
        chapter_no = cfg["default_chapter_no"]
        page_image = cfg["paths"]["page_images"].format(chapter_no=chapter_no, page_no=page_no)
        relative = cfg["paths"]["normalized_page"].format(chapter_no=chapter_no, page_no=page_no)
        normalized_path = output_dir / relative
        if bool(cfg["runtime"].get("reuse_normalized")) and normalized_path.exists():
            cached_page = _page_from_dict(read_json(normalized_path))
            if _can_reuse_normalized(cached_page, image_map, cfg, mineru_content_hash):
                pages.append(cached_page)
                continue

        blocks: list[BlockRecord] = []
        for index, item in enumerate(grouped.get(page_no, []), start=1):
            raw_type = clean_text(item.get("type")) or cfg["block_types"]["unknown"]
            block = BlockRecord(
                block_id=f"block_{index:03d}",
                block_type=_block_type(raw_type, cfg),
                bbox=_bbox(item),
                text=_item_text(item),
                markdown=clean_text(item.get("markdown") or item.get("md_content")),
                reading_order=index,
                confidence=float(item.get("confidence") or item.get("score") or 0.0),
                extracted_image_path=_resolve_extracted_image(_item_image_ref(item), image_map),
            )
            blocks.append(block)
        full_text = "\n".join(block.text for block in blocks if block.text)
        page = PageRecord(
            book_id=book.book_id,
            book_name=book.book_name,
            source_pdf=book.source_pdf,
            chapter_no=chapter_no,
            chapter_title="",
            page_index=page_no,
            page_label=str(page_no),
            page_image=page_image,
            width=0,
            height=0,
            blocks=blocks,
            full_text=full_text,
            tables=[asdict(block) for block in blocks if block.block_type == "table"],
            figures=[asdict(block) for block in blocks if block.block_type == "figure"],
            formulas=[asdict(block) for block in blocks if block.block_type == "formula"],
            prev_page=page_no - 1 if page_no > 1 else None,
            next_page=page_no + 1 if page_no < total_pages else None,
            mineru_parse_path=mineru_parse_path,
            mineru_content_hash=mineru_content_hash,
        )
        page.normalized_page_path = relative
        write_json(normalized_path, asdict(page))
        pages.append(page)
    return pages
