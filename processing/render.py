from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import append_jsonl
from ..core.models import BookRecord


# 每个 worker 进程只开一次 PDF，之后复用。原来是每渲染一页 fitz.open 一次整本：
# N 页就重新解析 N 遍文档结构，页树有瑕疵的书还会每次都往 stderr 刷一条
# "MuPDF error: format error: too many kids in page tree"，几百页就是几百行。
_DOCUMENT_CACHE: dict[str, Any] = {}


def silence_mupdf_chatter(enabled: bool = True) -> None:
    """关掉 MuPDF 直写 stderr 的逐条报错。

    这些字符串是 C 库直接写文件描述符 2 的，不走 Python logging，
    configure_logger 里那套按 logger 名压制的办法对它完全无效。
    """
    if not enabled:
        return
    try:
        import fitz  # type: ignore
    except ImportError:
        return
    tools = getattr(fitz, "TOOLS", None)
    for name in ("mupdf_display_errors", "mupdf_display_warnings"):
        toggle = getattr(tools, name, None)
        if callable(toggle):
            try:
                toggle(False)
            except Exception:
                pass


def drain_mupdf_warnings() -> str:
    """取出并清空 MuPDF 攒下的报错，交给调用方打一条日志。"""
    try:
        import fitz  # type: ignore
    except ImportError:
        return ""
    getter = getattr(getattr(fitz, "TOOLS", None), "mupdf_warnings", None)
    if not callable(getter):
        return ""
    try:
        return str(getter() or "").strip()
    except Exception:
        return ""


def _open_document(source_pdf: str, cfg: dict[str, Any]) -> Any:
    document = _DOCUMENT_CACHE.get(source_pdf)
    if document is not None:
        return document
    import fitz  # type: ignore

    silence_mupdf_chatter(bool(cfg.get("render", {}).get("silence_mupdf_errors", True)))
    document = fitz.open(source_pdf)
    _DOCUMENT_CACHE[source_pdf] = document
    return document


def _render_page_worker(args: tuple[str, str, dict[str, Any], int, dict[str, Any]]) -> tuple[str, dict[str, Any] | None, str, int]:
    source_pdf, output_dir_text, cfg, page_index, book_payload = args
    page_no = page_index + 1
    try:
        import fitz  # type: ignore
    except ImportError:
        return "", None, "missing_fitz", page_no

    output_dir = Path(output_dir_text)
    relative = cfg["paths"]["page_images"].format(chapter_no=cfg["default_chapter_no"], page_no=page_no)
    target = output_dir / relative
    if target.exists() and not bool(cfg["runtime"].get("rerender_pages")):
        return relative, None, "skipped", page_no

    scale = float(cfg["render"]["dpi"]) / 72.0
    target.parent.mkdir(parents=True, exist_ok=True)
    document = _open_document(source_pdf, cfg)
    page = document.load_page(page_index)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pixmap.save(str(target))
    return relative, {**book_payload, "page_index": page_no, "page_label": str(page_no), "page_image": relative}, "rendered", page_no


def render_pages(book: BookRecord, cfg: dict[str, Any]) -> list[str]:
    output_dir = Path(book.output_dir)
    logger = logging.getLogger(f"{cfg['logger_name']}.{book.book_id}")
    rendered: list[str] = []
    if not cfg["runtime"]["render_pages"]:
        logger.info("render_pages disabled book_id=%s", book.book_id)
        return rendered
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.warning("render_pages skipped book_id=%s reason=missing_pymupdf", book.book_id)
        return rendered

    silence_mupdf_chatter(bool(cfg.get("render", {}).get("silence_mupdf_errors", True)))
    with fitz.open(book.source_pdf) as document:
        page_count = document.page_count
    # 页树/交叉引用表有瑕疵的书，报错在这里一次性汇总成一条日志，
    # 而不是让 MuPDF 在每次打开时往 stderr 刷一遍。
    complaints = drain_mupdf_warnings()
    if complaints:
        logger.warning(
            "pdf structure complaints book_id=%s count=%s detail=%s",
            book.book_id,
            len(complaints.splitlines()),
            "; ".join(dict.fromkeys(complaints.splitlines()))[:300],
        )

    page_workers = max(1, int(cfg["runtime"].get("page_workers", 1)))
    jobs = [(book.source_pdf, str(output_dir), cfg, page_index, asdict(book)) for page_index in range(page_count)]
    page_dir = (output_dir / Path(
        cfg["paths"]["page_images"].format(chapter_no=cfg["default_chapter_no"], page_no=1)
    ).parent).resolve()
    logger.info(
        "render_pages start book_id=%s pages=%s workers=%s output_dir=%s",
        book.book_id,
        page_count,
        page_workers,
        page_dir,
    )
    if page_workers == 1 or len(jobs) <= 1:
        results = []
        for job in jobs:
            result = _render_page_worker(job)
            results.append(result)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=page_workers) as executor:
            futures = [executor.submit(_render_page_worker, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    results.append(future.result())
            except KeyboardInterrupt:
                # 所有页任务是一次性全提交的，不主动取消的话 with 退出时的
                # shutdown(wait=True) 会把几百个排队任务全跑完才罢休，
                # Ctrl-C 之后还得刷屏好几分钟。
                logger.warning("render_pages interrupted book_id=%s", book.book_id)
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    rendered_count = 0
    skipped_count = 0
    # 按页码排序，不是按相对路径字符串排；路径为空的失败项会被排到最前，
    # 顺带把 rendered 列表的顺序打乱。
    for relative, manifest_row, status, _ in sorted(results, key=lambda item: item[3]):
        if relative:
            rendered.append(relative)
        if status == "rendered":
            rendered_count += 1
        elif status == "skipped":
            skipped_count += 1
        if manifest_row is not None:
            append_jsonl(output_dir / cfg["paths"]["pages_manifest"], manifest_row)
    logger.info(
        "render_pages completed book_id=%s pages=%s rendered=%s skipped=%s output_dir=%s",
        book.book_id,
        page_count,
        rendered_count,
        skipped_count,
        page_dir,
    )
    return rendered
