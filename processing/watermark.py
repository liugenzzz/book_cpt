from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..core.io_utils import file_sha256, read_json, stable_json_hash, write_json
from ..core.models import BookRecord


class UnreadablePdfError(RuntimeError):
    """PDF 本身损坏/截断，pypdf 打不开。

    这类文件不是流水线的问题，也没有重试价值，由上层直接跳过而不是记成失败。
    """


@dataclass(frozen=True)
class WatermarkCleanResult:
    source_pdf: str
    cleaned_pdf: str
    cleaned: bool
    candidate_names: list[str]
    removed_invocations: int
    page_count: int
    content_hash: str
    source_hash: str = ""
    config_signature: str = ""
    # True 表示这一轮没有重新清理，直接沿用上一轮的产物。
    # 上游据此判断"输入 PDF 这一轮有没有变"，没变就不该作废缓存。
    reused: bool = False


def _watermark_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("watermark", {})


def watermark_cleaning_enabled(cfg: dict[str, Any]) -> bool:
    # 书籍 PDF 的版式比期刊杂得多，去水印可能误删正文里的可选内容组，
    # 所以默认关闭，需要时在 config.py 的 watermark.enabled 打开。
    return bool(_watermark_cfg(cfg).get("enabled", False))


def _probe_readable(source_pdf: Path) -> int:
    """只做"这份 PDF 打不打得开"的体检，不改内容。

    ingest 统计页数时有 fitz / 正则兜底，损坏文件照样能被扫进来；
    关掉去水印时也得有人把它们拦下来，否则要到 MinerU 那边才炸，还会重试三次。
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return 0
    try:
        reader = PdfReader(str(source_pdf))
        return len(reader.pages)
    except Exception as exc:
        raise UnreadablePdfError(f"{type(exc).__name__}: {exc}") from exc


def _config_signature(cfg: dict[str, Any]) -> str:
    """只有这几项会影响清理结果，用它们判断上一轮的产物还能不能沿用。"""
    watermark_cfg = _watermark_cfg(cfg)
    return stable_json_hash(
        {
            "enabled": bool(watermark_cfg.get("enabled", False)),
            "form_names": sorted(str(item) for item in watermark_cfg.get("form_names", [])),
            "image_sizes": sorted(
                [int(item[0]), int(item[1])]
                for item in watermark_cfg.get("image_sizes", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ),
            "min_page_coverage": float(watermark_cfg.get("min_page_coverage", 0.6) or 0.6),
        }
    )


def _previous_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        payload = read_json(report_path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _unchanged_since_previous_run(
    previous: dict[str, Any] | None,
    result: WatermarkCleanResult,
) -> bool:
    """这一轮扫出来的结果跟上一轮一模一样吗？

    走到这里说明快速路径没命中 —— 通常是上一轮的报告由旧版本写的、没记
    source_hash/config_signature。重扫一遍不贵，贵的是因此误判"输入 PDF 变了"
    而把 MinerU 缓存作废、整本书重跑。所以拿 content_hash 兜底比对：
    一致就说明产物没变，缓存仍然对应同一份 PDF。
    """
    if previous is None:
        return False
    if bool(previous.get("cleaned")) != result.cleaned:
        return False
    return str(previous.get("content_hash") or "") == str(result.content_hash or "")


def _reusable_previous_result(
    report_path: Path,
    book: BookRecord,
    signature: str,
) -> WatermarkCleanResult | None:
    """上一轮的清理结果还算数吗？

    算数的条件是源 PDF 没换（hash 相同）、判定参数没改（signature 相同）。
    这件事必须做对两件：一是省掉重复扫描（每页内容流都要完整解析一遍，不便宜），
    二是让 reused=True 传上去 —— 否则上游每轮都以为"输入 PDF 刚变过"，
    把 MinerU/normalized/samples/exports 缓存全作废，整本书永远在从头重跑。
    """
    payload = _previous_report(report_path)
    if payload is None:
        return None
    if str(payload.get("source_hash") or "") != str(book.file_hash or ""):
        return None
    if str(payload.get("config_signature") or "") != signature:
        return None
    cleaned = bool(payload.get("cleaned"))
    cleaned_pdf = str(payload.get("cleaned_pdf") or book.source_pdf)
    if cleaned and not Path(cleaned_pdf).is_file():
        return None
    candidates = payload.get("candidate_names")
    return WatermarkCleanResult(
        source_pdf=str(book.source_pdf),
        cleaned_pdf=cleaned_pdf,
        cleaned=cleaned,
        candidate_names=[str(item) for item in candidates] if isinstance(candidates, list) else [],
        removed_invocations=int(payload.get("removed_invocations") or 0),
        page_count=int(payload.get("page_count") or book.page_count),
        content_hash=str(payload.get("content_hash") or book.file_hash),
        source_hash=str(book.file_hash or ""),
        config_signature=signature,
        reused=True,
    )


def _operator_text(operator: Any) -> str:
    if isinstance(operator, bytes):
        return operator.decode("latin1")
    return str(operator)


def _resolve_pdf_object(obj: Any) -> Any:
    for _ in range(8):
        if obj is None:
            return None
        get_object = getattr(obj, "get_object", None)
        if not callable(get_object):
            return obj
        try:
            resolved = get_object()
        except Exception:
            return None
        if resolved is obj:
            return obj
        obj = resolved
    return obj


def _pdf_dict(obj: Any) -> Any:
    resolved = _resolve_pdf_object(obj)
    get_value = getattr(resolved, "get", None)
    return resolved if callable(get_value) else {}


def _xobject_dict(page: Any) -> Any:
    resources = _pdf_dict(page.get("/Resources"))
    return _pdf_dict(resources.get("/XObject"))


def _form_has_configured_image(form: Any, cfg: dict[str, Any]) -> bool:
    expected_sizes = {
        (int(item[0]), int(item[1]))
        for item in _watermark_cfg(cfg).get("image_sizes", [])
        if isinstance(item, (list, tuple)) and len(item) == 2
    }
    if not expected_sizes:
        return False
    resources = _pdf_dict(form.get("/Resources"))
    xobjects = _pdf_dict(resources.get("/XObject"))
    if not xobjects:
        return False
    for nested_ref in xobjects.values():
        nested = _pdf_dict(nested_ref)
        if not nested:
            continue
        if str(nested.get("/Subtype")) != "/Image":
            continue
        size = (int(nested.get("/Width") or 0), int(nested.get("/Height") or 0))
        if size in expected_sizes:
            return True
    return False


def _is_candidate_form(name: str, form: Any, cfg: dict[str, Any]) -> bool:
    configured_names = {str(item) for item in _watermark_cfg(cfg).get("form_names", [])}
    if configured_names and name in configured_names:
        return True
    if form.get("/OC") is not None:
        return True
    return _form_has_configured_image(form, cfg)


def _page_form_invocations(reader: Any, page: Any, cfg: dict[str, Any]) -> list[str]:
    from pypdf.generic import ContentStream  # type: ignore

    xobjects = _xobject_dict(page)
    if not xobjects:
        return []
    content = page.get_contents()
    if content is None:
        return []
    result: list[str] = []
    stream = ContentStream(content, reader)
    for operands, operator in stream.operations:
        if _operator_text(operator) != "Do" or not operands:
            continue
        name = str(operands[0])
        if name not in xobjects:
            continue
        form = _pdf_dict(xobjects[name])
        if not form:
            continue
        if str(form.get("/Subtype")) == "/Form" and _is_candidate_form(name, form, cfg):
            result.append(name)
    return result


def _find_repeated_watermark_forms(reader: Any, cfg: dict[str, Any]) -> list[str]:
    page_count = len(reader.pages)
    if page_count <= 0:
        return []
    counts: dict[str, int] = {}
    for page in reader.pages:
        seen_on_page = set(_page_form_invocations(reader, page, cfg))
        for name in seen_on_page:
            counts[name] = counts.get(name, 0) + 1
    min_page_coverage = float(_watermark_cfg(cfg).get("min_page_coverage", 0.6) or 0.6)
    min_pages = max(1, int(round(page_count * min_page_coverage)))
    return sorted(name for name, count in counts.items() if count >= min_pages)


def _remove_form_invocations(reader: Any, writer: Any, names: set[str]) -> int:
    from pypdf.generic import ContentStream, NameObject  # type: ignore

    removed = 0
    for page in reader.pages:
        content = page.get_contents()
        if content is None:
            writer.add_page(page)
            continue
        stream = ContentStream(content, reader)
        filtered = []
        for operands, operator in stream.operations:
            if _operator_text(operator) == "Do" and operands and str(operands[0]) in names:
                removed += 1
                continue
            filtered.append((operands, operator))
        stream.operations = filtered
        page[NameObject("/Contents")] = stream
        writer.add_page(page)
    return removed


def _output_paths(book: BookRecord, cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(book.output_dir)
    cleaned_template = _watermark_cfg(cfg).get("cleaned_pdf_path", "preprocessed/{book_id}_cleaned.pdf")
    report_template = _watermark_cfg(cfg).get("report_path", "preprocessed/watermark_cleaning.json")
    return (
        output_dir / str(cleaned_template).format(book_id=book.book_id),
        output_dir / str(report_template).format(book_id=book.book_id),
    )


def clean_pdf_watermarks(book: BookRecord, cfg: dict[str, Any]) -> WatermarkCleanResult:
    source_pdf = Path(book.source_pdf)
    cleaned_pdf, report_path = _output_paths(book, cfg)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    signature = _config_signature(cfg)
    fast_path = _reusable_previous_result(report_path, book, signature)
    if fast_path is not None:
        # 报告也刷一遍，否则磁盘上留的是上一轮 reused=False 的旧记录，
        # 排查时会被误读成"这一轮又重新清理了"。
        write_json(report_path, asdict(fast_path))
        return fast_path
    previous = _previous_report(report_path)

    def _finish(result: WatermarkCleanResult) -> WatermarkCleanResult:
        if _unchanged_since_previous_run(previous, result):
            result = replace(result, reused=True)
        write_json(report_path, asdict(result))
        return result

    if not watermark_cleaning_enabled(cfg):
        result = WatermarkCleanResult(
            source_pdf=str(source_pdf),
            cleaned_pdf=str(source_pdf),
            cleaned=False,
            candidate_names=[],
            removed_invocations=0,
            page_count=_probe_readable(source_pdf) or book.page_count,
            content_hash=book.file_hash,
            source_hash=book.file_hash,
            config_signature=signature,
        )
        return _finish(result)

    from pypdf import PdfReader, PdfWriter  # type: ignore

    try:
        reader = PdfReader(str(source_pdf))
    except Exception as exc:
        # 截断、加密、结构损坏都会落到这里。ingest 那边读页数有 fitz/正则兜底，
        # 所以坏文件照样能扫进来，必须在这里拦住并说清楚原因。
        raise UnreadablePdfError(f"{type(exc).__name__}: {exc}") from exc
    candidate_names = _find_repeated_watermark_forms(reader, cfg)
    if not candidate_names:
        result = WatermarkCleanResult(
            source_pdf=str(source_pdf),
            cleaned_pdf=str(source_pdf),
            cleaned=False,
            candidate_names=[],
            removed_invocations=0,
            page_count=len(reader.pages),
            content_hash=book.file_hash,
            source_hash=book.file_hash,
            config_signature=signature,
        )
        return _finish(result)

    cleaned_pdf.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    removed = _remove_form_invocations(reader, writer, set(candidate_names))
    with cleaned_pdf.open("wb") as handle:
        writer.write(handle)

    result = WatermarkCleanResult(
        source_pdf=str(source_pdf),
        cleaned_pdf=str(cleaned_pdf),
        cleaned=removed > 0,
        candidate_names=candidate_names,
        removed_invocations=removed,
        page_count=len(reader.pages),
        content_hash=file_sha256(cleaned_pdf, int(cfg["hash_chunk_size"])) if removed > 0 else book.file_hash,
        source_hash=book.file_hash,
        config_signature=signature,
    )
    return _finish(result)

