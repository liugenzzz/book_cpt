from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text
from ..services.clients import parse_jsonl_objects


ALPACA_QUALITY_DIMENSIONS = ("text_format_quality", "qa_relevance")
SHAREGPT_IMAGE_QUALITY_DIMENSIONS = (
    "text_format_quality",
    "qa_relevance",
    "visual_dependency",
    "image_question_alignment",
)
QUALITY_DIMENSION_LABELS = {
    "text_format_quality": "文本格式质量",
    "qa_relevance": "问答相关性",
    "visual_dependency": "视觉依赖度",
    "image_question_alignment": "图像-问题对应性",
    "pt_source_coverage": "PT源文覆盖度",
}
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
MOJIBAKE_SEQUENCE_RE = re.compile(
    r"(?:Ã.|Â.|â[€œ€™“”]|[åæçèéäöü]{2,}|锛|绗|鍥|琛|鏈|妯|搴|浣|犳)"
)
NON_ANSWER_RE = re.compile(
    r"(?:"
    r"无法.{0,8}(?:回答|判断|确定|完成)|"
    r"不能.{0,8}(?:回答|判断|确定|完成)|"
    r"(?:没有|未提供|缺少).{0,12}(?:问题|答案|图片|图像|数据|信息|上下文|材料)|"
    r"(?:图片|图像|数据|任务|样本|文件).{0,12}(?:清晰|可用|有效|正常|完整|合格)|"
    r"(?:答案|回答).{0,8}(?:不可用|为空|缺失)|"
    r"\b(?:N/A|none|null)\b"
    r")",
    flags=re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"\{(?:question|answer|instruction|input|output|text|image)[^{}]*\}", re.IGNORECASE)


def _short_text(value: Any, limit: int = 2400) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _task_format(task_type: str, cfg: dict[str, Any]) -> str:
    export_formats = cfg.get("export_formats", {})
    for format_name in ("alpaca", "sharegpt", "pt"):
        if task_type in set(export_formats.get(format_name, [])):
            return format_name
    return ""


def _quality_dimensions(sample: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, ...]:
    task_type = str(sample.get("task_type") or "")
    format_name = _task_format(task_type, cfg)
    if format_name == "alpaca":
        return ALPACA_QUALITY_DIMENSIONS
    if format_name == "sharegpt" and sample.get("images"):
        return SHAREGPT_IMAGE_QUALITY_DIMENSIONS
    return ()


def _question_text(sample: dict[str, Any]) -> str:
    return clean_text(sample.get("question") or sample.get("instruction") or "")


def _answer_text(sample: dict[str, Any]) -> str:
    if _has_value(sample.get("answer")):
        return clean_text(sample.get("answer"))
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else sample.get("output_payload")
    return clean_text(json.dumps(output, ensure_ascii=False, sort_keys=True, default=str))


def _looks_garbled(text: str) -> bool:
    if CONTROL_CHAR_RE.search(text):
        return True
    if MOJIBAKE_SEQUENCE_RE.search(text):
        return True
    latin1_count = sum(1 for char in text if "\u00c0" <= char <= "\u017f")
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return cjk_count > 0 and latin1_count >= 6 and latin1_count / max(1, len(text)) > 0.12


def _check_result(dimension: str, passed: bool, score: float, reasons: list[str] | str = "") -> dict[str, Any]:
    if isinstance(reasons, str):
        reason_text = reasons
    else:
        reason_text = "；".join(reason for reason in reasons if reason)
    return {
        "dimension": dimension,
        "label": QUALITY_DIMENSION_LABELS.get(dimension, dimension),
        "passed": bool(passed),
        "score": round(max(0.0, min(1.0, float(score))), 4),
        "reason": reason_text,
    }


def _text_format_quality_check(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    validation = cfg.get("validation", {})
    min_question_chars = int(validation.get("min_question_chars", 4))
    min_answer_chars = int(validation.get("min_answer_chars", 8))
    question = _question_text(sample)
    answer = _answer_text(sample)
    reasons: list[str] = []
    if len(question) < min_question_chars:
        reasons.append("问题文本过短")
    if len(answer) < min_answer_chars:
        reasons.append("答案文本过短")
    if _looks_garbled(question) or _looks_garbled(answer):
        reasons.append("问题或答案疑似存在乱码")
    if "```" in question or "```" in answer:
        reasons.append("问题或答案包含 Markdown 代码围栏")
    if PLACEHOLDER_RE.search(question) or PLACEHOLDER_RE.search(answer):
        reasons.append("问题或答案包含未替换占位符")
    score = 1.0 if not reasons else max(0.0, 1.0 - 0.25 * len(reasons))
    return _check_result("text_format_quality", not reasons, score, reasons)


def _qa_relevance_check(sample: dict[str, Any]) -> dict[str, Any]:
    question = _question_text(sample)
    answer = _answer_text(sample)
    reasons: list[str] = []
    if not question or not answer:
        reasons.append("缺少问题或答案文本")
    if question and answer and question == answer:
        reasons.append("答案与问题完全重复")
    if NON_ANSWER_RE.search(answer):
        reasons.append("答案像是在评价图片、数据或任务状态，而没有正面回答问题")
    score = 1.0 if not reasons else max(0.0, 1.0 - 0.35 * len(reasons))
    return _check_result("qa_relevance", not reasons, score, reasons)


def _local_quality_checks(sample: dict[str, Any], cfg: dict[str, Any], dimensions: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    if "text_format_quality" in dimensions:
        checks["text_format_quality"] = _text_format_quality_check(sample, cfg)
    if "qa_relevance" in dimensions:
        checks["qa_relevance"] = _qa_relevance_check(sample)
    return checks


def _quality_review_prompt(sample: dict[str, Any], cfg: dict[str, Any], dimensions: tuple[str, ...]) -> str:
    task_type = str(sample.get("task_type") or "")
    format_name = _task_format(task_type, cfg)
    dimension_lines = []
    definitions = {
        "text_format_quality": "评估问题和答案语言是否通顺、无乱码、格式是否规范。",
        "qa_relevance": (
            "评估回答是否切题、是否正面回应问题、是否完成用户要求、逻辑是否一致。"
            "主要看问答文本本身；必须判断答案是否真正回答了问题，而不是只评价图片、数据或任务是否可用。"
        ),
        "visual_dependency": (
            "评估问题是否必须依赖当前图片中的具体视觉信息才能可靠回答。"
            "重点判断“没有当前图片能否回答”，不要只看题干是否声称要看图。"
        ),
        "image_question_alignment": (
            "评估当前随请求附带的图片是否与问题匹配，以及答案是否被当前图片内容支持。"
            "必须检查实际图片证据；如果图片看不出问题所问对象或答案事实，应给低分。"
        ),
    }
    for dimension in dimensions:
        dimension_lines.append(f"- {dimension}（{QUALITY_DIMENSION_LABELS[dimension]}）：{definitions[dimension]}")

    payload = {
        "task_type": task_type,
        "target_format": format_name,
        "question": _question_text(sample),
        "answer": _answer_text(sample),
        "input_payload": sample.get("input_payload", {}),
        "output_payload": sample.get("output_payload", {}),
        "evidence": sample.get("evidence", {}),
        "image_count": len(sample.get("images") if isinstance(sample.get("images"), list) else []),
    }
    return (
        "你是书籍训练样本质量校验专家。请只基于给定样本，以及本请求实际附带的当前图片进行质量评估。\n"
        "如果附带图片，必须亲自核对图片内容；不要把 evidence 或题干声称当作图片匹配的充分证据。\n"
        "请按以下维度评分，每个 score 为 0 到 1，passed 表示该维度是否达到可用于训练的质量。\n"
        f"{chr(10).join(dimension_lines)}\n\n"
        "输出必须是一个合法 JSON 对象，不要使用 Markdown。结构如下：\n"
        "{\n"
        '  "is_valid": true,\n'
        '  "quality_score": 0.0,\n'
        '  "checks": {\n'
        '    "text_format_quality": {"passed": true, "score": 0.0, "reason": ""}\n'
        "  },\n"
        '  "reject_reason": "",\n'
        '  "fix_suggestion": ""\n'
        "}\n\n"
        "待校验样本 JSON：\n"
        f"{_short_text(payload, 5000)}"
    )


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed", "valid", "1"}:
            return True
        if lowered in {"false", "no", "fail", "failed", "invalid", "0"}:
            return False
    return default


def _parse_quality_review(text: str, dimensions: tuple[str, ...], min_dimension_score: float) -> dict[str, Any]:
    rows = parse_jsonl_objects(text)
    payload = rows[0] if rows else {}
    checks_payload = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    checks: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        raw = checks_payload.get(dimension) or checks_payload.get(QUALITY_DIMENSION_LABELS.get(dimension, ""))
        raw = raw if isinstance(raw, dict) else {}
        score = _coerce_score(raw.get("score"), 0.0)
        passed = _coerce_bool(raw.get("passed"), score >= min_dimension_score)
        checks[dimension] = _check_result(
            dimension,
            passed,
            score,
            str(raw.get("reason") or raw.get("reject_reason") or ""),
        )
    if payload.get("quality_score") is None:
        quality_score = sum(item["score"] for item in checks.values()) / max(1, len(checks))
    else:
        quality_score = _coerce_score(payload.get("quality_score"), 0.0)
    is_valid = _coerce_bool(payload.get("is_valid"), all(item["passed"] for item in checks.values()))
    return {
        "is_valid": is_valid,
        "quality_score": quality_score,
        "checks": checks,
        "reject_reason": str(payload.get("reject_reason") or ""),
        "fix_suggestion": str(payload.get("fix_suggestion") or ""),
    }


def _quality_review_image_paths(sample: dict[str, Any], output_dir: Path) -> list[Path]:
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    return [output_dir / str(image) for image in images]


def _passes_quality_dimensions(
    sample: dict[str, Any],
    output_dir: Path,
    cfg: dict[str, Any],
    vlm: Any | None,
) -> bool:
    dimensions = _quality_dimensions(sample, cfg)
    if not dimensions:
        return True

    metadata = sample.setdefault("metadata", {})
    validation = cfg.get("validation", {})
    min_quality_score = float(validation.get("quality_review_min_score", 0.65))
    min_dimension_score = float(validation.get("quality_review_min_dimension_score", 0.65))
    local_checks = _local_quality_checks(sample, cfg, dimensions)
    metadata["quality_checks"] = local_checks
    local_failed = [key for key, value in local_checks.items() if not value.get("passed")]
    if local_failed:
        metadata["quality_review"] = {
            "type": "local",
            "is_valid": False,
            "reject_reason": "local quality check failed: " + ",".join(local_failed),
        }
        return False

    if not bool(validation.get("quality_review_enabled", True)) or vlm is None:
        metadata["quality_review"] = {
            "type": "local",
            "is_valid": True,
            "skipped_model_review": vlm is None,
        }
        return True

    try:
        image_paths = _quality_review_image_paths(sample, output_dir) if "image_question_alignment" in dimensions else []
        response = vlm.chat(
            task_type=str(sample.get("task_type") or ""),
            prompt=_quality_review_prompt(sample, cfg, dimensions),
            images=image_paths,
        )
        review = _parse_quality_review(response.text, dimensions, min_dimension_score)
        checks = review["checks"]
        metadata["quality_checks"] = checks
        metadata["quality_review"] = {
            "type": "vlm",
            "provider": response.provider_name,
            "model": response.model,
            "dimensions": list(dimensions),
            "is_valid": bool(review["is_valid"]),
            "reject_reason": review["reject_reason"],
            "fix_suggestion": review["fix_suggestion"],
        }
        metadata["quality_score"] = float(review["quality_score"])
        return bool(review["is_valid"]) and review["quality_score"] >= min_quality_score and all(
            check["score"] >= min_dimension_score and check["passed"] for check in checks.values()
        )
    except Exception as exc:
        metadata["quality_review"] = {
            "type": "vlm",
            "is_valid": False,
            "error": str(exc),
        }
        return not bool(validation.get("quality_review_fail_closed", True))


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _source_pages(sample: dict[str, Any]) -> list[Any]:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    pages = evidence.get("source_pages")
    return pages if isinstance(pages, list) else []


def _valid_images(sample: dict[str, Any], output_dir: Path) -> bool:
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    return all((output_dir / str(image)).exists() for image in images)


def _required_fields_present(sample: dict[str, Any], cfg: dict[str, Any]) -> bool:
    task_type = str(sample.get("task_type") or "")
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    required = cfg.get("validation", {}).get("required_output_fields", {}).get(task_type, [])
    return all(_has_value(output.get(field)) or _has_value(sample.get("evidence", {}).get(field)) for field in required)


def _chart_answer_has_forbidden_reference(sample: dict[str, Any]) -> bool:
    if str(sample.get("task_type") or "") != "chart_table_to_text":
        return False
    text = json.dumps(sample.get("output_payload", {}), ensure_ascii=False)
    forbidden_patterns = [
        r"图号",
        r"章节号",
        r"本文",
        r"文中",
        r"书中",
        r"本书",
        r"该书",
        r"章节",
        r"本章",
        r"本节",
        r"第\s*[0-9一二三四五六七八九十百]+\s*[章节节]",
        r"图\s*[0-9０-９]+(?:\s*[-－—]\s*[0-9０-９]+)?",
        r"表\s*[0-9０-９]+(?:\s*[-－—]\s*[0-9０-９]+)?",
        r"(?:fig\.?|figure)\s*[0-9]+",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in forbidden_patterns)


def _pt_task_types(cfg: dict[str, Any]) -> set[str]:
    return {str(item) for item in cfg.get("export_formats", {}).get("pt", []) if str(item)}


def _pt_text(sample: dict[str, Any]) -> str:
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    return str(output.get("text") or sample.get("text") or "").strip()


def _pt_source_coverage_quality(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    validation = cfg.get("validation", {})
    output_text = clean_text(_pt_text(sample))
    input_payload = sample.get("input_payload") if isinstance(sample.get("input_payload"), dict) else {}
    source_text = clean_text(input_payload.get("raw_text"))
    reasons: list[str] = []
    if not output_text:
        return _check_result("pt_source_coverage", False, 0.0, "PT文本为空")
    if not source_text:
        return _check_result("pt_source_coverage", True, 1.0, "")

    score = 1.0
    ratio = len(output_text) / max(1, len(source_text))
    min_ratio = float(validation.get("min_pt_output_source_ratio", 0.65))
    if ratio < min_ratio:
        reasons.append(f"输出/输入长度比例过低：{ratio:.2f}")
        score -= 0.45

    heading_pattern = r"(?:^|\s)(第\s*[一二三四五六七八九十百千万0-9０-９]+\s*[章节篇]|[0-9０-９]+(?:\.[0-9０-９]+)*\s+[\u4e00-\u9fffA-Za-z][^\s。；:：]{1,40})"
    headings = [clean_text(item) for item in re.findall(heading_pattern, source_text)]
    missing = [heading for heading in dict.fromkeys(headings) if heading and heading not in output_text]
    max_missing = int(validation.get("max_missing_pt_section_headings", 0))
    if len(missing) > max_missing:
        reasons.append("缺少源文标题：" + "；".join(missing[:5]))
        score -= 0.45

    threshold = float(validation.get("min_pt_source_coverage_score", 0.70))
    return _check_result("pt_source_coverage", score >= threshold and not reasons, score, reasons)


def _pt_text_format_quality(sample: dict[str, Any]) -> dict[str, Any]:
    text = _pt_text(sample)
    reasons: list[str] = []
    if not text:
        reasons.append("PT文本为空")
    if _looks_garbled(text):
        reasons.append("PT文本疑似存在乱码")
    if "```" in text:
        reasons.append("PT文本包含 Markdown 代码围栏")
    forbidden = {"instruction", "input", "output", "messages", "conversations", "images", "metadata"}
    if any(re.search(rf"\b{re.escape(key)}\b\s*[:：]", text, flags=re.IGNORECASE) for key in forbidden):
        reasons.append("PT文本包含训练格式字段")
    score = 1.0 if not reasons else max(0.0, 1.0 - 0.25 * len(reasons))
    return _check_result("text_format_quality", not reasons, score, reasons)


def validate_sample(
    sample: dict[str, Any],
    output_dir: Path,
    cfg: dict[str, Any],
    vlm: Any | None = None,
) -> dict[str, Any] | None:
    task_type = str(sample.get("task_type") or "")
    if task_type not in set(cfg.get("task_types", [])):
        return None
    is_pt_task = task_type in _pt_task_types(cfg)
    if is_pt_task:
        min_chars = int(cfg.get("validation", {}).get("min_pt_text_chars", 80))
        if not _has_value(sample.get("id")) or len(clean_text(_pt_text(sample))) < min_chars:
            return None
    else:
        if not _has_value(sample.get("id")) or not _has_value(sample.get("instruction")) or not _has_value(sample.get("output_payload")):
            return None
    if not _required_fields_present(sample, cfg):
        return None
    if _chart_answer_has_forbidden_reference(sample):
        return None
    pages = _source_pages(sample)
    if not pages:
        return None
    multi_page_tasks = set(cfg.get("validation", {}).get("multi_page_tasks", []))
    if task_type in multi_page_tasks and len(set(pages)) < 2:
        return None
    image_required_tasks = set(cfg.get("validation", {}).get("image_required_tasks", []))
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    if task_type in image_required_tasks and not images:
        return None
    if images and not _valid_images(sample, output_dir):
        return None
    if is_pt_task:
        text_quality = _pt_text_format_quality(sample)
        coverage = _pt_source_coverage_quality(sample, cfg)
        metadata = sample.setdefault("metadata", {})
        checks = metadata.setdefault("quality_checks", {})
        if isinstance(checks, dict):
            checks["text_format_quality"] = text_quality
            checks["pt_source_coverage"] = coverage
        metadata["quality_score"] = (float(text_quality.get("score") or 0.0) + float(coverage.get("score") or 0.0)) / 2
        if not text_quality.get("passed") or not coverage.get("passed"):
            return None
    if not _passes_quality_dimensions(sample, output_dir, cfg, vlm):
        return None
    metadata = sample.setdefault("metadata", {})
    metadata["validator"] = "rule+vlm" if metadata.get("quality_review", {}).get("type") == "vlm" else "rule"
    metadata.setdefault("quality_score", float(cfg["validation"]["default_quality_score"]))
    return sample
