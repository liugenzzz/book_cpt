from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.io_utils import write_jsonl


def _answer(sample: dict[str, Any]) -> str:
    return json.dumps(sample.get("output_payload", {}), ensure_ascii=False)


def _input(sample: dict[str, Any]) -> str:
    value = sample.get("export_input")
    if isinstance(value, str) and value.strip():
        return value
    return json.dumps(sample.get("input_payload", {}), ensure_ascii=False)


def to_sharegpt(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    image_tokens = "".join(cfg["sharegpt_image_token"] for _ in images)
    instruction = str(sample.get("instruction") or "").replace(cfg["sharegpt_image_token"], "").strip()
    record: dict[str, Any] = {
        "id": sample["id"],
        "messages": [
            {"role": "user", "content": f"{image_tokens}{instruction}".strip()},
            {"role": "assistant", "content": _answer(sample)},
        ],
    }
    if images:
        record["images"] = images
    return record


def to_alpaca(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": sample.get("instruction", ""),
        "input": _input(sample),
        "output": _answer(sample),
    }


def to_pt(sample: dict[str, Any]) -> dict[str, Any]:
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    text = str(output.get("text") or sample.get("text") or "").strip()
    return {"text": text}


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return "；".join(_format_value(item) for item in value if _format_value(item))
    if isinstance(value, dict):
        return "；".join(f"{key}：{_format_value(child)}" for key, child in value.items() if _format_value(child))
    return str(value).strip()


def _chart_sample_to_pt(sample: dict[str, Any]) -> dict[str, str]:
    if str(sample.get("task_type") or "") != "chart_table_to_text":
        return {"text": ""}
    payload = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    if not payload:
        return {"text": ""}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    lines = [
        f"书名：{metadata.get('book_name', '')}".strip(),
        f"章节：{metadata.get('chapter_title') or metadata.get('chapter_no') or ''}".strip(),
        f"页码：{metadata.get('page_label') or metadata.get('page_index', '')}".strip(),
        "图表知识：",
    ]
    fields = [
        ("主题", payload.get("chart_subject") or payload.get("target_description")),
        ("可见信息", payload.get("visible_information")),
        ("图注表注", payload.get("caption_information")),
        ("正文解释", payload.get("text_explanation")),
        ("趋势关系", payload.get("trends")),
        ("解读", payload.get("interpretation")),
        ("结论", payload.get("conclusion")),
    ]
    for label, value in fields:
        text = _format_value(value)
        if text:
            lines.append(f"{label}：{text}")
    return {"text": "\n".join(line for line in lines if line and not line.endswith("：")).strip()}


def _pt_records_for_task(task_type: str, task_samples: list[dict[str, Any]], all_samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = [record for sample in task_samples if (record := to_pt(sample))["text"]]
    if task_type == "domain_knowledge_corpus":
        records.extend(record for sample in all_samples if (record := _chart_sample_to_pt(sample))["text"])
    return records


def export_task_files(output_dir: Path, samples: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    sharegpt_task_types = set(cfg["export_formats"]["sharegpt"])
    alpaca_task_types = set(cfg["export_formats"]["alpaca"])
    pt_task_types = set(cfg["export_formats"].get("pt", []))
    for task_type in cfg["task_types"]:
        task_samples = [sample for sample in samples if sample.get("task_type") == task_type]
        if task_type in sharegpt_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_sharegpt"].format(task_type=task_type),
                (to_sharegpt(sample, cfg) for sample in task_samples),
            )
        if task_type in alpaca_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_alpaca"].format(task_type=task_type),
                (to_alpaca(sample) for sample in task_samples),
            )
        if task_type in pt_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_pt"].format(task_type=task_type),
                _pt_records_for_task(task_type, task_samples, samples),
            )
