from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text, safe_name, truncate_text, utc_now
from ..core.models import SampleJob
from ..services.clients import VlmPool, parse_json_array, parse_jsonl_objects


JSON_OUTPUT_INSTRUCTION = (
    "只返回合法 JSON 数组，不要使用 markdown 代码块。"
    "每条样本必须包含 instruction 字段和该任务要求的全部输出字段。"
    "instruction 必须由你根据当前输入证据重新生成，保持表达多样性，不得复制固定模板句。"
    "instruction 必须紧扣当前任务和当前内容，不得偏离主题。"
    "证据不足或不适合该任务时返回 []。"
)

PT_JSONL_OUTPUT_INSTRUCTION = (
    "只返回 JSONL，不要使用 markdown 代码块。"
    "每一行必须是一个完整 JSON 对象，并且只包含 text 字段。"
    "不要生成 instruction、input、output、messages、conversations 或问答格式。"
    "如果输入没有可用于继续预训练的正文知识，请返回空内容。"
)

REFERENCE_MARK_RE = re.compile(
    r"\s*(?:\$?\s*\^\s*\{\s*)?[\[［]\s*\d+(?:\s*[-–—,，、]\s*\d+)*\s*[\]］](?:\s*\}\s*\$?)?"
)

TASK_SPECS: dict[str, dict[str, Any]] = {
    "paragraph_summary": {
        "format": "alpaca",
        "input_fields": ["paragraph_text"],
        "output_fields": ["summary", "keywords"],
        "instruction_rules": [
            "围绕段落摘要生成自然问题。",
            "问题应体现当前段落主题，避免固定模板化措辞。",
            "不得要求段落以外的信息。",
        ],
        "quality_rules": [
            "不得引入原段落中没有的新实体。",
            "摘要必须覆盖主题与核心结论。",
        ],
    },
    "title_body_alignment": {
        "format": "alpaca",
        "input_fields": ["title", "body_text"],
        "output_fields": ["label", "reason"],
        "instruction_rules": [
            "围绕标题和正文是否匹配生成自然问题。",
            "问题应提到标题-正文关系，但不要照搬固定问法。",
            "不得要求判断输入之外的章节结构。",
        ],
        "quality_rules": [
            "label 只能表达对齐、不对齐或部分对齐等明确判断。",
            "reason 必须同时依据标题和正文内容。",
        ],
    },
    "page_to_structured_description": {
        "format": "sharegpt",
        "input_fields": ["page_image"],
        "output_fields": ["page_type", "title", "sections", "main_topics", "key_elements"],
        "instruction_rules": [
            "围绕页面结构识别生成自然问题。",
            "问题应要求输出结构化页面信息。",
            "不得变成内容问答或知识解释问题。",
        ],
        "quality_rules": [
            "title 仅在页面中可见时填写。",
            "page_type 应来自常见页面类型，不要自由发挥。",
        ],
    },
    "chart_table_to_text": {
        "format": "sharegpt",
        "input_fields": ["chart_image"],
        "output_fields": ["chart_subject", "trends", "conclusion"],
        "instruction_rules": [
            "围绕当前提取图表的趋势、关系或结论生成自然问题。",
            "问题要有多样性，但只能要求解读可见图表信息。",
            "不得要求识别图号、章节号或书籍位置。",
        ],
        "quality_rules": [
            "只能描述图中可见趋势。",
            "不要引入图中不存在的变量名和结论。",
        ],
    },
    "page_content_restatement": {
        "format": "sharegpt",
        "input_fields": ["page_image"],
        "output_fields": ["restatement"],
        "instruction_rules": [
            "围绕当前页面内容重述生成自然问题。",
            "问题应要求保持原意并覆盖主要信息。",
            "不得要求页外知识或问答式推理。",
        ],
        "quality_rules": [
            "不得加入页外知识。",
            "必须保留页面主题和主要结论。",
        ],
    },
    "cross_page_synthesis": {
        "format": "sharegpt",
        "input_fields": ["page_images"],
        "output_fields": ["topic", "page1_focus", "page2_focus", "combined_summary"],
        "instruction_rules": [
            "围绕多页内容整合生成自然问题。",
            "问题应要求覆盖全部输入页面。",
            "不得只询问其中一页。",
        ],
        "quality_rules": [
            "答案必须覆盖所有输入页面。",
            "不能只总结其中一页。",
        ],
    },
    "chapter_key_conclusions": {
        "format": "alpaca",
        "input_fields": ["chapter_title", "chapter_text"],
        "output_fields": ["chapter_topic", "key_conclusions"],
        "instruction_rules": [
            "围绕章节片段的核心结论生成自然问题。",
            "问题应要求提炼多个完整结论。",
            "不得要求章节外背景信息。",
        ],
        "quality_rules": [
            "每条结论都应是完整句子。",
            "建议保留 3 到 6 条关键结论。",
        ],
    },
    "evidence_to_conclusion_chain": {
        "format": "sharegpt",
        "input_fields": ["page_image", "claim"],
        "output_fields": ["evidence", "reasoning", "conclusion"],
        "instruction_rules": [
            "围绕页面证据到结论的链条生成自然问题。",
            "问题应要求列出证据、推理步骤和结论。",
            "不得要求使用页面外证据。",
        ],
        "quality_rules": [
            "所有证据都必须可以在页面中找到。",
            "结论强度不能超过页面证据可支持的范围。",
        ],
    },
    "domain_knowledge_corpus": {
        "format": "pt",
        "input_fields": ["book_name", "chapter_path", "raw_text"],
        "output_fields": ["text"],
        "response_format": "jsonl",
        "requires_instruction": False,
        "instruction_rules": [
            "不要生成问答格式，不要生成 instruction/input/output。",
            "只输出适合 LLaMA-Factory stage: pt 的 JSONL 纯文本语料。",
            "每行 JSON 对象只能包含 text 字段。",
        ],
        "quality_rules": [
            "不得添加原文没有的知识、案例、结论或解释。",
            "尽量保留原书的知识密度、术语、定义、推导、公式、步骤和结构。",
            "删除目录、版权页、索引、参考文献列表、空白页、水印、重复页眉页脚等低价值内容。",
            "按语义完整性切块，不在句子、公式、表格中间截断。",
        ],
    },
}

EVIDENCE_KEYS = {
    "source_pages",
    "evidence_block_ids",
    "evidence_text",
    "visual_evidence",
    "evidence_text_by_page",
    "visual_evidence_by_page",
    "supporting_evidence",
}


def _metadata(job: SampleJob, cfg: dict[str, Any], sample_no: int) -> dict[str, Any]:
    block = job.block
    return {
        "source_pdf": job.book.source_pdf,
        "book_name": job.book.book_name,
        "book_id": job.book.book_id,
        "chapter_title": job.page.chapter_title,
        "chapter_no": job.page.chapter_no,
        "page_index": job.page.page_index,
        "page_label": job.page.page_label,
        "block_id": block.block_id if block else cfg["default_block_id"],
        "block_type": block.block_type if block else cfg["default_block_type"],
        "bbox": block.bbox if block else [],
        "image_path": job.images[0] if job.images else "",
        "mineru_parse_path": job.page.mineru_parse_path,
        "normalized_page_path": job.page.normalized_page_path,
        "task_type": job.task_type,
        "created_at": utc_now(cfg["timestamp_format"]),
        "pipeline_version": cfg["pipeline_version"],
        "sample_no": sample_no,
        "generator": "llm",
    }


def _sample_id(metadata: dict[str, Any]) -> str:
    book_id = safe_name(str(metadata.get("book_id") or metadata.get("book_name") or "book"), "book")
    chapter_no = str(metadata.get("chapter_no") or "unknown")
    page_no = int(metadata.get("page_index") or 0)
    block_id = safe_name(str(metadata.get("block_id") or "page"), "page")
    task_type = str(metadata.get("task_type") or "task")
    sample_no = int(metadata.get("sample_no") or 0)
    return f"{book_id}_ch{chapter_no}_p{page_no:03d}_{block_id}_{task_type}_{sample_no:06d}"


def _source_pages(job: SampleJob) -> list[int]:
    pages = job.page_window or [job.page]
    return [page.page_index for page in pages]


def _evidence_block_ids(job: SampleJob) -> list[str]:
    if job.block:
        return [job.block.block_id]
    ids: list[str] = []
    for page in job.page_window or [job.page]:
        ids.extend(block.block_id for block in page.blocks if clean_text(block.text))
    return ids[:24]


def _evidence_text(job: SampleJob) -> list[str]:
    if job.block and clean_text(job.block.text):
        return [job.block.text]
    texts: list[str] = []
    for page in job.page_window or [job.page]:
        for block in page.blocks:
            text = clean_text(block.text)
            if text:
                texts.append(text)
    return texts[:12]


def _window_text(job: SampleJob, limit: int) -> str:
    pages = job.page_window or [job.page]
    text = "\n\n".join(page.full_text for page in pages if clean_text(page.full_text))
    return truncate_text(text, limit)


def _truncate_raw_text(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    breakpoints = [cut.rfind("\n\n"), cut.rfind("\n"), cut.rfind("。")]
    breakpoint = max(breakpoints)
    if breakpoint > limit // 2:
        return cut[: breakpoint + 1].strip()
    return cut


def _clean_pt_corpus_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        line = REFERENCE_MARK_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if re.fullmatch(r"(第\s*)?\d{1,5}\s*(页)?", line):
            continue
        if re.fullmatch(r"[-—_=·•●○\s]{3,}", line):
            continue
        line = re.sub(r"\s+([，。；：！？、,.!?;:])", r"\1", line)
        cleaned_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def _corpus_chapter_path(job: SampleJob) -> str:
    value = job.source.get("chapter_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    titles: list[str] = []
    for page in job.page_window or [job.page]:
        title = clean_text(page.chapter_title)
        if title and title not in titles:
            titles.append(title)
    if titles:
        return " / ".join(titles)
    pages = job.page_window or [job.page]
    start = clean_text(pages[0].page_label) or str(pages[0].page_index)
    end = clean_text(pages[-1].page_label) or str(pages[-1].page_index)
    return f"页码 {start}" if start == end else f"页码 {start}-{end}"


def _corpus_raw_text(job: SampleJob, limit: int) -> str:
    raw = job.source.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return _truncate_raw_text(raw, limit)

    skip_types = {"header", "footer", "page_number", "unknown"}
    pages_text: list[str] = []
    for page in job.page_window or [job.page]:
        lines: list[str] = []
        page_label = clean_text(page.page_label) or str(page.page_index)
        lines.append(f"[页码：{page_label}]")
        for block in page.blocks:
            if block.block_type in skip_types:
                continue
            text = clean_text(block.markdown or block.text)
            if text:
                lines.append(text)
        page_text = "\n".join(lines).strip()
        if page_text:
            pages_text.append(page_text)
    return _truncate_raw_text("\n\n".join(pages_text), limit)


def _template_input(job: SampleJob, cfg: dict[str, Any]) -> dict[str, Any]:
    max_page_chars = int(cfg["generation"]["max_page_context_chars"])
    max_neighbor_chars = int(cfg["generation"]["max_neighbor_context_chars"])
    if job.task_type == "domain_knowledge_corpus":
        max_pt_chars = int(cfg["generation"].get("max_pt_context_chars", max_page_chars))
        return {
            "book_name": job.book.book_name,
            "chapter_path": _corpus_chapter_path(job),
            "raw_text": _corpus_raw_text(job, max_pt_chars),
        }
    if job.task_type == "paragraph_summary":
        text = job.block.text if job.block else job.page.full_text
        return {"paragraph_text": truncate_text(text, max_page_chars)}
    if job.task_type == "title_body_alignment":
        title = job.source.get("title", {}).get("text") if isinstance(job.source.get("title"), dict) else ""
        body_blocks = job.source.get("body_blocks") if isinstance(job.source.get("body_blocks"), list) else []
        body_text = "\n".join(str(item.get("text") or "") for item in body_blocks if isinstance(item, dict))
        return {"title": clean_text(title), "body_text": truncate_text(body_text, max_page_chars)}
    if job.task_type == "chapter_key_conclusions":
        return {
            "chapter_title": job.page.chapter_title or f"第 {job.page.page_index} 页起的章节片段",
            "chapter_text": _window_text(job, max_page_chars),
        }
    if job.task_type == "evidence_to_conclusion_chain":
        return {"page_image": job.images[0] if job.images else job.page.page_image, "claim": ""}
    if job.task_type == "cross_page_synthesis":
        return {"page_images": list(job.images)}
    if job.task_type == "chart_table_to_text":
        return {"chart_image": job.images[0] if job.images else ""}
    return {"page_image": job.images[0] if job.images else job.page.page_image}


def _model_context(job: SampleJob, cfg: dict[str, Any]) -> dict[str, Any]:
    max_page_chars = int(cfg["generation"]["max_page_context_chars"])
    max_neighbor_chars = int(cfg["generation"]["max_neighbor_context_chars"])
    page_window = [
        {
            "page_index": page.page_index,
            "page_image": page.page_image,
            "full_text": truncate_text(page.full_text, max_neighbor_chars),
            "blocks": [asdict(block) for block in page.blocks],
        }
        for page in job.page_window
    ]
    return {
        "template_input": _template_input(job, cfg),
        "page_ocr": truncate_text(job.page.full_text, max_page_chars),
        "layout_blocks": [asdict(block) for block in job.page.blocks],
        "block": asdict(job.block) if job.block else {},
        "page_window": page_window,
        "source": job.source,
    }


def _prompt(job: SampleJob, cfg: dict[str, Any]) -> str:
    spec = TASK_SPECS[job.task_type]
    if spec["format"] == "pt":
        template_input = _template_input(job, cfg)
        prompt = str(cfg["prompts"][job.task_type]).strip()
        return (
            f"{prompt}\n\n{PT_JSONL_OUTPUT_INSTRUCTION}\n\n"
            "输入信息：\n"
            f"书名：{template_input.get('book_name', '')}\n"
            f"章节路径：{template_input.get('chapter_path', '')}\n"
            "原始解析文本：\n"
            f"{template_input.get('raw_text', '')}"
        )

    payload = {
        "task_type": job.task_type,
        "target_format": spec["format"],
        "input_fields": spec["input_fields"],
        "output_fields": spec["output_fields"],
        "instruction_rules": spec["instruction_rules"],
        "quality_rules": spec["quality_rules"],
        "source_info": _metadata(job, cfg, 0),
        "context": _model_context(job, cfg),
        "expected_count": cfg["generation"]["samples_per_job"].get(job.task_type, 1),
    }
    prompt = str(cfg["prompts"][job.task_type]).strip()
    template_rule = (
        "请严格按 task_template.md 的任务模板生成样本："
        "instruction 必须由大模型根据当前输入证据生成，表达要多样化，不能复制固定模板句；"
        "输出 JSON 对象中除 instruction 和证据字段外，只保留 output_fields 指定字段。"
    )
    return f"{prompt}\n\n{template_rule}\n\n{JSON_OUTPUT_INSTRUCTION}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def _normalize_instruction(raw: Any, task_type: str, sample_no: int) -> str:
    instruction = clean_text(raw)
    return instruction.replace("<image>", "").strip()


def _split_output_and_evidence(row: dict[str, Any], job: SampleJob) -> tuple[dict[str, Any], dict[str, Any]]:
    output_fields = set(TASK_SPECS[job.task_type]["output_fields"])
    evidence = {key: row.get(key) for key in EVIDENCE_KEYS if key in row}
    evidence.setdefault("source_pages", _source_pages(job))
    evidence.setdefault("evidence_block_ids", _evidence_block_ids(job))
    evidence.setdefault("evidence_text", _evidence_text(job))
    evidence.setdefault("visual_evidence", list(job.images))
    output = {key: row.get(key) for key in output_fields if key in row}
    return output, evidence


def _pt_samples_from_rows(
    rows: list[dict[str, Any]],
    job: SampleJob,
    cfg: dict[str, Any],
    response_provider: str,
    response_model: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    template_input = _template_input(job, cfg)
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        metadata = _metadata(job, cfg, index)
        metadata["generator_provider"] = response_provider
        metadata["generator_model"] = response_model
        metadata["target_format"] = "pt"
        metadata["llamafactory_stage"] = "pt"
        evidence = {
            "source_pages": _source_pages(job),
            "evidence_block_ids": _evidence_block_ids(job),
            "evidence_text": _evidence_text(job),
            "visual_evidence": [],
        }
        sample = {
            "id": _sample_id(metadata),
            "task_type": job.task_type,
            "input_payload": template_input,
            "output_payload": {"text": text},
            "evidence": evidence,
            "metadata": metadata,
            "images": [],
            "text": text,
            "question": "",
            "answer": text,
            "export_input": template_input.get("raw_text", ""),
        }
        samples.append(sample)
    return samples


def _generate_pt_rule_based(job: SampleJob, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    template_input = _template_input(job, cfg)
    raw_text = _clean_pt_corpus_text(str(template_input.get("raw_text") or ""))
    if not raw_text:
        return []
    header = (
        f"书名：{template_input.get('book_name', '')}\n"
        f"章节：{template_input.get('chapter_path', '')}\n\n"
    )
    samples = _pt_samples_from_rows([{"text": header + raw_text}], job, cfg, "heuristic", "rule")
    for sample in samples:
        metadata = sample.setdefault("metadata", {})
        metadata["generator"] = "rule_based_pt_cleaner"
    return samples


def generate_for_job(job: SampleJob, output_dir: Path, cfg: dict[str, Any], vlm: VlmPool | None) -> list[dict[str, Any]]:
    if job.task_type == "domain_knowledge_corpus":
        return _generate_pt_rule_based(job, cfg)

    if vlm is None or bool(cfg["runtime"]["skip_vlm"]):
        return []

    image_paths = [output_dir / image for image in job.images if (output_dir / image).exists()]
    response = vlm.chat(task_type=job.task_type, prompt=_prompt(job, cfg), images=image_paths)
    if TASK_SPECS[job.task_type]["format"] == "pt":
        rows = parse_jsonl_objects(response.text)
        return _pt_samples_from_rows(rows, job, cfg, response.provider_name, response.model)

    rows = parse_json_array(response.text)

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        output_payload, evidence = _split_output_and_evidence(row, job)
        if not output_payload:
            continue
        metadata = _metadata(job, cfg, index)
        metadata["generator_provider"] = response.provider_name
        metadata["generator_model"] = response.model
        metadata["target_format"] = TASK_SPECS[job.task_type]["format"]
        instruction = _normalize_instruction(row.get("instruction"), job.task_type, index)
        if not instruction:
            continue
        template_input = _template_input(job, cfg)
        sample = {
            "id": _sample_id(metadata),
            "task_type": job.task_type,
            "instruction": instruction,
            "input_payload": template_input,
            "output_payload": output_payload,
            "evidence": evidence,
            "metadata": metadata,
            "images": list(job.images),
        }
        sample["question"] = instruction
        sample["answer"] = json.dumps(output_payload, ensure_ascii=False, indent=2)
        sample["export_input"] = _format_alpaca_input(job.task_type, template_input)
        samples.append(sample)
    return samples


def _format_alpaca_input(task_type: str, payload: dict[str, Any]) -> str:
    if task_type == "paragraph_summary":
        return str(payload.get("paragraph_text") or "")
    if task_type == "title_body_alignment":
        return f"标题：{payload.get('title', '')}\n\n正文：{payload.get('body_text', '')}".strip()
    if task_type == "chapter_key_conclusions":
        return f"章节标题：{payload.get('chapter_title', '')}\n\n章节内容：{payload.get('chapter_text', '')}".strip()
    return json.dumps(payload, ensure_ascii=False, indent=2)
