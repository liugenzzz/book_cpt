from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from book_cpt.core.config_loader import load_config
from book_cpt.tasks.validation import validate_sample


class FakeVlm:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, task_type: str, prompt: str, images: list[Path] | None = None) -> Any:
        self.calls.append({"task_type": task_type, "prompt": prompt, "images": images or []})
        return SimpleNamespace(
            text=json.dumps(self.response, ensure_ascii=False),
            provider_name="fake-provider",
            model="fake-model",
        )


def _base_sample(task_type: str, output_payload: dict[str, Any], images: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "sample-1",
        "task_type": task_type,
        "instruction": "请根据当前材料概括主要内容。",
        "question": "请根据当前材料概括主要内容。",
        "answer": json.dumps(output_payload, ensure_ascii=False),
        "input_payload": {"paragraph_text": "航空器结构需要兼顾强度、刚度和重量约束。"},
        "output_payload": output_payload,
        "evidence": {"source_pages": [1], "evidence_text": ["航空器结构需要兼顾强度、刚度和重量约束。"]},
        "metadata": {},
        "images": images or [],
    }


def _passing_review(dimensions: list[str]) -> dict[str, Any]:
    return {
        "is_valid": True,
        "quality_score": 0.91,
        "checks": {
            dimension: {"passed": True, "score": 0.9, "reason": "ok"}
            for dimension in dimensions
        },
        "reject_reason": "",
        "fix_suggestion": "",
    }


def test_alpaca_quality_checks_reject_obvious_non_answer(tmp_path) -> None:
    cfg = load_config()
    sample = _base_sample("paragraph_summary", {"summary": "图片清晰，数据可用。"})

    assert validate_sample(sample, tmp_path, cfg) is None
    assert sample["metadata"]["quality_review"]["type"] == "local"


def test_alpaca_quality_review_records_two_dimensions(tmp_path) -> None:
    cfg = load_config()
    sample = _base_sample(
        "paragraph_summary",
        {"summary": "航空器结构设计需要在强度、刚度和重量之间取得平衡。"},
    )
    vlm = FakeVlm(_passing_review(["text_format_quality", "qa_relevance"]))

    valid = validate_sample(sample, tmp_path, cfg, vlm=vlm)

    assert valid is sample
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["images"] == []
    assert set(sample["metadata"]["quality_checks"]) == {"text_format_quality", "qa_relevance"}
    assert sample["metadata"]["validator"] == "rule+vlm"


def test_sharegpt_quality_review_uses_current_images(tmp_path) -> None:
    cfg = load_config()
    image_path = tmp_path / "images" / "pages" / "chunknown" / "p001.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    sample = _base_sample(
        "page_content_restatement",
        {"restatement": "页面展示了航空器结构设计中强度、刚度和重量约束之间的关系。"},
        ["images/pages/chunknown/p001.png"],
    )
    vlm = FakeVlm(
        _passing_review(
            [
                "text_format_quality",
                "qa_relevance",
                "visual_dependency",
                "image_question_alignment",
            ]
        )
    )

    valid = validate_sample(sample, tmp_path, cfg, vlm=vlm)

    assert valid is sample
    assert vlm.calls[0]["images"] == [image_path]
    assert set(sample["metadata"]["quality_checks"]) == {
        "text_format_quality",
        "qa_relevance",
        "visual_dependency",
        "image_question_alignment",
    }


def test_sharegpt_quality_review_rejects_low_visual_dependency(tmp_path) -> None:
    cfg = load_config()
    image_path = tmp_path / "images" / "pages" / "chunknown" / "p001.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    sample = _base_sample(
        "page_content_restatement",
        {"restatement": "页面展示了航空器结构设计中强度、刚度和重量约束之间的关系。"},
        ["images/pages/chunknown/p001.png"],
    )
    review = _passing_review(
        [
            "text_format_quality",
            "qa_relevance",
            "visual_dependency",
            "image_question_alignment",
        ]
    )
    review["is_valid"] = False
    review["quality_score"] = 0.52
    review["checks"]["visual_dependency"] = {
        "passed": False,
        "score": 0.2,
        "reason": "问题不需要当前图片也能回答",
    }
    vlm = FakeVlm(review)

    assert validate_sample(sample, tmp_path, cfg, vlm=vlm) is None
    assert sample["metadata"]["quality_checks"]["visual_dependency"]["passed"] is False
