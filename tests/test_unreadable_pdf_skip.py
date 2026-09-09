from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from book_cpt.core.models import BookRecord
from book_cpt.processing.watermark import UnreadablePdfError, clean_pdf_watermarks


def _book(tmpdir: str, pdf: Path) -> BookRecord:
    return BookRecord(
        book_id="broken",
        book_name="broken",
        source_pdf=str(pdf),
        output_dir=str(Path(tmpdir) / "out" / "broken"),
        page_count=0,
        file_size=pdf.stat().st_size,
        file_hash="h",
        status="pending",
        created_at="t",
        pipeline_version="test",
    )


def _minimal_pdf() -> bytes:
    """最小可解析 PDF，用来验证"没坏的文件不会被误伤"。"""
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _cfg(tmpdir: str) -> dict:
    return {
        "runtime": {"output_root": str(Path(tmpdir) / "out"), "skip_vlm": True},
        "watermark": {"enabled": False},
        "hash_chunk_size": 65536,
        "paths": {"skipped_books": "skipped_books.jsonl", "manifest": "manifest.jsonl"},
        "logger_name": "book_cpt_test",
        "statuses": {"done": "done"},
    }


class UnreadablePdfDetectionTests(unittest.TestCase):
    def test_truncated_pdf_raises_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "truncated.pdf"
            # 有 PDF 头但没有 %%EOF —— 就是用户遇到的那种截断文件
            pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n")
            with self.assertRaises(UnreadablePdfError) as ctx:
                clean_pdf_watermarks(_book(tmp, pdf), _cfg(tmp))
            self.assertIn("PdfStreamError", str(ctx.exception))

    def test_empty_file_raises_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "empty.pdf"
            pdf.write_bytes(b"")
            with self.assertRaises(UnreadablePdfError):
                clean_pdf_watermarks(_book(tmp, pdf), _cfg(tmp))

    def test_disabled_watermark_cleaning_still_detects_broken_pdf(self) -> None:
        # 书籍侧去水印默认关闭，但坏文件仍要在这一步被拦下来，
        # 否则要到 MinerU 才炸，还会白白重试三次。
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "truncated.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            cfg = _cfg(tmp)
            cfg["watermark"]["enabled"] = False
            with self.assertRaises(UnreadablePdfError):
                clean_pdf_watermarks(_book(tmp, pdf), cfg)

    def test_disabled_watermark_cleaning_leaves_a_good_pdf_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "ok.pdf"
            pdf.write_bytes(_minimal_pdf())
            cfg = _cfg(tmp)
            cfg["watermark"]["enabled"] = False
            result = clean_pdf_watermarks(_book(tmp, pdf), cfg)
            self.assertFalse(result.cleaned)
            self.assertEqual(result.cleaned_pdf, str(pdf))


class PipelineSkipTests(unittest.TestCase):
    def test_unreadable_pdf_is_skipped_not_failed(self) -> None:
        from book_cpt.app import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "truncated.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            book = _book(tmp, pdf)
            cfg = _cfg(tmp)

            result = pipeline._process_book(book, cfg)

            self.assertEqual(result["skipped"], "unreadable_pdf")
            self.assertNotIn("error", result)
            self.assertEqual(result["sample_count"], 0)
            self.assertIn("PdfStreamError", result["skip_reason"])

    def test_skipped_books_are_recorded_and_counted_separately(self) -> None:
        from book_cpt.app import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            output_root.mkdir(parents=True)
            books = [
                SimpleNamespace(book_id="good", output_dir=str(output_root / "good"), source_pdf="a.pdf"),
                SimpleNamespace(book_id="bad", output_dir=str(output_root / "bad"), source_pdf="b.pdf"),
            ]

            def fake_process(book, cfg, vlm=None):
                if book.book_id == "bad":
                    return {
                        "book_id": "bad",
                        "output_dir": book.output_dir,
                        "sample_count": 0,
                        "skipped": "unreadable_pdf",
                        "skip_reason": "PdfStreamError: truncated",
                        "source_pdf": book.source_pdf,
                    }
                return {"book_id": "good", "output_dir": book.output_dir, "sample_count": 3}

            cfg = {
                "runtime": {
                    "input_dir": str(tmp), "output_root": str(output_root), "book_workers": 1,
                    "skip_vlm": True, "progress": False, "recursive": True,
                },
                "paths": {"skipped_books": "skipped_books.jsonl"},
            }

            with patch.object(pipeline, "_apply_options", return_value=cfg), patch.object(
                pipeline, "_resolve_config_runtime_paths", return_value=cfg
            ), patch.object(pipeline, "scan_books", return_value=books), patch.object(
                pipeline, "_process_book", side_effect=fake_process
            ), patch.object(pipeline, "load_config", return_value=cfg):
                results = pipeline.run_pipeline()

            self.assertEqual(len(results), 2)
            skipped = [item for item in results if item.get("skipped")]
            failed = [item for item in results if item.get("error")]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(len(failed), 0, "损坏文件不该记进失败统计")

            log = output_root / "skipped_books.jsonl"
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["book_id"], "bad")
            self.assertEqual(rows[0]["source_pdf"], "b.pdf")
            self.assertIn("truncated", rows[0]["reason"])


if __name__ == "__main__":
    unittest.main()


class ShippedWatermarkConfigTests(unittest.TestCase):
    def test_watermark_cleaning_is_enabled_by_default(self) -> None:
        from book_cpt.core.config_loader import load_config
        from book_cpt.processing.watermark import watermark_cleaning_enabled

        self.assertTrue(watermark_cleaning_enabled(load_config()))

    def test_plain_pdf_is_passed_through_untouched(self) -> None:
        """没有水印的书不该被改写，也不该因此让缓存失效。"""
        from book_cpt.core.config_loader import load_config
        from book_cpt.processing.watermark import clean_pdf_watermarks

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "plain.pdf"
            pdf.write_bytes(_minimal_pdf())
            cfg = load_config()
            cfg["runtime"]["output_root"] = str(Path(tmp) / "out")
            book = _book(tmp, pdf)
            result = clean_pdf_watermarks(book, cfg)
            self.assertFalse(result.cleaned)
            self.assertEqual(result.cleaned_pdf, str(pdf))
            self.assertEqual(result.candidate_names, [])
            self.assertEqual(result.content_hash, book.file_hash)
