from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from book_cpt.core.config_loader import load_config
from book_cpt.core.io_utils import read_json
from book_cpt.core.models import BookRecord
from book_cpt.processing.watermark import clean_pdf_watermarks


def _watermarked_pdf(path: Path, pages: int = 6, stamped: int | None = None) -> None:
    """造一本盖着 /OC 水印层的 PDF；stamped 指定盖在前几页上（默认每页都盖）。"""
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
        TextStringObject,
    )

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)

    ocg = DictionaryObject(
        {NameObject("/Type"): NameObject("/OCG"), NameObject("/Name"): TextStringObject("Watermark")}
    )
    ocg_ref = writer._add_object(ocg)
    form = DecodedStreamObject()
    form.set_data(b"q 1 0 0 1 0 0 cm 0.9 g 10 10 100 40 re f Q\n")
    form[NameObject("/Type")] = NameObject("/XObject")
    form[NameObject("/Subtype")] = NameObject("/Form")
    form[NameObject("/BBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), NumberObject(200), NumberObject(200)]
    )
    form[NameObject("/OC")] = ocg_ref
    form_ref = writer._add_object(form)

    stamp_upto = pages if stamped is None else stamped
    for index, page in enumerate(writer.pages):
        if index >= stamp_upto:
            continue
        resources = page[NameObject("/Resources")]
        xobject = DictionaryObject()
        xobject[NameObject("/Fm0")] = form_ref
        resources[NameObject("/XObject")] = xobject
        content = DecodedStreamObject()
        content.set_data(b"q /Fm0 Do Q\n")
        page[NameObject("/Contents")] = writer._add_object(content)

    with path.open("wb") as handle:
        writer.write(handle)


class WatermarkIdempotencyTests(unittest.TestCase):
    """去水印必须是幂等的。

    原来每轮都重扫源 PDF、重写产物、返回 cleaned=True，而 _process_book 一看到
    cleaned=True 就把 reuse_mineru/normalized/samples/exports 全关掉 ——
    带水印的书于是每轮都从头重跑 MinerU 和全部样本，断点续跑形同虚设。
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pdf = Path(self.tmp) / "wm.pdf"
        _watermarked_pdf(self.pdf)
        self.output_root = Path(self.tmp) / "out"
        self.book_dir = self.output_root / "book1"
        self.book_dir.mkdir(parents=True)
        self.book = BookRecord(
            book_id="book1",
            book_name="b",
            source_pdf=str(self.pdf),
            output_dir=str(self.book_dir),
            page_count=6,
            file_size=os.path.getsize(self.pdf),
            file_hash="source-hash-1",
            status="ready",
            created_at="t",
            pipeline_version="book_prompt_v1",
        )

    def _cfg(self, **watermark):
        cfg = load_config()
        cfg["runtime"]["output_root"] = str(self.output_root)
        cfg["watermark"].update(watermark)
        return cfg

    def test_first_run_cleans_and_reports_a_real_change(self) -> None:
        result = clean_pdf_watermarks(self.book, self._cfg())
        self.assertTrue(result.cleaned)
        self.assertFalse(result.reused)
        self.assertEqual(result.candidate_names, ["/Fm0"])
        self.assertEqual(result.removed_invocations, 6)
        self.assertTrue(Path(result.cleaned_pdf).is_file())

    def test_later_runs_reuse_instead_of_re_cleaning(self) -> None:
        cfg = self._cfg()
        first = clean_pdf_watermarks(self.book, cfg)
        mtime = Path(first.cleaned_pdf).stat().st_mtime_ns
        for _ in range(3):
            again = clean_pdf_watermarks(self.book, cfg)
            self.assertTrue(again.cleaned)
            self.assertTrue(again.reused, "沿用上一轮产物时必须置 reused，否则上游会作废缓存")
            self.assertEqual(again.cleaned_pdf, first.cleaned_pdf)
            self.assertEqual(again.content_hash, first.content_hash)
        self.assertEqual(Path(first.cleaned_pdf).stat().st_mtime_ns, mtime, "产物不该被重写")

    def test_a_genuinely_different_pdf_is_not_treated_as_reuse(self) -> None:
        """源 PDF 真的换了内容（清理产物跟着变）才该报 reused=False。"""
        cfg = self._cfg()
        first = clean_pdf_watermarks(self.book, cfg)
        _watermarked_pdf(self.pdf, pages=9)  # 同一路径，换成另一本书
        replaced = replace(self.book, file_hash="source-hash-2", page_count=9)
        result = clean_pdf_watermarks(replaced, cfg)
        self.assertFalse(result.reused)
        self.assertNotEqual(result.content_hash, first.content_hash)

    def test_same_bytes_under_a_new_source_hash_still_counts_as_reuse(self) -> None:
        """快速路径失配（比如报告是旧版本写的、没记 source_hash），
        但重扫出来的产物逐字节相同 —— 缓存仍然对应同一份 PDF，不该作废。"""
        cfg = self._cfg()
        first = clean_pdf_watermarks(self.book, cfg)
        result = clean_pdf_watermarks(replace(self.book, file_hash="written-by-old-version"), cfg)
        self.assertTrue(result.reused)
        self.assertEqual(result.content_hash, first.content_hash)

    def test_config_change_that_flips_the_verdict_is_not_reuse(self) -> None:
        # 水印只盖在 6 页里的前 3 页：覆盖率阈值 0.4 判定为水印，0.9 则不算
        _watermarked_pdf(self.pdf, pages=6, stamped=3)
        book = replace(self.book, file_hash="partial-stamp")
        cleaned = clean_pdf_watermarks(book, self._cfg(min_page_coverage=0.4))
        self.assertTrue(cleaned.cleaned)
        result = clean_pdf_watermarks(book, self._cfg(min_page_coverage=0.9))
        self.assertFalse(result.cleaned)
        self.assertFalse(result.reused)

    def test_deleted_cleaned_pdf_is_regenerated_and_reuse_still_holds(self) -> None:
        cfg = self._cfg()
        first = clean_pdf_watermarks(self.book, cfg)
        Path(first.cleaned_pdf).unlink()
        result = clean_pdf_watermarks(self.book, cfg)
        self.assertTrue(Path(result.cleaned_pdf).is_file(), "产物要重新生成")
        self.assertEqual(result.content_hash, first.content_hash)
        self.assertTrue(result.reused, "重建出来逐字节相同，下游缓存仍然有效")

    def test_clean_book_also_skips_the_rescan_on_later_runs(self) -> None:
        """没水印的书更要省：不然每轮都要把每页内容流完整解析一遍。"""
        plain = Path(self.tmp) / "plain.pdf"
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with plain.open("wb") as handle:
            writer.write(handle)
        book = replace(self.book, source_pdf=str(plain), file_hash="plain-hash")

        cfg = self._cfg()
        first = clean_pdf_watermarks(book, cfg)
        self.assertFalse(first.cleaned)
        self.assertFalse(first.reused)
        again = clean_pdf_watermarks(book, cfg)
        self.assertFalse(again.cleaned)
        self.assertTrue(again.reused)
        self.assertEqual(again.cleaned_pdf, str(plain))

    def test_report_records_what_reuse_is_judged_on(self) -> None:
        cfg = self._cfg()
        clean_pdf_watermarks(self.book, cfg)
        report = read_json(self.book_dir / "preprocessed" / "watermark_cleaning.json")
        self.assertEqual(report["source_hash"], "source-hash-1")
        self.assertTrue(report["config_signature"])
        self.assertFalse(report["reused"])

    def test_report_reflects_the_latest_run_not_a_stale_one(self) -> None:
        """排查时会直接看这个文件，不能留着上一轮 reused=False 的旧记录。"""
        cfg = self._cfg()
        clean_pdf_watermarks(self.book, cfg)
        clean_pdf_watermarks(self.book, cfg)
        report = read_json(self.book_dir / "preprocessed" / "watermark_cleaning.json")
        self.assertTrue(report["reused"])
        self.assertTrue(report["cleaned"])


class PipelineCacheInvalidationTests(unittest.TestCase):
    """cleaned + reused 不该作废缓存；cleaned + 本轮真改过 才该。"""

    def _run_flags(self, *, cleaned: bool, reused: bool) -> dict:
        from types import SimpleNamespace
        from unittest.mock import patch

        from book_cpt.app import pipeline
        from book_cpt.processing.watermark import WatermarkCleanResult

        captured: dict = {}

        def fake_render(book, cfg):
            captured["reuse_mineru"] = bool(cfg["runtime"]["reuse_mineru"])
            captured["reuse_samples"] = bool(cfg["runtime"]["reuse_samples"])
            captured["rerender_pages"] = bool(cfg["runtime"].get("rerender_pages"))
            captured["source_pdf"] = book.source_pdf
            raise RuntimeError("stop here")

        result = WatermarkCleanResult(
            source_pdf="/x/orig.pdf",
            cleaned_pdf="/x/cleaned.pdf" if cleaned else "/x/orig.pdf",
            cleaned=cleaned,
            candidate_names=["/Fm0"] if cleaned else [],
            removed_invocations=6 if cleaned else 0,
            page_count=6,
            content_hash="h",
            reused=reused,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config()
            cfg["runtime"]["output_root"] = tmp
            cfg["runtime"]["skip_vlm"] = True
            cfg["runtime"]["reuse_exports"] = False
            cfg["runtime"]["reuse_mineru"] = True  # 等价于命令行 --reuse-mineru
            book = BookRecord(
                book_id="b1", book_name="b", source_pdf="/x/orig.pdf",
                output_dir=str(Path(tmp) / "b1"), page_count=6, file_size=1,
                file_hash="h", status="ready", created_at="t", pipeline_version="book_prompt_v1",
            )
            with patch.object(pipeline, "clean_pdf_watermarks", return_value=result), patch.object(
                pipeline, "render_pages", side_effect=fake_render
            ):
                pipeline._process_book(book, cfg)
        return captured

    def test_reused_cleaning_keeps_the_caches(self) -> None:
        flags = self._run_flags(cleaned=True, reused=True)
        self.assertTrue(flags["reuse_mineru"])
        self.assertTrue(flags["reuse_samples"])
        self.assertFalse(flags["rerender_pages"])
        self.assertEqual(flags["source_pdf"], "/x/cleaned.pdf", "仍要用清理后的 PDF")

    def test_fresh_cleaning_invalidates_the_caches(self) -> None:
        flags = self._run_flags(cleaned=True, reused=False)
        self.assertFalse(flags["reuse_mineru"])
        self.assertFalse(flags["reuse_samples"])
        self.assertTrue(flags["rerender_pages"])

    def test_no_watermark_leaves_everything_alone(self) -> None:
        flags = self._run_flags(cleaned=False, reused=False)
        self.assertTrue(flags["reuse_mineru"])
        self.assertEqual(flags["source_pdf"], "/x/orig.pdf")


if __name__ == "__main__":
    unittest.main()
