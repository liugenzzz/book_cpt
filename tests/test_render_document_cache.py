from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from book_cpt.processing import render


class _FakePixmap:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"png")


class _FakePage:
    def get_pixmap(self, **_kwargs: object) -> _FakePixmap:
        return _FakePixmap()


class _FakeDocument:
    def __init__(self) -> None:
        self.closed = False
        self.loaded: list[int] = []

    def load_page(self, index: int) -> _FakePage:
        self.loaded.append(index)
        return _FakePage()

    def close(self) -> None:
        self.closed = True


class _FakeFitz(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("fitz")
        self.open_calls: list[str] = []
        self.documents: list[_FakeDocument] = []
        self.TOOLS = types.SimpleNamespace(
            mupdf_display_errors=lambda _value: None,
            mupdf_display_warnings=lambda _value: None,
            mupdf_warnings=lambda: "",
        )

    def open(self, path: str) -> _FakeDocument:  # noqa: A003 - 对齐 fitz 的接口名
        self.open_calls.append(str(path))
        document = _FakeDocument()
        self.documents.append(document)
        return document

    def Matrix(self, *_args: float) -> object:  # noqa: N802 - 对齐 fitz 的接口名
        return object()


def _cfg() -> dict:
    return {
        "paths": {"page_images": "images/pages/ch{chapter_no}/p{page_no:03d}.png"},
        "default_chapter_no": "1",
        "render": {"dpi": 180},
        "runtime": {"rerender_pages": False},
    }


class DocumentCacheTests(unittest.TestCase):
    """原来每渲染一页就 fitz.open 一整本：N 页解析 N 遍文档结构，
    页树有瑕疵的书还会每次往 stderr 刷一条 MuPDF error。"""

    def setUp(self) -> None:
        render._DOCUMENT_CACHE.clear()
        self.fake = _FakeFitz()
        patcher = patch.dict(sys.modules, {"fitz": self.fake})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(render._DOCUMENT_CACHE.clear)

    def _render(self, tmp: str, pages: range) -> None:
        for index in pages:
            render._render_page_worker(("/x/book.pdf", tmp, _cfg(), index, {}))

    def test_same_pdf_is_opened_once_across_many_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._render(tmp, range(40))
        self.assertEqual(self.fake.open_calls, ["/x/book.pdf"])
        self.assertEqual(self.fake.documents[0].loaded, list(range(40)))

    def test_every_page_image_is_still_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._render(tmp, range(5))
            written = sorted(p.name for p in (Path(tmp) / "images/pages/ch1").glob("*.png"))
        self.assertEqual(written, [f"p{index:03d}.png" for index in range(1, 6)])

    def test_a_second_pdf_gets_its_own_handle(self) -> None:
        # 每本书有自己的 output_dir，页图路径不会互相撞上 skip 分支
        with tempfile.TemporaryDirectory() as tmp:
            out_a, out_b = str(Path(tmp) / "a"), str(Path(tmp) / "b")
            render._render_page_worker(("/x/a.pdf", out_a, _cfg(), 0, {}))
            render._render_page_worker(("/x/b.pdf", out_b, _cfg(), 0, {}))
            render._render_page_worker(("/x/a.pdf", out_a, _cfg(), 1, {}))
        self.assertEqual(self.fake.open_calls, ["/x/a.pdf", "/x/b.pdf"])
        self.assertEqual(self.fake.documents[0].loaded, [0, 1])
        self.assertEqual(self.fake.documents[1].loaded, [0])

    def test_existing_page_is_skipped_without_opening_the_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "images/pages/ch1/p001.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"old")
            _, _, status, _ = render._render_page_worker(("/x/book.pdf", tmp, _cfg(), 0, {}))
        self.assertEqual(status, "skipped")
        self.assertEqual(self.fake.open_calls, [])


class SilenceMupdfTests(unittest.TestCase):
    def test_toggle_is_called_when_enabled(self) -> None:
        calls: list[object] = []
        fake = _FakeFitz()
        fake.TOOLS.mupdf_display_errors = lambda value: calls.append(value)
        with patch.dict(sys.modules, {"fitz": fake}):
            render.silence_mupdf_chatter(True)
        self.assertEqual(calls, [False])

    def test_nothing_happens_when_disabled(self) -> None:
        calls: list[object] = []
        fake = _FakeFitz()
        fake.TOOLS.mupdf_display_errors = lambda value: calls.append(value)
        with patch.dict(sys.modules, {"fitz": fake}):
            render.silence_mupdf_chatter(False)
        self.assertEqual(calls, [])

    def test_old_pymupdf_without_the_api_does_not_crash(self) -> None:
        fake = _FakeFitz()
        fake.TOOLS = types.SimpleNamespace()
        with patch.dict(sys.modules, {"fitz": fake}):
            render.silence_mupdf_chatter(True)
            self.assertEqual(render.drain_mupdf_warnings(), "")


if __name__ == "__main__":
    unittest.main()
