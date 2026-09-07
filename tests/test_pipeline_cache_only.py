from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from book_cpt.app import pipeline
from book_cpt.app.cli import build_parser
from book_cpt.core.models import BookRecord, PipelineOptions
from book_cpt.processing.ingest import scan_cached_books
from book_cpt.services.mineru import (
    MinerUParseIncompleteError,
    mineru_cache_is_valid,
    parse_book_with_mineru,
)


class CacheOnlyDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {"paths": {"manifest": "manifest.jsonl"}}

    def test_restores_cached_book_without_source_pdf_and_rebinds_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            book_dir = output_root / "cached_book"
            book_dir.mkdir(parents=True)
            missing_pdf = Path(tmpdir) / "deleted.pdf"
            manifest = {
                "book_id": "cached-book",
                "book_name": "Cached Book",
                "source_pdf": str(missing_pdf),
                "output_dir": "/old/machine/outputs/cached_book",
                "page_count": 2,
                "file_size": 123,
                "file_hash": "old-hash",
                "status": "ready",
                "created_at": "2026-08-01T00:00:00Z",
                "pipeline_version": "1.0",
            }
            (book_dir / "manifest.jsonl").write_text(
                json.dumps(manifest, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            books = scan_cached_books(output_root, self.cfg)

            self.assertEqual(len(books), 1)
            self.assertEqual(books[0].book_id, "cached-book")
            self.assertEqual(Path(books[0].output_dir), book_dir)
            self.assertEqual(Path(books[0].source_pdf), missing_pdf)
            self.assertFalse(missing_pdf.exists())

    def test_raises_clear_error_when_output_root_has_no_cached_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            output_root.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, r"manifest\.jsonl"):
                scan_cached_books(output_root, self.cfg)

    def test_raises_clear_error_when_manifest_has_no_complete_book_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            book_dir = output_root / "broken_book"
            book_dir.mkdir(parents=True)
            manifest_path = book_dir / "manifest.jsonl"
            manifest_path.write_text('{"book_id": "broken"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, str(manifest_path).replace("\\", r"\\")):
                scan_cached_books(output_root, self.cfg)

    def test_restores_cached_book_from_mineru_dir_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "book_test"
            book_id = "V03356_manual"
            book_dir = output_root / book_id
            (book_dir / "mineru/raw").mkdir(parents=True)
            (book_dir / "mineru/parsed").mkdir(parents=True)
            content = [
                {"type": "text", "page_idx": 0, "text": "cached page one"},
                {"type": "text", "page_idx": 2, "text": "cached page three"},
            ]
            (book_dir / f"mineru/raw/{book_id}.json").write_text("{}", encoding="utf-8")
            (book_dir / f"mineru/parsed/{book_id}.json").write_text(
                json.dumps(content),
                encoding="utf-8",
            )
            (book_dir / "mineru/image_map.json").write_text("{}", encoding="utf-8")

            books = scan_cached_books(output_root, self.cfg)

            self.assertEqual(len(books), 1)
            self.assertEqual(books[0].book_id, book_id)
            self.assertEqual(books[0].book_name, book_id)
            self.assertEqual(Path(books[0].output_dir), book_dir)
            self.assertEqual(Path(books[0].source_pdf), book_dir / f"{book_id}.pdf")
            self.assertEqual(books[0].page_count, 3)
            self.assertFalse((book_dir / "manifest.jsonl").exists())

    def test_mineru_dir_with_invalid_parsed_json_raises_contextual_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "book_test"
            book_id = "V03356_manual"
            book_dir = output_root / book_id
            (book_dir / "mineru/raw").mkdir(parents=True)
            (book_dir / "mineru/parsed").mkdir(parents=True)
            (book_dir / f"mineru/raw/{book_id}.json").write_text("{}", encoding="utf-8")
            parsed_path = book_dir / f"mineru/parsed/{book_id}.json"
            parsed_path.write_text("{invalid", encoding="utf-8")
            (book_dir / "mineru/image_map.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"cache-only MinerU parsed cache invalid"):
                scan_cached_books(output_root, self.cfg)


class CacheOnlyOptionTests(unittest.TestCase):
    def test_parser_accepts_cache_only_without_input_dir(self) -> None:
        args = build_parser().parse_args(
            ['--output-root', 'outputs', '--task', 'domain_knowledge_corpus', '--cache-only']
        )
        self.assertTrue(args.cache_only)
        self.assertIsNone(args.input_dir)
        self.assertIsNone(args.book)

    def test_cache_only_forces_cache_reuse_and_disables_pdf_stages(self) -> None:
        cfg = {
            'runtime': {
                'cache_only': False,
                'reuse_mineru': False,
                'reuse_normalized': False,
                'render_pages': True,
                'crop_blocks': True,
            }
        }
        actual = pipeline._apply_options(cfg, PipelineOptions(cache_only=True))
        self.assertTrue(actual['runtime']['cache_only'])
        self.assertTrue(actual['runtime']['reuse_mineru'])
        self.assertTrue(actual['runtime']['reuse_normalized'])
        self.assertFalse(actual['runtime']['render_pages'])
        self.assertFalse(actual['runtime']['crop_blocks'])

    def test_cache_only_rejects_pdf_inputs_and_force_rebuild(self) -> None:
        conflicting = (
            PipelineOptions(cache_only=True, input_dir=Path('books')),
            PipelineOptions(cache_only=True, input_books=[{'path': 'book.pdf'}]),
            PipelineOptions(cache_only=True, force_rebuild=True),
        )
        for options in conflicting:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, 'cache-only'):
                    pipeline._validate_options(options)

    def test_configured_cache_only_rejects_force_rebuild_override(self) -> None:
        cfg = {
            'runtime': {
                'cache_only': True,
                'force_rebuild': False,
                'reuse_mineru': True,
                'reuse_normalized': True,
                'reuse_samples': True,
                'reuse_exports': True,
                'render_pages': False,
                'crop_blocks': False,
            }
        }

        with self.assertRaisesRegex(ValueError, 'cache-only'):
            pipeline._apply_options(cfg, PipelineOptions(force_rebuild=True))

    def test_configured_cache_only_rejects_pdf_input_overrides(self) -> None:
        cfg = {
            'runtime': {
                'cache_only': True,
                'force_rebuild': False,
                'reuse_mineru': True,
                'reuse_normalized': True,
                'render_pages': False,
                'crop_blocks': False,
            }
        }
        conflicting = (
            PipelineOptions(input_dir=Path('books')),
            PipelineOptions(input_books=[{'path': 'book.pdf'}]),
        )

        for options in conflicting:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, 'cache-only'):
                    pipeline._apply_options(cfg, options)


class CacheOnlyMinerUTests(unittest.TestCase):
    def test_cache_only_reuses_complete_cache_even_when_reuse_flag_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / 'outputs'
            output_dir = output_root / 'cached_book'
            (output_dir / 'mineru/raw').mkdir(parents=True)
            (output_dir / 'mineru/parsed').mkdir(parents=True)
            content = [{'type': 'text', 'page_idx': 0, 'text': '有效航空知识。' * 30}]
            (output_dir / 'mineru/raw/cached-book.json').write_text('{}', encoding='utf-8')
            (output_dir / 'mineru/parsed/cached-book.json').write_text(
                json.dumps(content, ensure_ascii=False),
                encoding='utf-8',
            )
            (output_dir / 'mineru/image_map.json').write_text('{}', encoding='utf-8')
            book = BookRecord(
                book_id='cached-book',
                book_name='Cached Book',
                source_pdf=str(Path(tmpdir) / 'deleted.pdf'),
                output_dir=str(output_dir),
                page_count=1,
                file_size=1,
                file_hash='hash',
                status='ready',
                created_at='2026-08-01T00:00:00Z',
                pipeline_version='1.0',
            )
            cfg = {
                'runtime': {
                    'cache_only': True,
                    'reuse_mineru': False,
                    'output_root': str(output_root),
                },
                'paths': {
                    'mineru_raw': 'mineru/raw/{book_id}.json',
                    'mineru_parsed': 'mineru/parsed/{book_id}.json',
                    'image_map': 'mineru/image_map.json',
                    'mineru_status': 'mineru/status/{book_id}.json',
                },
                'mineru': {},
            }

            with patch('book_cpt.services.mineru.MinerUClient') as client:
                result = parse_book_with_mineru(book, cfg)

            self.assertEqual(result.content, content)
            self.assertFalse(result.refreshed)
            client.assert_not_called()

    def test_cache_validity_rejects_invalid_raw_json_in_cache_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'outputs/cached_book'
            (output_dir / 'mineru/raw').mkdir(parents=True)
            (output_dir / 'mineru/parsed').mkdir(parents=True)
            content = [{'type': 'text', 'page_idx': 0, 'text': '有效航空知识。' * 30}]
            (output_dir / 'mineru/raw/cached-book.json').write_text('{invalid', encoding='utf-8')
            (output_dir / 'mineru/parsed/cached-book.json').write_text(
                json.dumps(content, ensure_ascii=False),
                encoding='utf-8',
            )
            (output_dir / 'mineru/image_map.json').write_text('{}', encoding='utf-8')
            book = BookRecord(
                book_id='cached-book',
                book_name='Cached Book',
                source_pdf=str(Path(tmpdir) / 'deleted.pdf'),
                output_dir=str(output_dir),
                page_count=1,
                file_size=1,
                file_hash='hash',
                status='ready',
                created_at='2026-08-01T00:00:00Z',
                pipeline_version='1.0',
            )
            cfg = {
                'runtime': {'cache_only': True},
                'paths': {
                    'mineru_raw': 'mineru/raw/{book_id}.json',
                    'mineru_parsed': 'mineru/parsed/{book_id}.json',
                    'image_map': 'mineru/image_map.json',
                    'mineru_status': 'mineru/status/{book_id}.json',
                },
                'mineru': {},
            }

            self.assertFalse(mineru_cache_is_valid(book, cfg))

    def test_cache_validity_requires_raw_payload_in_cache_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'outputs/cached_book'
            (output_dir / 'mineru/parsed').mkdir(parents=True)
            content = [{'type': 'text', 'page_idx': 0, 'text': '有效航空知识。' * 30}]
            (output_dir / 'mineru/parsed/cached-book.json').write_text(
                json.dumps(content, ensure_ascii=False),
                encoding='utf-8',
            )
            (output_dir / 'mineru/image_map.json').write_text('{}', encoding='utf-8')
            book = BookRecord(
                book_id='cached-book',
                book_name='Cached Book',
                source_pdf=str(Path(tmpdir) / 'deleted.pdf'),
                output_dir=str(output_dir),
                page_count=1,
                file_size=1,
                file_hash='hash',
                status='ready',
                created_at='2026-08-01T00:00:00Z',
                pipeline_version='1.0',
            )
            cfg = {
                'runtime': {'cache_only': True},
                'paths': {
                    'mineru_raw': 'mineru/raw/{book_id}.json',
                    'mineru_parsed': 'mineru/parsed/{book_id}.json',
                    'image_map': 'mineru/image_map.json',
                    'mineru_status': 'mineru/status/{book_id}.json',
                },
                'mineru': {},
            }

            self.assertFalse(mineru_cache_is_valid(book, cfg))

    def test_missing_cache_files_never_fall_back_to_mineru_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / 'outputs'
            output_dir = output_root / 'cached_book'
            output_dir.mkdir(parents=True)
            book = BookRecord(
                book_id='cached-book',
                book_name='Cached Book',
                source_pdf=str(Path(tmpdir) / 'deleted.pdf'),
                output_dir=str(output_dir),
                page_count=1,
                file_size=1,
                file_hash='hash',
                status='ready',
                created_at='2026-08-01T00:00:00Z',
                pipeline_version='1.0',
            )
            cfg = {
                'runtime': {
                    'cache_only': True,
                    'reuse_mineru': True,
                    'output_root': str(output_root),
                },
                'paths': {
                    'mineru_raw': 'mineru/raw/{book_id}.json',
                    'mineru_parsed': 'mineru/parsed/{book_id}.json',
                    'image_map': 'mineru/image_map.json',
                    'mineru_status': 'mineru/status/{book_id}.json',
                },
                'mineru': {},
            }

            with patch('book_cpt.services.mineru.MinerUClient') as client:
                client.return_value.parse_pdf.side_effect = AssertionError('MinerU must not be called')
                with self.assertRaisesRegex(FileNotFoundError, 'cache-only'):
                    parse_book_with_mineru(book, cfg)

            client.assert_not_called()

    def test_invalid_cache_never_falls_back_to_mineru_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / 'outputs'
            output_dir = output_root / 'cached_book'
            (output_dir / 'mineru/raw').mkdir(parents=True)
            (output_dir / 'mineru/parsed').mkdir(parents=True)
            (output_dir / 'mineru/raw/cached-book.json').write_text('{}', encoding='utf-8')
            (output_dir / 'mineru/parsed/cached-book.json').write_text('{}', encoding='utf-8')
            (output_dir / 'mineru/image_map.json').write_text('{}', encoding='utf-8')
            book = BookRecord(
                book_id='cached-book',
                book_name='Cached Book',
                source_pdf=str(Path(tmpdir) / 'deleted.pdf'),
                output_dir=str(output_dir),
                page_count=1,
                file_size=1,
                file_hash='hash',
                status='ready',
                created_at='2026-08-01T00:00:00Z',
                pipeline_version='1.0',
            )
            cfg = {
                'runtime': {'cache_only': True, 'reuse_mineru': True, 'output_root': str(output_root)},
                'paths': {
                    'mineru_raw': 'mineru/raw/{book_id}.json',
                    'mineru_parsed': 'mineru/parsed/{book_id}.json',
                    'image_map': 'mineru/image_map.json',
                    'mineru_status': 'mineru/status/{book_id}.json',
                },
                'mineru': {},
            }

            with patch('book_cpt.services.mineru.MinerUClient') as client:
                with self.assertRaisesRegex(MinerUParseIncompleteError, 'cache-only'):
                    parse_book_with_mineru(book, cfg)

            client.assert_not_called()


class CacheOnlyIntegrationTests(unittest.TestCase):
    def test_generates_domain_corpus_from_output_cache_without_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / 'outputs'
            book_dir = output_root / 'cached_book'
            book_id = 'cached-book'
            (book_dir / 'mineru/raw').mkdir(parents=True)
            (book_dir / 'mineru/parsed').mkdir(parents=True)
            missing_pdf = Path(tmpdir) / 'deleted.pdf'
            manifest = {
                'book_id': book_id,
                'book_name': 'Cached Book',
                'source_pdf': str(missing_pdf),
                'output_dir': '/old/outputs/cached_book',
                'page_count': 1,
                'file_size': 100,
                'file_hash': 'source-hash',
                'status': 'ready',
                'created_at': '2026-08-01T00:00:00Z',
                'pipeline_version': 'book_prompt_v1',
            }
            text = '\n'.join(
                f'第{index}节介绍航空发动机系统的结构、工作原理和运行限制。'
                f'关键参数{index}用于判断部件状态并支持安全维护决策。'
                for index in range(1, 81)
            )
            content = [{'type': 'text', 'page_idx': 0, 'text': text}]
            (book_dir / 'manifest.jsonl').write_text(
                json.dumps(manifest, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            (book_dir / f'mineru/raw/{book_id}.json').write_text('{}', encoding='utf-8')
            (book_dir / f'mineru/parsed/{book_id}.json').write_text(
                json.dumps(content, ensure_ascii=False),
                encoding='utf-8',
            )
            (book_dir / 'mineru/image_map.json').write_text('{}', encoding='utf-8')

            with patch.object(pipeline, 'scan_books', side_effect=AssertionError('PDF scan is forbidden')):
                with patch.object(pipeline, '_build_vlm_pool', side_effect=AssertionError('VLM pool is forbidden')):
                    with patch('book_cpt.services.mineru.MinerUClient') as mineru_client:
                        results = pipeline.run_pipeline(
                            PipelineOptions(
                                output_root=output_root,
                                cache_only=True,
                                book_workers=1,
                                max_workers=1,
                                task_types=['domain_knowledge_corpus'],
                            )
                        )

            self.assertFalse(missing_pdf.exists())
            self.assertNotIn('error', results[0])
            mineru_client.assert_not_called()
            export_path = book_dir / 'exports/pt/domain_knowledge_corpus.jsonl'
            exported = [json.loads(line) for line in export_path.read_text(encoding='utf-8').splitlines()]
            self.assertGreater(len(exported), 0)
            self.assertEqual(set(exported[0]), {'text'})
            self.assertTrue(exported[0]['text'].strip())

    def test_generates_domain_corpus_from_mineru_book_dir_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / 'book_test'
            book_id = 'V03356_manual'
            book_dir = output_root / book_id
            (book_dir / 'mineru/raw').mkdir(parents=True)
            (book_dir / 'mineru/parsed').mkdir(parents=True)
            text = '\n'.join(
                f'Aircraft system cached maintenance paragraph {index} describes components, limits, checks, and operating procedures.'
                for index in range(1, 90)
            )
            content = [{'type': 'text', 'page_idx': 0, 'text': text}]
            (book_dir / f'mineru/raw/{book_id}.json').write_text('{}', encoding='utf-8')
            (book_dir / f'mineru/parsed/{book_id}.json').write_text(
                json.dumps(content),
                encoding='utf-8',
            )
            (book_dir / 'mineru/image_map.json').write_text('{}', encoding='utf-8')

            with patch.object(pipeline, 'scan_books', side_effect=AssertionError('PDF scan is forbidden')):
                with patch.object(pipeline, '_build_vlm_pool', side_effect=AssertionError('VLM pool is forbidden')):
                    with patch('book_cpt.services.mineru.MinerUClient') as mineru_client:
                        results = pipeline.run_pipeline(
                            PipelineOptions(
                                output_root=output_root,
                                cache_only=True,
                                book_workers=1,
                                max_workers=1,
                                task_types=['domain_knowledge_corpus'],
                            )
                        )

            self.assertNotIn('error', results[0])
            mineru_client.assert_not_called()
            self.assertTrue((book_dir / 'manifest.jsonl').exists())
            export_path = book_dir / 'exports/pt/domain_knowledge_corpus.jsonl'
            exported = [json.loads(line) for line in export_path.read_text(encoding='utf-8').splitlines()]
            self.assertGreater(len(exported), 0)
            self.assertEqual(set(exported[0]), {'text'})
            self.assertTrue(exported[0]['text'].strip())


if __name__ == '__main__':
    unittest.main()
