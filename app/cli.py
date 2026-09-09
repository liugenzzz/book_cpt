from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from book_cpt.app.pipeline import run_pipeline
    from book_cpt.core.models import PipelineOptions
else:
    from .pipeline import run_pipeline
    from ..core.models import PipelineOptions


def parse_book_spec(value: str) -> dict[str, str]:
    if "=" not in value:
        return {"path": value}
    name, path = value.split("=", 1)
    return {"book_name": name.strip(), "path": path.strip()}


def parse_task_specs(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    task_types: list[str] = []
    for value in values:
        for item in value.split(","):
            task_type = item.strip()
            if task_type:
                task_types.append(task_type)
    return task_types or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument(
        "--book",
        action="append",
        default=None,
        help="指定输入书籍，格式为 书籍名称=PDF路径；可重复传入。也可只传 PDF路径。",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Discover books from output-root manifests and never read source PDFs.",
    )
    parser.add_argument("--reuse-mineru", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--book-workers", type=int, default=None, help="书籍级并发数量。")
    parser.add_argument("--max-workers", type=int, default=None, help="样本生成并发数量。")
    parser.add_argument("--page-workers", type=int, default=None, help="页面渲染并发数量。")
    parser.add_argument("--crop-workers", type=int, default=None, help="页面裁剪并发数量。")
    parser.add_argument("--mineru-workers", type=int, default=None, help="MinerU 解析并发数量。")
    parser.add_argument("--mineru-retry-count", type=int, default=None, help="MinerU 失败重试次数。")
    parser.add_argument("--mineru-min-page-coverage", type=float, default=None, help="MinerU 最小页覆盖率。")
    parser.add_argument("--no-reuse-normalized", action="store_true", help="不复用 normalized 页面缓存。")
    parser.add_argument("--no-reuse-samples", action="store_true", help="不复用已去重样本缓存。")
    parser.add_argument("--no-reuse-exports", action="store_true", help="不复用已导出文件。")
    parser.add_argument("--force-rebuild", action="store_true", help="重新构建 MinerU 之后的全部缓存。")
    parser.add_argument("--min-page-text-chars", type=int, default=None, help="整页任务最小文本长度。")
    parser.add_argument("--min-block-text-chars", type=int, default=None, help="局部块任务最小文本长度。")
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=None,
        help="只生成指定任务类型；可重复传入，也可使用逗号分隔。默认生成 config.py 中启用的全部任务。",
    )
    parser.add_argument(
        "--tasks",
        dest="tasks",
        action="append",
        help="同 --task，例如 --tasks image_caption,page_ocr_qa。",
    )
    parser.add_argument("--limit-books", type=int, default=None)
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=None,
        help="递归扫描 input-dir 下所有子目录的 PDF。",
    )
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="只扫描 input-dir 一级目录。")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别，默认 INFO；DEBUG 会打出 metric/checkpoint 明细。",
    )
    parser.add_argument("--no-progress", dest="progress", action="store_false", default=None, help="关闭进度条。")
    parser.add_argument("--quiet", action="store_true", help="只显示进度条和错误（等价于 --log-level WARNING）。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_pipeline(
        PipelineOptions(
            config_path=Path(args.config) if args.config else None,
            input_dir=Path(args.input_dir) if args.input_dir else None,
            input_books=[parse_book_spec(item) for item in args.book] if args.book else None,
            output_root=Path(args.output_root) if args.output_root else None,
            cache_only=True if args.cache_only else None,
            reuse_mineru=False if args.force_rebuild else (True if args.reuse_mineru else None),
            skip_vlm=True if args.skip_vlm else None,
            book_workers=args.book_workers,
            max_workers=args.max_workers,
            page_workers=args.page_workers,
            crop_workers=args.crop_workers,
            mineru_workers=args.mineru_workers,
            mineru_retry_count=args.mineru_retry_count,
            mineru_min_page_coverage=args.mineru_min_page_coverage,
            reuse_normalized=False if args.force_rebuild or args.no_reuse_normalized else None,
            reuse_samples=False if args.force_rebuild or args.no_reuse_samples else None,
            reuse_exports=False if args.force_rebuild or args.no_reuse_exports else None,
            force_rebuild=True if args.force_rebuild else None,
            min_page_text_chars=args.min_page_text_chars,
            min_block_text_chars=args.min_block_text_chars,
            task_types=parse_task_specs(args.tasks),
            limit_books=args.limit_books,
            recursive=args.recursive,
            log_level="WARNING" if args.quiet else args.log_level,
            progress=args.progress,
        )
    )
    skipped = [item for item in results if item.get("skipped")]
    failed = [item for item in results if item.get("error")]
    ok = [item for item in results if not item.get("error") and not item.get("skipped")]
    total_samples = sum(int(item.get("sample_count", 0)) for item in results)
    print(f"\n完成 {len(ok)}/{len(results)} 本书，样本 {total_samples} 条", file=sys.stderr)
    if skipped:
        print(f"跳过 {len(skipped)} 本（文件损坏，已记入 skipped_books.jsonl）:", file=sys.stderr)
        for item in skipped[:10]:
            print(f"  - {item.get('book_id')}: {str(item.get('skip_reason'))[:120]}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... 另有 {len(skipped) - 10} 本", file=sys.stderr)
    if failed:
        print(f"失败 {len(failed)} 本:", file=sys.stderr)
        for item in failed[:20]:
            print(f"  - {item.get('book_id')}: {str(item.get('error'))[:160]}", file=sys.stderr)
        if len(failed) > 20:
            print(f"  ... 另有 {len(failed) - 20} 本", file=sys.stderr)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
