from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.config_loader import deep_merge, load_config, package_root
from ..core.io_utils import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from ..core.logging_utils import RunState, configure_logger
from ..core.models import PipelineOptions
from ..processing.crop import crop_blocks
from ..processing.ingest import scan_books, scan_cached_books, scan_input_books
from ..processing.normalize import normalize_pages
from ..processing.render import render_pages
from ..services.clients import VlmPool
from ..services.mineru import mineru_cache_is_valid, parse_book_with_mineru
from ..tasks.dedup import deduplicate
from ..tasks.exporters import export_task_files
from ..tasks.generation import generate_for_job
from ..tasks.routing import build_sample_jobs
from ..tasks.validation import validate_sample


def _validate_options(options: PipelineOptions) -> None:
    if not options.cache_only:
        return
    conflicts: list[str] = []
    if options.input_dir is not None:
        conflicts.append('--input-dir')
    if options.input_books is not None:
        conflicts.append('--book')
    if options.force_rebuild:
        conflicts.append('--force-rebuild')
    if conflicts:
        joined = ', '.join(conflicts)
        raise ValueError(f'--cache-only cannot be combined with {joined}')


def _apply_options(cfg: dict[str, Any], options: PipelineOptions) -> dict[str, Any]:
    override: dict[str, Any] = {"runtime": {}}
    if options.input_dir is not None:
        override["runtime"]["input_dir"] = str(options.input_dir)
    if options.input_books is not None:
        override["runtime"]["input_books"] = options.input_books
    if options.output_root is not None:
        override["runtime"]["output_root"] = str(options.output_root)
    if options.cache_only is not None:
        override['runtime']['cache_only'] = bool(options.cache_only)
    if options.reuse_mineru is not None:
        override["runtime"]["reuse_mineru"] = bool(options.reuse_mineru)
    if options.skip_vlm is not None:
        override["runtime"]["skip_vlm"] = bool(options.skip_vlm)
    if options.book_workers is not None:
        override["runtime"]["book_workers"] = int(options.book_workers)
    if options.max_workers is not None:
        override["runtime"]["max_workers"] = int(options.max_workers)
    if options.page_workers is not None:
        override["runtime"]["page_workers"] = int(options.page_workers)
    if options.crop_workers is not None:
        override["runtime"]["crop_workers"] = int(options.crop_workers)
    if options.reuse_normalized is not None:
        override["runtime"]["reuse_normalized"] = bool(options.reuse_normalized)
    if options.reuse_samples is not None:
        override["runtime"]["reuse_samples"] = bool(options.reuse_samples)
    if options.reuse_exports is not None:
        override["runtime"]["reuse_exports"] = bool(options.reuse_exports)
    if options.force_rebuild is not None:
        override["runtime"]["force_rebuild"] = bool(options.force_rebuild)
    if options.mineru_workers is not None:
        override.setdefault("mineru", {})["max_concurrency"] = int(options.mineru_workers)
    if options.mineru_retry_count is not None:
        override.setdefault("mineru", {})["retry_count"] = int(options.mineru_retry_count)
    if options.mineru_min_page_coverage is not None:
        override.setdefault("mineru", {})["min_page_coverage"] = float(options.mineru_min_page_coverage)
    if options.min_page_text_chars is not None:
        override.setdefault("routing", {})["min_page_text_chars"] = int(options.min_page_text_chars)
    if options.min_block_text_chars is not None:
        override.setdefault("routing", {})["min_block_text_chars"] = int(options.min_block_text_chars)
    cfg = deep_merge(cfg, override)
    if bool(cfg['runtime'].get('cache_only')) and (
        options.input_dir is not None or options.input_books is not None
    ):
        raise ValueError('--cache-only cannot be combined with --input-dir or --book')
    if bool(cfg['runtime'].get('cache_only')) and bool(cfg['runtime'].get('force_rebuild')):
        raise ValueError('--cache-only cannot be combined with --force-rebuild')
    if bool(cfg['runtime'].get('cache_only')):
        cfg['runtime']['reuse_mineru'] = True
        cfg['runtime']['reuse_normalized'] = True
        cfg['runtime']['render_pages'] = False
        cfg['runtime']['crop_blocks'] = False
    if bool(cfg["runtime"].get("force_rebuild")):
        cfg["runtime"]["reuse_mineru"] = False
        cfg["runtime"]["reuse_normalized"] = False
        cfg["runtime"]["reuse_samples"] = False
        cfg["runtime"]["reuse_exports"] = False
    if options.task_types is not None:
        cfg = _select_task_types(cfg, options.task_types)
    return cfg


def _select_task_types(cfg: dict[str, Any], task_types: list[str]) -> dict[str, Any]:
    requested = [str(task_type).strip() for task_type in task_types if str(task_type).strip()]
    known_tasks = list(cfg.get("task_types", []))
    known_task_set = {str(task_type) for task_type in known_tasks}
    unknown = [task_type for task_type in requested if task_type not in known_task_set]
    if unknown:
        raise ValueError(f"Unknown task type(s): {', '.join(unknown)}")
    selected = list(dict.fromkeys(requested))
    if not selected:
        raise ValueError("At least one task type must be selected.")

    selected_set = set(selected)
    cfg["task_types"] = [task_type for task_type in known_tasks if task_type in selected_set]
    for key, task_list in list(cfg.get("export_formats", {}).items()):
        if isinstance(task_list, list):
            cfg["export_formats"][key] = [task_type for task_type in task_list if task_type in selected_set]
    enabled_tasks = cfg.setdefault("routing", {}).setdefault("enabled_tasks", {})
    for task_type in known_tasks:
        enabled_tasks[task_type] = task_type in selected_set
    for provider in cfg.get("vlm_pool", {}).get("providers", []):
        if not isinstance(provider, dict):
            continue
        provider_tasks = provider.get("task_types")
        if isinstance(provider_tasks, list):
            provider["task_types"] = [task_type for task_type in provider_tasks if task_type in selected_set]
    return cfg


def _resolve_config_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _resolve_config_runtime_paths(cfg: dict[str, Any], options: PipelineOptions) -> dict[str, Any]:
    base_dir = Path(options.config_path).resolve().parent if options.config_path else package_root().parent
    if options.input_dir is None:
        cfg["runtime"]["input_dir"] = _resolve_config_path(str(cfg["runtime"]["input_dir"]), base_dir)
    if options.output_root is None:
        cfg["runtime"]["output_root"] = _resolve_config_path(str(cfg["runtime"]["output_root"]), base_dir)
    if options.input_books is None:
        input_books = cfg["runtime"].get("input_books")
        if isinstance(input_books, list):
            for item in input_books:
                if not isinstance(item, dict):
                    continue
                raw_path = item.get("path") or item.get("source_pdf") or item.get("address") or item.get("url")
                if raw_path and not Path(str(raw_path)).is_absolute():
                    item["path"] = _resolve_config_path(str(raw_path), base_dir)
    return cfg


def _generation_state_path(output_dir: Path, cfg: dict[str, Any]) -> Path:
    return output_dir / cfg["paths"].get("sample_generation_state", "samples/generation_state.json")


def _job_resume_key_from_values(
    *,
    task_type: str,
    book_id: str,
    chapter_no: str,
    page_index: int,
    block_id: str,
) -> str:
    return f"{task_type}|{book_id}|{chapter_no}|{page_index}|{block_id}"


def _job_resume_key(job: Any, cfg: dict[str, Any]) -> str:
    page = getattr(job, "page", None)
    block = getattr(job, "block", None)
    page_index = int(getattr(page, "page_index", 0) or 0)
    block_id = getattr(block, "block_id", "") or cfg.get("default_block_id", "page")
    return _job_resume_key_from_values(
        task_type=str(getattr(job, "task_type", "")),
        book_id=str(getattr(getattr(job, "book", None), "book_id", "")),
        chapter_no=str(getattr(page, "chapter_no", "") or ""),
        page_index=page_index,
        block_id=str(block_id),
    )


def _sample_resume_key(sample: dict[str, Any], cfg: dict[str, Any]) -> str:
    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    task_type = str(sample.get("task_type") or metadata.get("task_type") or "")
    page_index = int(metadata.get("page_index") or 0)
    return _job_resume_key_from_values(
        task_type=task_type,
        book_id=str(metadata.get("book_id") or ""),
        chapter_no=str(metadata.get("chapter_no") or ""),
        page_index=page_index,
        block_id=str(metadata.get("block_id") or cfg.get("default_block_id", "page")),
    )


def _read_generation_state(output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = _generation_state_path(output_dir, cfg)
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _generation_state_matches(payload: dict[str, Any], cfg: dict[str, Any], mineru_content_hash: str) -> bool:
    return (
        bool(mineru_content_hash)
        and payload.get("mineru_content_hash") == mineru_content_hash
        and payload.get("pipeline_version") == cfg.get("pipeline_version", "")
    )


def _completed_generation_jobs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    completed = payload.get("completed_jobs")
    if not isinstance(completed, dict):
        return {}
    return {str(key): value for key, value in completed.items() if isinstance(value, dict)}


def _write_generation_state(
    output_dir: Path,
    cfg: dict[str, Any],
    mineru_content_hash: str,
    completed_jobs: dict[str, dict[str, Any]],
) -> None:
    write_json(
        _generation_state_path(output_dir, cfg),
        {
            "mineru_content_hash": mineru_content_hash,
            "pipeline_version": cfg.get("pipeline_version", ""),
            "completed_jobs": completed_jobs,
        },
    )


def _load_resume_raw_samples(
    output_dir: Path,
    cfg: dict[str, Any],
    job_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    counts_by_key: dict[str, int] = {}
    for task_type in cfg["task_types"]:
        path = output_dir / cfg["paths"]["sample_raw"].format(task_type=task_type)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    sample = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(sample, dict):
                    continue
                key = _sample_resume_key(sample, cfg)
                if key not in job_keys:
                    continue
                counts_by_key[key] = counts_by_key.get(key, 0) + 1
                samples.append(sample)
    return samples, counts_by_key


def _generate_samples_for_jobs(
    *,
    jobs: list[Any],
    output_dir: Path,
    cfg: dict[str, Any],
    vlm: VlmPool | None,
    state: RunState,
    mineru_content_hash: str = "",
) -> list[dict[str, Any]]:
    resume_enabled = bool(cfg["runtime"].get("reuse_samples")) and bool(mineru_content_hash)
    job_keys = [_job_resume_key(job, cfg) for job in jobs]
    state_payload = _read_generation_state(output_dir, cfg) if resume_enabled else {}
    state_matches = resume_enabled and _generation_state_matches(state_payload, cfg, mineru_content_hash)
    completed_jobs = _completed_generation_jobs(state_payload) if state_matches else {}
    raw_counts_by_key: dict[str, int] = {}

    if state_matches:
        raw_samples, raw_counts_by_key = _load_resume_raw_samples(output_dir, cfg, set(job_keys))
    else:
        for task_type in cfg["task_types"]:
            write_jsonl(output_dir / cfg["paths"]["sample_raw"].format(task_type=task_type), [])
        if mineru_content_hash:
            _write_generation_state(output_dir, cfg, mineru_content_hash, {})
        completed_jobs = {}
        raw_samples = []

    max_workers = max(1, int(cfg["runtime"]["max_workers"]))
    max_pending = max(max_workers, int(cfg["runtime"].get("vlm_max_pending", max_workers)))

    def sample_count_is_resumable(key: str) -> bool:
        raw_count = raw_counts_by_key.get(key, 0)
        checkpoint = completed_jobs.get(key, {})
        sample_count = checkpoint.get("sample_count")
        if isinstance(sample_count, int):
            return sample_count == 0 or raw_count >= sample_count
        return raw_count > 0

    def mark_job_completed(job: Any, job_key: str, samples: list[dict[str, Any]]) -> None:
        if not mineru_content_hash:
            return
        completed_jobs[job_key] = {
            "task_type": str(getattr(job, "task_type", "")),
            "book_id": str(getattr(getattr(job, "book", None), "book_id", "")),
            "sample_count": len(samples),
        }
        _write_generation_state(output_dir, cfg, mineru_content_hash, completed_jobs)

    def add_samples(job: Any, job_key: str, samples: list[dict[str, Any]]) -> None:
        for sample in samples:
            raw_samples.append(sample)
            append_jsonl(output_dir / cfg["paths"]["sample_raw"].format(task_type=job.task_type), sample)
        mark_job_completed(job, job_key, samples)

    if max_workers == 1 or len(jobs) <= 1:
        for index, job in enumerate(jobs):
            job_key = job_keys[index]
            if sample_count_is_resumable(job_key):
                continue
            try:
                samples = generate_for_job(job, output_dir, cfg, vlm)
                add_samples(job, job_key, samples)
            except Exception as exc:
                state.error({"stage": "generate", "book_id": job.book.book_id, "task_type": job.task_type, "error": str(exc)})
        return raw_samples

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Any, Any] = {}
        job_iter = iter(enumerate(jobs))
        exhausted = False
        while futures or not exhausted:
            while not exhausted and len(futures) < max_pending:
                try:
                    job_index, job = next(job_iter)
                except StopIteration:
                    exhausted = True
                    break
                if sample_count_is_resumable(job_keys[job_index]):
                    continue
                futures[executor.submit(generate_for_job, job, output_dir, cfg, vlm)] = (job_index, job)
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job_index, job = futures.pop(future)
                job_key = job_keys[job_index]
                try:
                    samples = future.result()
                except Exception as exc:
                    state.error({"stage": "generate", "book_id": job.book.book_id, "task_type": job.task_type, "error": str(exc)})
                    continue
                add_samples(job, job_key, samples)
    return raw_samples


def _expected_export_paths(output_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for task_type in cfg["task_types"]:
        paths.extend(_expected_export_paths_for_task(output_dir, cfg, task_type))
    return paths


def _expected_export_paths_for_task(output_dir: Path, cfg: dict[str, Any], task_type: str) -> list[Path]:
    paths: list[Path] = []
    format_paths = {
        "sharegpt": "export_sharegpt",
        "alpaca": "export_alpaca",
        "pt": "export_pt",
    }
    for format_name, path_key in format_paths.items():
        if path_key not in cfg["paths"]:
            continue
        if task_type in cfg.get("export_formats", {}).get(format_name, []):
            paths.append(output_dir / cfg["paths"][path_key].format(task_type=task_type))
    return paths


def _deduped_path_for_task(output_dir: Path, cfg: dict[str, Any], task_type: str) -> Path:
    return output_dir / cfg["paths"]["sample_deduped"].format(task_type=task_type)


def _load_deduped_samples(output_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for task_type in cfg["task_types"]:
        path = _deduped_path_for_task(output_dir, cfg, task_type)
        samples.extend(item for item in iter_jsonl(path) if isinstance(item, dict))
    return samples


def _deduped_paths_exist(output_dir: Path, cfg: dict[str, Any]) -> bool:
    paths = [_deduped_path_for_task(output_dir, cfg, task_type) for task_type in cfg["task_types"]]
    return all(path.exists() for path in paths) and any(path.stat().st_size > 0 for path in paths)


def _read_sample_cache_state(output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / cfg["paths"].get("sample_cache_state", "samples/cache_state.json")
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _cached_completed_task_types(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> set[str]:
    if not mineru_content_hash:
        return set()
    payload = _read_sample_cache_state(output_dir, cfg)
    if payload.get("mineru_content_hash") != mineru_content_hash:
        return set()
    completed = payload.get("completed_task_types")
    if not isinstance(completed, list):
        completed = payload.get("task_types")
    if not isinstance(completed, list):
        return set()
    return {str(task_type) for task_type in completed if str(task_type)}


def _task_outputs_exist(output_dir: Path, cfg: dict[str, Any], task_type: str) -> bool:
    export_paths = _expected_export_paths_for_task(output_dir, cfg, task_type)
    return (
        bool(export_paths)
        and _deduped_path_for_task(output_dir, cfg, task_type).exists()
        and all(path.exists() for path in export_paths)
    )


def _task_outputs_have_content(output_dir: Path, cfg: dict[str, Any], task_type: str) -> bool:
    if not _task_outputs_exist(output_dir, cfg, task_type):
        return False
    export_paths = _expected_export_paths_for_task(output_dir, cfg, task_type)
    paths = [_deduped_path_for_task(output_dir, cfg, task_type), *export_paths]
    return any(path.stat().st_size > 0 for path in paths)


def _completed_task_types(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> set[str]:
    cached = _cached_completed_task_types(output_dir, cfg, mineru_content_hash)
    selected = {str(task_type) for task_type in cfg["task_types"]}
    completed = {
        task_type
        for task_type in selected
        if task_type in cached and _task_outputs_exist(output_dir, cfg, task_type)
    }
    completed.update(task_type for task_type in selected if _task_outputs_have_content(output_dir, cfg, task_type))
    return completed


def _sample_cache_matches(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> bool:
    completed = _completed_task_types(output_dir, cfg, mineru_content_hash)
    return set(cfg["task_types"]).issubset(completed)


def _ordered_task_types(cfg: dict[str, Any], task_types: set[str]) -> list[str]:
    ordered = [task_type for task_type in cfg.get("task_types", []) if task_type in task_types]
    extras = sorted(task_type for task_type in task_types if task_type not in ordered)
    return ordered + extras


def _write_sample_cache_state(
    output_dir: Path,
    cfg: dict[str, Any],
    mineru_content_hash: str,
    completed_task_types: set[str] | None = None,
) -> None:
    previous = _cached_completed_task_types(output_dir, cfg, mineru_content_hash)
    current = set(cfg["task_types"]) if completed_task_types is None else set(completed_task_types)
    completed = current | previous
    ordered_completed = _ordered_task_types(cfg, completed)
    write_json(
        output_dir / cfg["paths"].get("sample_cache_state", "samples/cache_state.json"),
        {
            "mineru_content_hash": mineru_content_hash,
            "task_types": ordered_completed,
            "completed_task_types": ordered_completed,
            "pipeline_version": cfg.get("pipeline_version", ""),
        },
    )


def _exports_exist(output_dir: Path, cfg: dict[str, Any]) -> bool:
    expected = _expected_export_paths(output_dir, cfg)
    return bool(expected) and all(path.exists() for path in expected) and any(path.stat().st_size > 0 for path in expected)


def _export_reuse_ready(output_dir: Path, cfg: dict[str, Any]) -> bool:
    if not _exports_exist(output_dir, cfg):
        return False
    payload = _read_sample_cache_state(output_dir, cfg)
    if not payload:
        return True
    completed = payload.get("completed_task_types")
    if not isinstance(completed, list):
        completed = payload.get("task_types")
    if not isinstance(completed, list):
        return False
    completed_set = {str(task_type) for task_type in completed if str(task_type)}
    return all(
        task_type in completed_set or _task_outputs_have_content(output_dir, cfg, task_type)
        for task_type in cfg["task_types"]
    )


def _cfg_for_task_types(cfg: dict[str, Any], task_types: list[str]) -> dict[str, Any]:
    return _select_task_types(deepcopy(cfg), task_types)


def _build_vlm_pool(cfg: dict[str, Any]) -> VlmPool:
    cfg = deep_merge(
        cfg,
        {
            "vlm_pool": {
                "provider_defaults": {
                    "min_interval_seconds": float(cfg["runtime"].get("vlm_min_interval_seconds", 0.0)),
                    "max_image_bytes": int(cfg["generation"].get("max_image_bytes", 0)),
                    "max_image_side": int(cfg["generation"].get("max_image_side", 1600)),
                    "image_jpeg_quality": int(cfg["generation"].get("image_jpeg_quality", 85)),
                }
            }
        },
    )
    return VlmPool.from_config(cfg, cfg["prompts"])


def _selected_tasks_need_vlm(cfg: dict[str, Any]) -> bool:
    return any(str(task_type) != "domain_knowledge_corpus" for task_type in cfg.get("task_types", []))


def _process_book(book: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(book.output_dir)
    logger = configure_logger(f"{cfg['logger_name']}.{book.book_id}")
    state = RunState(logger)
    vlm = None if bool(cfg["runtime"]["skip_vlm"]) or not _selected_tasks_need_vlm(cfg) else _build_vlm_pool(cfg)

    logger.info("start book_id=%s pdf=%s", book.book_id, book.source_pdf)
    append_jsonl(output_dir / cfg["paths"]["manifest"], asdict(book))
    try:
        if (
            bool(cfg["runtime"].get("reuse_exports"))
            and _export_reuse_ready(output_dir, cfg)
            and mineru_cache_is_valid(book, cfg)
        ):
            logger.info("reuse exports book_id=%s", book.book_id)
            return {"book_id": book.book_id, "output_dir": str(output_dir), "sample_count": 0, "reused": "exports"}

        render_pages(book, cfg)
        state.checkpoint("render_pages", cfg["statuses"]["done"])
        mineru_result = parse_book_with_mineru(book, cfg)
        content = mineru_result.content
        parsed_path = mineru_result.parsed_path
        image_map = mineru_result.image_map
        state.checkpoint("mineru", mineru_result.status)
        pages = normalize_pages(book, content, parsed_path, image_map, cfg)
        crop_blocks(pages, output_dir, cfg)
        state.checkpoint("normalize_and_crop", len(pages))
        export_all_tasks = True
        if (
            bool(cfg["runtime"].get("reuse_samples"))
            and _deduped_paths_exist(output_dir, cfg)
            and _sample_cache_matches(output_dir, cfg, mineru_result.content_hash)
        ):
            deduped = _load_deduped_samples(output_dir, cfg)
            logger.info("reuse deduped samples book_id=%s samples=%s", book.book_id, len(deduped))
        else:
            completed_tasks = (
                _completed_task_types(output_dir, cfg, mineru_result.content_hash)
                if bool(cfg["runtime"].get("reuse_samples"))
                else set()
            )
            pending_tasks = [task_type for task_type in cfg["task_types"] if task_type not in completed_tasks]
            existing_deduped = []
            if completed_tasks:
                completed_cfg = _cfg_for_task_types(cfg, _ordered_task_types(cfg, completed_tasks))
                existing_deduped = _load_deduped_samples(output_dir, completed_cfg)
                logger.info(
                    "reuse completed task samples book_id=%s tasks=%s samples=%s",
                    book.book_id,
                    ",".join(completed_cfg["task_types"]),
                    len(existing_deduped),
                )
            if not pending_tasks:
                deduped = existing_deduped
                _write_sample_cache_state(output_dir, cfg, mineru_result.content_hash, completed_tasks)
                export_all_tasks = not bool(cfg["runtime"].get("reuse_exports"))
                logger.info("all selected tasks already completed book_id=%s", book.book_id)
            else:
                pending_task_set = set(pending_tasks)
                jobs = [job for job in build_sample_jobs(book, pages, cfg) if job.task_type in pending_task_set]
                job_counts = {task_type: 0 for task_type in pending_tasks}
                for job in jobs:
                    job_counts[job.task_type] = job_counts.get(job.task_type, 0) + 1
                logger.info(
                    "generate pending tasks book_id=%s tasks=%s jobs=%s",
                    book.book_id,
                    ",".join(pending_tasks),
                    len(jobs),
                )
                pending_cfg = _cfg_for_task_types(cfg, pending_tasks)
                raw_samples = _generate_samples_for_jobs(
                    jobs=jobs,
                    output_dir=output_dir,
                    cfg=pending_cfg,
                    vlm=vlm,
                    state=state,
                    mineru_content_hash=mineru_result.content_hash,
                )
                validated: list[dict[str, Any]] = []
                for sample in raw_samples:
                    try:
                        valid = validate_sample(sample, output_dir, pending_cfg, vlm=vlm)
                    except Exception as exc:
                        state.error(
                            {
                                "stage": "validate",
                                "book_id": book.book_id,
                                "task_type": sample.get("task_type"),
                                "sample_id": sample.get("id"),
                                "error": str(exc),
                            }
                        )
                        continue
                    if valid is not None:
                        validated.append(valid)
                for task_type in pending_cfg["task_types"]:
                    write_jsonl(
                        output_dir / pending_cfg["paths"]["sample_validated"].format(task_type=task_type),
                        (sample for sample in validated if sample.get("task_type") == task_type),
                    )
                new_deduped = deduplicate(validated, pending_cfg)
                for task_type in pending_cfg["task_types"]:
                    write_jsonl(
                        output_dir / pending_cfg["paths"]["sample_deduped"].format(task_type=task_type),
                        (sample for sample in new_deduped if sample.get("task_type") == task_type),
                    )
                sample_counts = {task_type: 0 for task_type in pending_tasks}
                for sample in new_deduped:
                    task_type = str(sample.get("task_type") or "")
                    sample_counts[task_type] = sample_counts.get(task_type, 0) + 1
                completed_pending = {
                    task_type
                    for task_type in pending_tasks
                    if job_counts.get(task_type, 0) == 0 or sample_counts.get(task_type, 0) > 0
                }
                if completed_pending:
                    completed_pending_cfg = _cfg_for_task_types(cfg, _ordered_task_types(cfg, completed_pending))
                    export_task_files(
                        output_dir,
                        [sample for sample in new_deduped if sample.get("task_type") in completed_pending],
                        completed_pending_cfg,
                    )
                incomplete_pending = [task_type for task_type in pending_tasks if task_type not in completed_pending]
                if incomplete_pending:
                    logger.info("pending tasks produced no valid samples book_id=%s tasks=%s", book.book_id, ",".join(incomplete_pending))
                completed_tasks.update(completed_pending)
                _write_sample_cache_state(output_dir, cfg, mineru_result.content_hash, completed_tasks)
                deduped = existing_deduped + new_deduped
                export_all_tasks = not bool(cfg["runtime"].get("reuse_exports"))
        if export_all_tasks:
            export_task_files(output_dir, deduped, cfg)
        state.increment("books_completed")
        state.increment("samples_exported", len(deduped))
        state.checkpoint("export", cfg["statuses"]["done"])
        result = {"book_id": book.book_id, "output_dir": str(output_dir), "sample_count": len(deduped)}
        logger.info("completed book_id=%s samples=%s", book.book_id, len(deduped))
        return result
    except Exception as exc:
        state.error({"stage": "book", "book_id": book.book_id, "error": str(exc)})
        logger.exception("failed book_id=%s", book.book_id)
        return {"book_id": book.book_id, "output_dir": str(output_dir), "sample_count": 0, "error": str(exc)}


def run_pipeline(options: PipelineOptions | None = None) -> list[dict[str, Any]]:
    options = options or PipelineOptions()
    _validate_options(options)
    cfg = _apply_options(load_config(options.config_path), options)
    cfg = _resolve_config_runtime_paths(cfg, options)
    input_dir = Path(cfg["runtime"]["input_dir"])
    output_root = Path(cfg["runtime"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    input_books = cfg["runtime"].get("input_books")
    if bool(cfg['runtime'].get('cache_only')):
        books = scan_cached_books(output_root, cfg)
    elif isinstance(input_books, list) and input_books:
        books = scan_input_books(input_books, output_root, cfg)
    else:
        books = scan_books(input_dir, output_root, cfg)
    if options.limit_books:
        books = books[: options.limit_books]

    results: list[dict[str, Any]] = []
    book_workers = max(1, int(cfg["runtime"]["book_workers"]))
    if book_workers == 1 or len(books) <= 1:
        return [_process_book(book, cfg) for book in books]

    with ProcessPoolExecutor(max_workers=book_workers) as executor:
        futures = {executor.submit(_process_book, book, cfg): index for index, book in enumerate(books)}
        ordered_results: list[dict[str, Any] | None] = [None] * len(books)
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()
    results = [result for result in ordered_results if result is not None]
    return results
