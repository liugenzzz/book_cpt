from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from book_cpt.app import pipeline


def _cfg(**runtime):
    base = {"max_workers": 1, "vlm_max_pending": 1}
    base.update(runtime)
    return {
        "task_types": ["paragraph_summary"],
        "paths": {"sample_raw": "samples/raw/{task_type}.jsonl"},
        "runtime": base,
    }


def _jobs(count):
    book = SimpleNamespace(book_id="b1")
    return [
        SimpleNamespace(
            task_type="paragraph_summary",
            book=book,
            page=SimpleNamespace(page_index=index, chapter_no="1"),
            block=None,
        )
        for index in range(count)
    ]


def _run(cfg, jobs, generate):
    state = SimpleNamespace(error=lambda payload: None)
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(pipeline, "generate_for_job", side_effect=generate):
            return pipeline._generate_samples_for_jobs(
                jobs=jobs, output_dir=Path(tmp), cfg=cfg, vlm=None, state=state
            )


class AllJobsFailedTests(unittest.TestCase):
    """整批全挂通常是 provider 配错，静默返回 0 条会被误读成"这本书没内容"。"""

    def test_every_job_failing_raises_with_a_diagnosable_message(self) -> None:
        def always_fail(job, *_args):
            raise RuntimeError("VLM HTTP 401 from http://x: bad key")

        with self.assertRaises(pipeline.GenerationBatchError) as ctx:
            _run(_cfg(), _jobs(3), always_fail)
        message = str(ctx.exception)
        self.assertIn("book_id=b1", message)
        self.assertIn("jobs=3", message)
        self.assertIn("paragraph_summary:3", message)
        self.assertIn("401", message)

    def test_partial_failure_still_returns_the_good_samples(self) -> None:
        def fail_first(job, *_args):
            if job.page.page_index == 0:
                raise RuntimeError("boom")
            return [{"id": f"s{job.page.page_index}", "task_type": "paragraph_summary"}]

        samples = _run(_cfg(), _jobs(3), fail_first)
        self.assertEqual([sample["id"] for sample in samples], ["s1", "s2"])

    def test_empty_job_list_is_not_an_error(self) -> None:
        self.assertEqual(_run(_cfg(), [], lambda *_: []), [])


class RawSampleOrderTests(unittest.TestCase):
    """并发完成顺序是乱的，但返回的样本必须仍按 job 顺序排列，否则每次跑出来的
    deduped/exports 行序都不一样，diff 不了。"""

    def test_samples_follow_job_order_not_completion_order(self) -> None:
        def generate(job, *_args):
            index = job.page.page_index
            return [
                {"id": f"p{index}-{sample_no}", "task_type": "paragraph_summary"}
                for sample_no in range(2)
            ]

        samples = _run(_cfg(max_workers=4, vlm_max_pending=4), _jobs(5), generate)
        self.assertEqual(
            [sample["id"] for sample in samples],
            [f"p{index}-{sample_no}" for index in range(5) for sample_no in range(2)],
        )


if __name__ == "__main__":
    unittest.main()


class HeartbeatTests(unittest.TestCase):
    """一个任务都没完成时也得出声。

    log_progress 只在任务完成时调用，wait() 原来还没有超时 —— 整批请求都卡在
    模型侧时日志彻底安静，跟进程死了分不出来。单请求 timeout 2400s，真能静 40 分钟。
    """

    def _run(self, *, hold_seconds: float, heartbeat: float):
        import logging
        import threading

        logger = logging.getLogger("book_cpt_heartbeat_test")
        cfg = {
            "task_types": ["paragraph_summary"],
            "paths": {"sample_raw": "samples/raw/{task_type}.jsonl"},
            "runtime": {
                "max_workers": 2,
                "vlm_max_pending": 2,
                "generation_heartbeat_seconds": heartbeat,
            },
            "vlm_pool": {"providers": [{"name": "a", "timeout": 2400}]},
        }
        release = threading.Event()

        def slow_generate(job, *_args):
            release.wait(timeout=hold_seconds)
            return [{"id": f"s{job.page.page_index}", "task_type": "paragraph_summary"}]

        state = SimpleNamespace(error=lambda payload: None, logger=logger)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pipeline, "generate_for_job", side_effect=slow_generate):
                with self.assertLogs(logger, level="INFO") as captured:
                    threading.Timer(hold_seconds, release.set).start()
                    pipeline._generate_samples_for_jobs(
                        jobs=_jobs(2), output_dir=Path(tmp), cfg=cfg, vlm=None, state=state
                    )
        return captured.output

    def test_heartbeat_fires_while_nothing_completes(self) -> None:
        lines = [line for line in self._run(hold_seconds=0.6, heartbeat=0.15) if "等待中" in line]
        self.assertTrue(lines, "没有任务完成时必须打心跳")
        self.assertIn("在飞=2", lines[0])
        self.assertIn("timeout=2400s", lines[0])

    def test_no_heartbeat_when_jobs_finish_promptly(self) -> None:
        lines = [line for line in self._run(hold_seconds=0.0, heartbeat=30.0) if "等待中" in line]
        self.assertEqual(lines, [])

    def test_progress_still_logged_after_the_wait(self) -> None:
        lines = [line for line in self._run(hold_seconds=0.6, heartbeat=0.15) if "generate progress" in line]
        self.assertTrue(lines)
        self.assertIn("jobs=2/2", lines[-1])
