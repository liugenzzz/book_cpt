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
