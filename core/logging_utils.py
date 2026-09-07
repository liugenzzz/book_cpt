from __future__ import annotations

import logging
from typing import Any


def configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


class RunState:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.metrics: dict[str, Any] = {}
        self.checkpoints: dict[str, Any] = {}

    def error(self, payload: dict[str, Any]) -> None:
        self.logger.error("pipeline_error %s", payload)

    def increment(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = int(self.metrics.get(key, 0)) + amount
        self.logger.info("metric %s=%s", key, self.metrics[key])

    def checkpoint(self, key: str, payload: Any) -> None:
        self.checkpoints[key] = payload
        self.logger.info("checkpoint %s=%s", key, payload)
