"""Book raw knowledge-expression data generation package."""

from __future__ import annotations

import importlib
import sys


_MODULE_ALIASES = {
    "cli": "app.cli",
    "pipeline": "app.pipeline",
    "config_loader": "core.config_loader",
    "io_utils": "core.io_utils",
    "logging_utils": "core.logging_utils",
    "models": "core.models",
    "clients": "services.clients",
    "mineru": "services.mineru",
    "crop": "processing.crop",
    "image_quality": "processing.image_quality",
    "ingest": "processing.ingest",
    "normalize": "processing.normalize",
    "render": "processing.render",
    "dedup": "tasks.dedup",
    "exporters": "tasks.exporters",
    "generation": "tasks.generation",
    "routing": "tasks.routing",
    "validation": "tasks.validation",
}

for old_name, new_name in _MODULE_ALIASES.items():
    sys.modules.setdefault(f"{__name__}.{old_name}", importlib.import_module(f".{new_name}", __name__))
