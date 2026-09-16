#!/usr/bin/env python3
"""依赖自检：跑主流程前先确认该装的都装了。

用法: python check_deps.py
"""
from __future__ import annotations

import importlib
import sys

CHECKS = [
    ("requests", "requests", "必需", "调用 MinerU / VLM 接口，缺了直接报错"),
    ("pypdf", "pypdf", "必需", "PDF 页数统计、损坏检测、水印清理"),
    ("PyMuPDF", "fitz", "重要", "页面渲染成图；缺了 render_pages 会静默跳过"),
    ("Pillow", "PIL", "重要", "图像归一化 / 裁剪 / 质量检测；缺了相关步骤降级"),
]

MIN_PYTHON = (3, 9)

missing: list[str] = []
failed = False
print(f"python {sys.version.split()[0]}  {sys.executable}")
if sys.version_info < MIN_PYTHON:
    failed = True
    print(
        f"  [过低] 需要 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 及以上 —— "
        "Ctrl-C 取消排队任务用的 shutdown(cancel_futures=True) 是 3.9 才有的"
    )
else:
    print(f"  [OK]   >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
print()
for pkg, module, level, why in CHECKS:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "") or getattr(mod, "version", "")
        if module == "fitz":
            version = getattr(mod, "__doc__", "") or ""
            version = version.strip().splitlines()[0] if version else ""
        print(f"  [OK]   {pkg:<10} {version}")
    except Exception as exc:
        missing.append(pkg)
        print(f"  [缺失] {pkg:<10} ({level}) — {why}")
        print(f"         {type(exc).__name__}: {exc}")

print()
if missing:
    print("安装命令:")
    print(f"  pip install {' '.join(missing)} -i https://mirrors.aliyun.com/pypi/simple/")
if missing or failed:
    sys.exit(1)
print("依赖齐全。")
