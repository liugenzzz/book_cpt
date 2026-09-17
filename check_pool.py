#!/usr/bin/env python3
"""模型池探活：逐个打 VLM 和 MinerU 实例，报哪个通哪个不通。

用法:
    python check_pool.py                # 探全部（只探 enabled 的 VLM）
    python check_pool.py --all          # 连 enabled=False 的也探
    python check_pool.py --chat         # VLM 额外发一次真实 chat 请求（更慢更准）
    python check_pool.py --timeout 5
    python check_pool.py --config /path/to/config.py

退出码：全通 0，有任何一个不通 1。
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from book_cpt.core.config_loader import load_config  # noqa: E402


def _chat_base(url: str) -> str:
    """把 .../v1/chat/completions 还原成 .../v1，用来打 /models。"""
    return url.split("/chat/completions")[0].rstrip("/")


def _probe_vlm(provider: dict[str, Any], timeout: float, do_chat: bool) -> dict[str, Any]:
    import requests

    name = str(provider.get("name") or "?")
    url = str(provider.get("url") or "")
    want_model = str(provider.get("model") or "")
    row: dict[str, Any] = {
        "kind": "VLM",
        "name": name,
        "target": url.split("/v1/")[0],
        "disabled": provider.get("enabled") is False,
        "ok": False,
        "detail": "",
    }
    headers = {}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    try:
        resp = requests.get(f"{_chat_base(url)}/models", headers=headers, timeout=timeout)
    except Exception as exc:
        row["detail"] = f"{type(exc).__name__}: {exc}"[:110]
        return row
    if resp.status_code >= 400:
        hint = {401: " (api_key 不对)", 404: " (路径不对)"}.get(resp.status_code, "")
        row["detail"] = f"HTTP {resp.status_code}{hint}: {resp.text[:70]}"
        return row
    try:
        served = [str(item.get("id")) for item in (resp.json().get("data") or [])]
    except Exception:
        served = []
    if want_model and want_model not in served:
        # 名字对不上，流水线发请求时服务端会 404，必须当失败报出来
        row["detail"] = f"model 名不匹配：配的 {want_model}，服务端有 {served or '(空)'}"
        return row
    row["ok"] = True
    row["detail"] = f"models={want_model or served}"

    if do_chat:
        payload = {
            "model": want_model,
            "messages": [{"role": "user", "content": "说“好”"}],
            "max_tokens": 4,
            "temperature": 0,
        }
        try:
            chat = requests.post(url, json=payload, headers=headers, timeout=max(timeout, 30))
        except Exception as exc:
            row["ok"] = False
            row["detail"] = f"chat 打不通：{type(exc).__name__}: {exc}"[:110]
            return row
        if chat.status_code >= 400:
            row["ok"] = False
            row["detail"] = f"chat HTTP {chat.status_code}: {chat.text[:70]}"
        else:
            row["detail"] += " | chat OK"
    return row


def _probe_mineru(provider: Any, timeout: float) -> dict[str, Any]:
    import requests

    url = str(provider.cfg.get("url") or "").rstrip("/")
    row: dict[str, Any] = {
        "kind": "MinerU",
        "name": provider.name,
        "target": url,
        "disabled": False,
        "ok": False,
        "detail": "",
    }
    try:
        resp = requests.get(f"{url}/docs", timeout=timeout)
    except Exception as exc:
        row["detail"] = f"{type(exc).__name__}: {exc}"[:110]
        return row
    if resp.status_code >= 400:
        row["detail"] = f"HTTP {resp.status_code}"
        return row
    row["ok"] = True
    row["detail"] = f"docs 可达，并发 {provider.max_concurrency}"

    server_url = str(provider.cfg.get("server_url") or "").rstrip("/")
    if server_url:
        # vlm-http-client 后端真正干活的是 server_url，不通的话解析会 409
        try:
            vl = requests.get(f"{server_url}/v1/models", timeout=timeout)
            row["detail"] += f" | server_url {'OK' if vl.status_code < 400 else f'HTTP {vl.status_code}'}"
        except Exception as exc:
            row["ok"] = False
            row["detail"] += f" | server_url 打不通：{type(exc).__name__}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--all", action="store_true", help="连 enabled=False 的实例也探")
    parser.add_argument("--chat", action="store_true", help="VLM 额外发一次真实 chat 请求")
    args = parser.parse_args()

    try:
        import requests  # noqa: F401
    except ImportError:
        print("缺 requests：pip install -r book_cpt/requirements.txt")
        sys.exit(1)

    cfg = load_config(args.config)
    from book_cpt.services.mineru import mineru_providers

    providers = cfg.get("vlm_pool", {}).get("providers") or []
    vlm = [p for p in providers if isinstance(p, dict) and (args.all or p.get("enabled") is not False)]
    mineru = mineru_providers(cfg)

    print(f"探测 {len(vlm)} 个 VLM 实例 + {len(mineru)} 个 MinerU 实例，超时 {args.timeout}s"
          f"{'，含 chat 实测' if args.chat else ''}\n")

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(_probe_vlm, p, args.timeout, args.chat) for p in vlm]
        futures += [pool.submit(_probe_mineru, p, args.timeout) for p in mineru]
        rows = [f.result() for f in futures]

    width = max((len(r["name"]) for r in rows), default=10)
    down = []
    for row in sorted(rows, key=lambda r: (r["kind"], r["name"])):
        mark = "OK  " if row["ok"] else "不通"
        tag = " [停用]" if row["disabled"] else ""
        print(f"  [{mark}] {row['kind']:<6} {row['name']:<{width}}{tag}  {row['target']}")
        if not row["ok"] or row["disabled"]:
            print(f"           {row['detail']}")
        if not row["ok"] and not row["disabled"]:
            down.append(row)

    ok_count = sum(1 for r in rows if r["ok"] and not r["disabled"])
    print(f"\n可用 {ok_count}/{len([r for r in rows if not r['disabled']])}")
    if down:
        print("不通的实例（流水线会自动冷却绕开，但容量相应缩水）:")
        for row in down:
            print(f"  - {row['name']}: {row['detail']}")
        sys.exit(1)
    print("全部可达。")


if __name__ == "__main__":
    main()
