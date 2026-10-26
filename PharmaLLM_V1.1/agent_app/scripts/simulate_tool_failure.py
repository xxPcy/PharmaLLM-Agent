#!/usr/bin/env python3
"""Run only against MODEL_MODE=fake, ENABLE_FAULTS=true in a development environment."""

import argparse
import asyncio
import json
import httpx
from client_common import register, run_case


async def main(args):
    async with httpx.AsyncClient(base_url=args.base_url, timeout=300, trust_env=False) as client:
        ready = await client.get("/health/ready")
        ready.raise_for_status()
        if ready.json().get("model_mode") != "fake":
            raise SystemExit("故障脚本仅允许 FakeModel 模式")
        headers, _ = await register(client)
        cases = [
            ("[[fault:model_503]]", "MODEL_UNAVAILABLE"),
            ("[[fault:model_timeout]]", "MODEL_TIMEOUT"),
            ("[[fault:bad_route]]", "ROUTE_INVALID"),
            ("检索 [[fault:tool_timeout]]", "TOOL_TIMEOUT"),
            ("检索 [[fault:tool_schema]]", "TOOL_BAD_RESPONSE"),
        ]
        for message, code in cases:
            result = await run_case(client, headers, message)
            print(json.dumps(result, ensure_ascii=False))
            assert result["error_code"] == code, "请确认 ENABLE_FAULTS=true，并重启 API"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    asyncio.run(main(parser.parse_args()))
