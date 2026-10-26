#!/usr/bin/env python3
"""P7 only: run real measurements; never create a pre-filled performance report."""

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import subprocess
import time

import httpx
from client_common import register, run_case


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def gpu_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


async def main(args):
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise SystemExit("评测集为空")
    levels = [int(x) for x in args.concurrency.split(",")]
    if not levels or any(n < 1 or n > 64 for n in levels):
        raise SystemExit("并发量范围为 1–64")
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=420, limits=httpx.Limits(max_connections=256), trust_env=False
    ) as client:
        ready = await client.get("/health/ready")
        ready.raise_for_status()
        tokens = []
        if args.tokens_file:
            # Tokens are operator-owned credentials. Never include them in a report or log.
            tokens = [
                {"Authorization": "Bearer " + value} for value in json.loads(args.tokens_file.read_text())
            ]
        elif args.allow_register:
            for _ in range(max(levels)):
                headers, _ = await register(client)
                tokens.append(headers)
        else:
            raise SystemExit("请提供 --tokens-file（不同用户的 Token JSON 数组），或显式 --allow-register")
        if len(tokens) < max(levels):
            raise SystemExit("每个并发 Worker 需要一个不同用户，避免把单用户配额误当吞吐上限")
        report = {
            "schema_version": "evaluation-1.1",
            "measured_at": datetime.now(UTC).isoformat(),
            "system": platform.platform(),
            "readiness": ready.json(),
            "cases_file": str(args.cases),
            "gpu_before": gpu_snapshot(),
            "gpu_after": [],
            "levels": [],
            "notes": [
                "No clinical/RAG quality judgments are inferred from latency or tool names.",
                "GPU snapshots are not peak memory measurements.",
            ],
        }
        for level in levels:
            queue = asyncio.Queue()
            for index in range(args.requests):
                queue.put_nowait((index, cases[index % len(cases)]))
            results = []
            started = time.monotonic()

            async def worker(index):
                while not queue.empty():
                    try:
                        sample_id, case = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        result = await run_case(client, tokens[index], case["message"])
                        result["sample_id"] = sample_id
                        result["case_id"] = case["id"]
                        result["expected_tool"] = case.get("expected_tool")
                        result["tool_match"] = (
                            case.get("expected_tool") in result["tools"]
                            if case.get("expected_tool")
                            else not result["tools"]
                        )
                        results.append(result)
                    except Exception as exc:
                        # Do not dump response bodies, headers, or tokens on HTTP failure.
                        results.append(
                            {
                                "sample_id": sample_id,
                                "case_id": case["id"],
                                "status": "failed",
                                "error_code": type(exc).__name__,
                            }
                        )

            await asyncio.gather(*(worker(i) for i in range(level)))
            elapsed = time.monotonic() - started
            success = [r for r in results if r["status"] == "completed"]
            latency = [r["latency_seconds"] for r in success]
            ttft = [r["ttft_seconds"] for r in success if r["ttft_seconds"] is not None]
            usage = [call["usage"] for r in success for call in r.get("usage", {}).get("calls", [])]
            metrics = {
                "concurrency": level,
                "requests": args.requests,
                "duration_seconds": elapsed,
                "p50_seconds": percentile(latency, 0.5),
                "p95_seconds": percentile(latency, 0.95),
                "ttft_p95_seconds": percentile(ttft, 0.95),
                "requests_per_second": len(results) / elapsed,
                "error_rate": 1 - len(success) / len(results),
                "invalid_route_rate": sum(r.get("error_code") == "ROUTE_INVALID" for r in results)
                / len(results),
                "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage) if usage else None,
                "tool_match_rate": sum(bool(r.get("tool_match")) for r in results) / len(results),
                "results": sorted(results, key=lambda r: r["sample_id"]),
            }
            report["levels"].append(metrics)
            print(json.dumps({k: v for k, v in metrics.items() if k != "results"}, ensure_ascii=False))
        report["gpu_after"] = gpu_snapshot()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        args.output.chmod(0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--cases", type=Path, default=Path("tests/evaluation/cases.jsonl"))
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--allow-register", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    if args.requests < 1:
        raise SystemExit("requests 必须大于 0")
    asyncio.run(main(args))
