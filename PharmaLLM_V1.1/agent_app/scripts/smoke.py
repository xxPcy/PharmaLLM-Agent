#!/usr/bin/env python3
import argparse
import asyncio
import json
import time

import httpx
from client_common import register, run_case


async def main(args):
    async with httpx.AsyncClient(base_url=args.base_url, timeout=300, trust_env=False) as client:
        response = await client.get("/health/ready")
        response.raise_for_status()
        print("Readiness:", response.json())
        a, _ = await register(client)
        b, _ = await register(client)
        result = await run_case(client, a, "你好，请介绍你的功能。")
        assert result["status"] == "completed", result
        denied = await client.get(f"/api/v1/runs/{result['run_id']}/events", headers=b)
        assert denied.status_code == 404
        memory = await client.post(
            "/api/v1/memories",
            headers=a,
            json={"kind": "language_preference", "value": "zh", "consent": True},
        )
        memory.raise_for_status()
        deleted = await client.delete(f"/api/v1/memories/{memory.json()['id']}", headers=a)
        deleted.raise_for_status()
        draft = await client.post("/api/v1/jobs/drafts", headers=a, json={"title": "Synthetic smoke job"})
        draft.raise_for_status()
        job = draft.json()
        assert job["status"] == "draft"
        print("Run:", json.dumps(result, ensure_ascii=False))
        print("Isolation, preference deletion, draft creation: passed")
        if args.queue:
            queued = await client.post(
                f"/api/v1/jobs/{job['id']}/confirm", headers=a, json={"confirmed": True}
            )
            queued.raise_for_status()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                status = await client.get(f"/api/v1/jobs/{job['id']}", headers=a)
                status.raise_for_status()
                job = status.json()
                if job["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.5)
            assert job["status"] == "succeeded", job
            asset = await client.get(f"/api/v1/assets/{job['result']['asset_id']}/download", headers=a)
            asset.raise_for_status()
            print("Queue, worker, authorized artifact download: passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--queue", action="store_true")
    asyncio.run(main(parser.parse_args()))
