#!/usr/bin/env python3
"""Read-only authenticated health check; never loads weights or submits an image."""

import asyncio
from pharma_agent.segmentation import SegmentationGateway
from pharma_agent.settings import Settings


async def main():
    result = await SegmentationGateway(Settings()).probe_detail()
    print(f"分割服务：{result['status']}；{result['message']}")
    if result["code"]:
        print("错误码：" + result["code"])
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
