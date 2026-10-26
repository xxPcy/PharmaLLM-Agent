import json
import secrets
import time
from uuid import uuid4


async def register(client):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": f"fixture-{uuid4().hex}@example.test",
            "password": secrets.token_hex(6),
            "tenant_name": "Synthetic engineering fixture",
        },
    )
    response.raise_for_status()
    data = response.json()
    return {"Authorization": "Bearer " + data["access_token"]}, data


async def events(client, path, headers):
    async with client.stream("GET", path, headers=headers) as response:
        response.raise_for_status()
        kind, event_id, lines = "", 0, []
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("id:"):
                event_id = int(line[3:].strip())
            elif line.startswith("data:"):
                lines.append(line[5:].strip())
            elif not line and lines:
                yield event_id, kind, json.loads("\n".join(lines))
                kind, event_id, lines = "", 0, []


async def run_case(client, headers, message):
    conversation = await client.post(
        "/api/v1/conversations", headers=headers, json={"title": "Synthetic evaluation"}
    )
    conversation.raise_for_status()
    cid = conversation.json()["id"]
    start = time.monotonic()
    response = await client.post(
        f"/api/v1/conversations/{cid}/runs",
        headers={**headers, "Idempotency-Key": uuid4().hex},
        json={"message": message, "stream": True},
    )
    response.raise_for_status()
    run = response.json()
    first, tools, terminal = None, [], None
    async for _, kind, payload in events(client, run["events_url"], headers):
        if kind == "delta" and first is None:
            first = time.monotonic() - start
        if kind == "tool_start":
            tools.append(payload["tool_name"])
        if kind in {"completed", "error"}:
            terminal = kind
    state = await client.get(f"/api/v1/runs/{run['run_id']}", headers=headers)
    state.raise_for_status()
    state = state.json()
    return {
        "run_id": run["run_id"],
        "trace_id": run["trace_id"],
        "status": state["status"],
        "error_code": state["error_code"],
        "tools": tools,
        "terminal_event": terminal,
        "latency_seconds": time.monotonic() - start,
        "ttft_seconds": first,
        "usage": state["usage"],
        "answer_characters": len(state.get("answer") or ""),
        "citations_count": len(state["citations"]),
    }
