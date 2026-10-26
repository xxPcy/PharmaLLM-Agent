import asyncio
import os
import time
from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from pharma_agent.db import AgentRun, Conversation, now
from pharma_agent.jobs import execute_mock_job
from pharma_agent.main import create_app

pytestmark = pytest.mark.integration


def register(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": f"{uuid4().hex}@example.test", "password": "TestPass123"},
    )
    assert response.status_code == 201, response.text
    data = response.json()
    return {"Authorization": "Bearer " + data["access_token"]}, data


def conversation(client, headers):
    response = client.post("/api/v1/conversations", json={"title": "Fixture"}, headers=headers)
    assert response.status_code == 201
    return response.json()["id"]


def run(client, headers, conversation_id, message="你好", key=None):
    response = client.post(
        f"/api/v1/conversations/{conversation_id}/runs",
        json={"message": message},
        headers={**headers, "Idempotency-Key": key or uuid4().hex},
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def wait(client, headers, run_id):
    for _ in range(200):
        state = client.get(f"/api/v1/runs/{run_id}", headers=headers).json()
        if state["status"] in {"completed", "failed", "cancelled"}:
            return state
        time.sleep(0.05)
    pytest.fail("Run did not finish")


def test_other_tenant_cannot_read_history_events_jobs_memories_assets(client):
    a, _ = register(client)
    b, _ = register(client)
    cid = conversation(client, a)
    rid = run(client, a, cid)
    assert wait(client, a, rid)["status"] == "completed"
    job = client.post("/api/v1/jobs/drafts", json={"title": "synthetic"}, headers=a).json()
    memory = client.post(
        "/api/v1/memories", json={"kind": "language_preference", "value": "zh", "consent": True}, headers=a
    ).json()
    asset = client.post(
        "/api/v1/assets", files={"file": ("fixture.txt", b"synthetic", "text/plain")}, headers=a
    ).json()
    for path in [
        f"/conversations/{cid}/messages",
        f"/runs/{rid}",
        f"/runs/{rid}/events",
        f"/jobs/{job['id']}",
        f"/assets/{asset['asset_id']}/download",
    ]:
        assert client.get("/api/v1" + path, headers=b).status_code == 404
    assert client.delete(f"/api/v1/memories/{memory['id']}", headers=b).status_code == 404
    assert client.get("/api/v1/memories", headers=b).json() == []


def test_same_tenant_other_user_is_also_isolated(client):
    a, data = register(client)
    svc = client.app.state.services

    async def add_member():
        from pharma_agent.db import User, new_id
        from pharma_agent.security import hash_password, issue_token, Actor

        async with svc.sessions() as session, session.begin():
            user = User(
                id=new_id("user"),
                tenant_id=data["tenant_id"],
                email=f"{uuid4().hex}@example.test",
                password_hash=await hash_password("SyntheticPassword_123"),
                role="member",
            )
            session.add(user)
        return issue_token(svc.settings, Actor(user.id, user.tenant_id, user.role))

    b = {"Authorization": "Bearer " + client.portal.call(add_member)}
    cid = conversation(client, a)
    assert client.get(f"/api/v1/conversations/{cid}/messages", headers=b).status_code == 404


def test_idempotency_reuses_run_and_rejects_changed_payload(client):
    headers, _ = register(client)
    cid = conversation(client, headers)
    rid = run(client, headers, cid, key="stable-key")
    assert run(client, headers, cid, key="stable-key") == rid
    response = client.post(
        f"/api/v1/conversations/{cid}/runs",
        json={"message": "different"},
        headers={**headers, "Idempotency-Key": "stable-key"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    wait(client, headers, rid)


def test_busy_lock_is_database_backed(client):
    headers, _ = register(client)
    cid = conversation(client, headers)
    svc = client.app.state.services
    # Freeze this instance's claim loop temporarily to keep the run queued deterministically.
    svc.runs.stopping = True
    time.sleep(0.6)
    rid = run(client, headers, cid)
    response = client.post(f"/api/v1/conversations/{cid}/runs", json={"message": "second"}, headers=headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONVERSATION_BUSY"
    assert client.post(f"/api/v1/runs/{rid}/cancel", headers=headers).json()["status"] == "cancelled"


def test_sse_replay_has_stable_event_ids(client):
    headers, _ = register(client)
    rid = run(client, headers, conversation(client, headers))
    wait(client, headers, rid)
    response = client.get(f"/api/v1/runs/{rid}/events", headers=headers)
    ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
    assert ids == sorted(set(ids)) and len(ids) > 2
    assert "event: completed" in response.text
    tail = client.get(f"/api/v1/runs/{rid}/events", headers={**headers, "Last-Event-ID": str(ids[-2])})
    assert f"id: {ids[-1]}\n" in tail.text and f"id: {ids[-2]}\n" not in tail.text


def test_invalid_route_answers_without_executing_a_tool(client):
    headers, _ = register(client)
    rid = run(client, headers, conversation(client, headers), "[[fault:bad_route]]")
    state = wait(client, headers, rid)
    assert state["status"] == "completed" and state["error_code"] == "ROUTE_FALLBACK"
    events = client.get(f"/api/v1/runs/{rid}/events", headers=headers).text
    assert "event: tool_start" not in events


@pytest.mark.parametrize(
    "message,code",
    [
        ("[[fault:model_503]]", "MODEL_UNAVAILABLE"),
        ("检索 [[fault:tool_timeout]]", "TOOL_TIMEOUT"),
        ("检索 [[fault:tool_schema]]", "TOOL_BAD_RESPONSE"),
    ],
)
def test_faults_have_trace_and_release_conversation(client, message, code):
    headers, _ = register(client)
    cid = conversation(client, headers)
    state = wait(client, headers, run(client, headers, cid, message))
    assert state["status"] == "failed" and state["error_code"] == code
    assert len(state["trace_id"]) == 32 and int(state["trace_id"], 16) > 0
    assert wait(client, headers, run(client, headers, cid))["status"] == "completed"


def test_deleted_preference_is_not_reused(client):
    headers, _ = register(client)
    memory = client.post(
        "/api/v1/memories",
        json={"kind": "language_preference", "value": "en", "consent": True},
        headers=headers,
    ).json()
    first = wait(client, headers, run(client, headers, conversation(client, headers), "查询偏好"))
    assert "language_preference" in first["answer"]
    assert client.delete(f"/api/v1/memories/{memory['id']}", headers=headers).status_code == 200
    second = wait(client, headers, run(client, headers, conversation(client, headers), "查询偏好"))
    assert "language_preference" not in second["answer"]


def test_rag_citations_respect_tenant_scope(client):
    a, _ = register(client)
    b, _ = register(client)
    content = "检索 ALPHAZX 专属演示资料。这只是授权隔离测试资料。"
    doc = client.post(
        "/api/v1/documents",
        json={
            "title": "tenant-only",
            "source": "synthetic://" + uuid4().hex,
            "source_version": "v1",
            "license": "CC0",
            "published_at": "2026-10-21",
            "content": content,
        },
        headers=a,
    ).json()
    result = wait(client, a, run(client, a, conversation(client, a), content))
    assert any(c["document_id"] == doc["document_id"] for c in result["citations"])
    chunk = next(c["chunk_id"] for c in result["citations"] if c["document_id"] == doc["document_id"])
    assert client.get(f"/api/v1/references/{chunk}", headers=b).status_code == 404
    other = wait(client, b, run(client, b, conversation(client, b), content))
    assert all(c["document_id"] != doc["document_id"] for c in other["citations"])


def test_expired_run_can_be_recovered(client):
    headers, data = register(client)
    cid = conversation(client, headers)
    svc = client.app.state.services
    svc.runs.stopping = True
    time.sleep(0.6)
    rid = run(client, headers, cid)

    async def expire():
        async with svc.sessions() as session, session.begin():
            row = await session.get(AgentRun, rid, with_for_update=True)
            row.status = "running"
            row.worker_id = "dead-process"
            row.lease_until = now() - timedelta(seconds=30)
        await svc.runs.recover_expired()

    client.portal.call(expire)
    state = client.get(f"/api/v1/runs/{rid}", headers=headers).json()
    assert state["error_code"] == "RUN_INTERRUPTED" and state["status"] == "failed"
    assert client.get("/api/v1/conversations", headers=headers).json()[0]["active_run_id"] is None


def test_restart_preserves_business_history_and_checkpoint(client, settings):
    headers, _ = register(client)
    cid = conversation(client, headers)
    first = run(client, headers, cid, "第一轮合成信息")
    assert wait(client, headers, first)["status"] == "completed"
    with TestClient(create_app(settings)) as restarted:
        messages = restarted.get(f"/api/v1/conversations/{cid}/messages", headers=headers).json()
        assert any(m["content"] == "第一轮合成信息" for m in messages)
        second = run(restarted, headers, cid, "继续第二轮")
        assert wait(restarted, headers, second)["status"] == "completed"

        async def checkpoint_messages():
            svc = restarted.app.state.services
            async with svc.sessions() as session:
                conv = await session.get(Conversation, cid)
            state = await svc.graph.aget_state({"configurable": {"thread_id": conv.thread_id}})
            return [m.content for m in state.values["messages"]]

        checkpoint = restarted.portal.call(checkpoint_messages)
        assert "第一轮合成信息" in checkpoint and "继续第二轮" in checkpoint


@pytest.mark.skipif(not os.environ.get("TEST_REDIS_URL"), reason="requires isolated Redis database")
def test_confirm_and_worker_are_idempotent(client, settings):
    headers, _ = register(client)
    job = client.post(
        "/api/v1/jobs/drafts", json={"title": "synthetic"}, headers={**headers, "Idempotency-Key": "job-key"}
    ).json()
    assert job["status"] == "draft"
    for _ in range(2):
        response = client.post(f"/api/v1/jobs/{job['id']}/confirm", json={"confirmed": True}, headers=headers)
        assert response.status_code == 202
    asyncio.run(execute_mock_job(settings, job["id"]))
    one = client.get(f"/api/v1/jobs/{job['id']}", headers=headers).json()
    asyncio.run(execute_mock_job(settings, job["id"]))
    two = client.get(f"/api/v1/jobs/{job['id']}", headers=headers).json()
    assert one["status"] == two["status"] == "succeeded"
    assert one["result"]["asset_id"] == two["result"]["asset_id"]
    assert (
        client.get(f"/api/v1/assets/{one['result']['asset_id']}/download", headers=headers).status_code == 200
    )


def test_validation_error_does_not_echo_password(client):
    response = client.post("/api/v1/auth/register", json={"email": "invalid", "password": "TOP_SECRET"})
    assert response.status_code == 422
    assert "TOP_SECRET" not in response.text


def test_tool_loop_has_a_fixed_budget(client):
    headers, _ = register(client)
    rid = run(client, headers, conversation(client, headers), "[[fault:loop]]")
    state = wait(client, headers, rid)
    assert state["error_code"] == "TOOL_BUDGET_EXCEEDED"
    stream = client.get(f"/api/v1/runs/{rid}/events", headers=headers).text
    assert stream.count("event: tool_start") == client.app.state.services.graph_policy["max_tool_calls"]


def test_running_cancellation_releases_the_conversation(client):
    headers, _ = register(client)
    cid = conversation(client, headers)
    rid = run(client, headers, cid, "[[fault:slow]]")
    for _ in range(100):
        if client.get(f"/api/v1/runs/{rid}", headers=headers).json()["status"] == "running":
            break
        time.sleep(0.02)
    response = client.post(f"/api/v1/runs/{rid}/cancel", headers=headers)
    assert response.status_code == 200
    assert wait(client, headers, rid)["status"] == "cancelled"
    assert wait(client, headers, run(client, headers, cid))["status"] == "completed"


def test_source_versions_cannot_be_silently_overwritten(client):
    headers, _ = register(client)
    item = {
        "title": "immutable",
        "source": "synthetic://" + uuid4().hex,
        "source_version": "v1",
        "license": "CC0",
        "published_at": "2026-10-21",
        "content": "original synthetic content",
    }
    assert client.post("/api/v1/documents", json=item, headers=headers).status_code == 201
    response = client.post("/api/v1/documents", json={**item, "content": "changed"}, headers=headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DOCUMENT_VERSION_CONFLICT"
