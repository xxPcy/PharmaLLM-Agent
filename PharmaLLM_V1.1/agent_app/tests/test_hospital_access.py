"""HTTP/RBAC and policy tests against isolated SQLite; no broker, inference or live DB.

SQLite exercises queries/transactions but does not verify PostgreSQL row locks.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateTable
from langgraph.checkpoint.memory import InMemorySaver

from pharma_agent.api import router as api_router
from pharma_agent.db import (
    AgentRun,
    Asset,
    AuditEvent,
    Conversation,
    Department,
    Message,
    RunEvent,
    SegmentationReview,
    TaskOutbox,
    Tenant,
    ToolJob,
    User,
    UserMemory,
    now,
)
from pharma_agent.errors import AppError
from pharma_agent.graph import RunContext, build_graph
from pharma_agent.hospital import router as hospital_router
from pharma_agent.imaging import ImagingStore
from pharma_agent.jobs import QueueService
from pharma_agent.main import error_response
from pharma_agent.models import ModelGateway, config_load
from pharma_agent.runs import RunService
from pharma_agent.schemas import RegisterInput, ReviewDecision, SegmentationInput, TeamInput
from pharma_agent.security import Actor, issue_token, password_hasher
from pharma_agent.segmentation import SegmentationGateway
from pharma_agent.settings import Settings
from pharma_agent.storage import LocalStorage
from pharma_agent.tools import ToolContext, ToolGateway, ToolRegistry


@pytest.fixture
def hospital(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        for cls in (
            Tenant,
            Department,
            User,
            UserMemory,
            Conversation,
            AgentRun,
            Message,
            RunEvent,
            Asset,
            ToolJob,
            TaskOutbox,
            SegmentationReview,
            AuditEvent,
        ):
            ddl = str(CreateTable(cls.__table__).compile(engine))
            # Only the fixture needs SQLite's INTEGER rowid auto-increment syntax.
            if cls is RunEvent:
                ddl = ddl.replace("id BIGINT NOT NULL", "id INTEGER NOT NULL")
            connection.exec_driver_sql(ddl)

    def normalized(row):
        if row is not None and hasattr(row, "__mapper__"):
            for column in row.__mapper__.columns:
                value = getattr(row, column.key)
                if isinstance(value, datetime) and value.tzinfo is None:
                    set_committed_value(row, column.key, value.replace(tzinfo=UTC))
        return row

    class Values:
        def __init__(self, values):
            self.values = [normalized(row) for row in values]

        def all(self):
            return self.values

        def first(self):
            return self.values[0] if self.values else None

    class Connection:
        def __init__(self, session):
            self.session = session

        async def get(self, model, key, **kwargs):
            return normalized(self.session.get(model, key, **kwargs))

        async def scalar(self, query):
            return normalized(self.session.scalar(query))

        async def scalars(self, query):
            return Values(self.session.scalars(query).all())

        async def execute(self, query):
            return self.session.execute(query)

        def add(self, row):
            self.session.add(row)

        async def flush(self):
            self.session.flush()

        @asynccontextmanager
        async def begin(self):
            with self.session.begin():
                yield

    @asynccontextmanager
    async def sessions():
        with Session(engine, expire_on_commit=False) as session:
            yield Connection(session)

    settings = Settings(
        _env_file=None,
        jwt_secret="synthetic-hospital-secret-" * 3,
        app_env="test",
        model_mode="fake",
        rate_limit_per_minute=0,
        storage_root=tmp_path / "assets",
        segmentation_base_url="http://unused.invalid",
    )
    svc = SimpleNamespace(
        settings=settings,
        sessions=sessions,
        redis=None,
        imaging=ImagingStore(LocalStorage(settings.storage_root), settings),
        segmentation=SegmentationGateway(settings),
    )
    svc.models = ModelGateway(settings, None)
    svc.storage = svc.imaging.storage
    svc.graph_policy = config_load(Path("configs/graph.yaml"))
    svc.tools = ToolGateway(ToolRegistry(settings))
    svc.runs, svc.queue = RunService(svc), QueueService(svc)
    app = FastAPI()
    app.state.services = svc

    @app.middleware("http")
    async def request_ids(request, call_next):
        request.state.request_id = "req-synthetic"
        request.state.trace_id = "0" * 32
        return await call_next(request)

    @app.exception_handler(AppError)
    async def error(request: Request, exc):
        return error_response(request, exc)

    app.include_router(api_router)
    app.include_router(hospital_router)
    hashed = password_hasher.hash("synthetic-password-1234")
    actors = {
        "admin": Actor("admin", "t", "owner", "bone"),
        "doctor": Actor("doctor", "t", "doctor", "bone"),
        "reviewer": Actor("reviewer", "t", "reviewer", "bone"),
        "other_dept": Actor("other-dept", "t", "reviewer", "heart"),
        "other_tenant": Actor("other-tenant", "other", "reviewer", "other-bone"),
    }
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Tenant(id="t", name="Synthetic", login_code="guke"),
                Tenant(id="other", name="Other", login_code="other"),
            ]
        )
        session.flush()
        session.add_all(
            [
                Department(id=id, tenant_id=tenant, name=id)
                for id, tenant in [("bone", "t"), ("heart", "t"), ("other-bone", "other")]
            ]
        )
        session.flush()
        session.add_all(
            [
                User(
                    id=a.id,
                    tenant_id=a.tenant_id,
                    role=a.role,
                    department_id=a.department_id,
                    email=a.id + "@example.org",
                    display_name=a.id,
                    password_hash=hashed,
                )
                for a in actors.values()
            ]
        )
        session.flush()
        session.add(Conversation(id="c", tenant_id="t", owner_user_id="doctor", thread_id="th"))

    def headers(name):
        return {"Authorization": "Bearer " + issue_token(settings, actors[name])}

    def case(number=1):
        source_id, mask_id = "asset_" + f"{number:032x}", "asset_" + f"{number + 100:032x}"
        image = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.diag([2.0, 3.0, 4.0, 1.0]))
        image.header.set_xyzt_units("mm")
        source = svc.imaging.save_volume(source_id, image)
        source.update(
            kind="volume",
            name="synthetic.nii.gz",
            channels=1,
            modality="T1",
            conversation_id="c",
            expires_at=(now() + timedelta(days=1)).isoformat(),
        )
        svc.imaging.save_meta(source_id, source)
        data = np.zeros(image.shape, np.uint8)
        data[:2, :2, :2] = 1
        mask = nib.Nifti1Image(data, image.affine, image.header)
        meta = svc.imaging.accept_mask(
            mask_id, mask.to_bytes(), source_id, {"0": "background", "1": "femoralhead", "2": "acetabulum"}
        )
        meta.update(conversation_id="c", expires_at=source["expires_at"])
        svc.imaging.save_meta(mask_id, meta)
        job_id = "job_" + f"{number:032x}"
        with Session(engine) as session, session.begin():
            for asset_id in (source_id, mask_id):
                svc.imaging.storage.write(
                    asset_id + ".bin", image.to_bytes() if asset_id == source_id else mask.to_bytes()
                )
                session.add(
                    Asset(
                        id=asset_id,
                        tenant_id="t",
                        owner_user_id="doctor",
                        object_key=asset_id + ".bin",
                        file_hash="synthetic",
                        mime="application/gzip",
                        size_bytes=1024,
                        status="ready",
                    )
                )
            session.flush()
            session.add(
                ToolJob(
                    id=job_id,
                    tenant_id="t",
                    user_id="doctor",
                    tool_name="segment_medical_volume",
                    status="succeeded",
                    idempotency_key=job_id,
                    input_json={"conversation_id": "c", "file_id": source_id},
                    result_json={"asset_id": mask_id, "source_asset_id": source_id},
                )
            )
        return job_id, source_id, mask_id

    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, engine=engine, svc=svc, actors=actors, headers=headers, case=case
        )
    svc.queue.celery.close()
    engine.dispose()


def test_readable_team_login_legacy_id_and_new_registration(hospital):
    h = hospital
    for code in ("GUKE", "t"):
        response = h.client.post(
            "/api/v1/auth/login",
            json={"tenant_id": code, "email": "admin@example.org", "password": "synthetic-password-1234"},
        )
        assert response.status_code == 200
        assert response.json()["tenant_code"] == "guke"
    payload = {
        "tenant_code": "Research-Lab",
        "tenant_name": "Research",
        "email": "new@example.org",
        "password": "TestPass123",
    }
    response = h.client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    assert response.json()["tenant_code"] == "research-lab"
    assert response.json()["role"] == "owner"
    assert h.client.post("/api/v1/auth/register", json=payload).status_code == 409
    assert RegisterInput(**payload).tenant_code == "research-lab"


def test_numeric_team_registration_login_and_rename(hospital):
    h = hospital
    payload = {"tenant_code": "2026", "tenant_name": "Synthetic numeric team",
               "email": "numeric@example.org", "password": "TestPass123"}
    response = h.client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    account = response.json()
    assert account["tenant_code"] == "2026" and account["tenant_id"].startswith("tenant_")
    login = {"tenant_id": "2026", "email": payload["email"], "password": payload["password"]}
    assert h.client.post("/api/v1/auth/login", json=login).status_code == 200
    assert h.client.post("/api/v1/auth/register", json=payload).status_code == 409
    headers = {"Authorization": "Bearer " + account["access_token"]}
    renamed = h.client.patch("/api/v1/organization", headers=headers,
                             json={"name": payload["tenant_name"], "login_code": "002026"})
    assert renamed.status_code == 200 and renamed.json()["tenant_code"] == "002026"
    assert h.client.post("/api/v1/auth/login", json={**login, "tenant_id": "002026"}).status_code == 200
    assert h.client.post("/api/v1/auth/login", json={**login, "tenant_id": account["tenant_id"]}).status_code == 200


def test_new_password_boundaries_for_registration_and_member_creation(hospital):
    h = hospital
    for length in (5, 6, 12, 13):
        password = "1" * length
        payload = {"tenant_code": f"password-{length}", "email": f"boundary{length}@example.org",
                   "password": password}
        response = h.client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == (201 if 6 <= length <= 12 else 422)
        if response.status_code == 201:
            assert h.client.post("/api/v1/auth/login", json={"tenant_id": payload["tenant_code"],
                "email": payload["email"], "password": password}).status_code == 200
        member = {"email": f"member{length}@example.org", "display_name": "Synthetic",
                  "password": password, "department_id": "bone", "role": "doctor"}
        response = h.client.post("/api/v1/organization/members", headers=h.headers("admin"), json=member)
        assert response.status_code == (201 if 6 <= length <= 12 else 422)


def test_team_code_normalization_and_invalid_input():
    assert TeamInput(name="Synthetic", login_code=" 2026 ").login_code == "2026"
    assert TeamInput(name="Synthetic", login_code=" 2026-LAB ").login_code == "2026-lab"
    for code in ("26", "-2026", "20_26"):
        with pytest.raises(ValidationError):
            TeamInput(name="Synthetic", login_code=code)


def test_organization_management_and_last_admin(hospital):
    h = hospital
    assert h.client.get("/api/v1/organization", headers=h.headers("doctor")).status_code == 403
    payload = {
        "email": "new-doctor@example.org",
        "password": "TestPass123",
        "display_name": "Synthetic doctor",
        "role": "doctor",
        "department_id": "other-bone",
    }
    assert (
        h.client.post("/api/v1/organization/members", headers=h.headers("admin"), json=payload).status_code
        == 422
    )
    payload["department_id"] = "bone"
    assert (
        h.client.post("/api/v1/organization/members", headers=h.headers("admin"), json=payload).status_code
        == 201
    )
    change = {"display_name": "Admin", "role": "doctor", "department_id": "bone", "status": "active"}
    assert (
        h.client.patch("/api/v1/organization/members/admin", headers=h.headers("admin"), json=change).json()[
            "error"
        ]["code"]
        == "LAST_ADMIN"
    )
    assert (
        h.client.patch(
            "/api/v1/organization",
            headers=h.headers("admin"),
            json={"name": "New team", "login_code": "bone-lab"},
        ).status_code
        == 200
    )
    assert (
        h.client.patch(
            "/api/v1/organization",
            headers=h.headers("admin"),
            json={"name": "New team", "login_code": "other"},
        ).status_code
        == 409
    )


def test_review_scope_assets_and_decision(hospital):
    h = hospital
    job_id, source, mask = h.case()
    assert h.client.get("/api/v1/reviews", headers=h.headers("reviewer")).json() == []
    response = h.client.post(f"/api/v1/jobs/{job_id}/review", headers=h.headers("doctor"))
    assert response.status_code == 201
    review = response.json()
    assert (
        h.client.post(f"/api/v1/jobs/{job_id}/review", headers=h.headers("doctor")).json()["id"]
        == review["id"]
    )
    for who in ("other_dept", "other_tenant"):
        assert h.client.get("/api/v1/reviews", headers=h.headers(who)).json() == []
        assert (
            h.client.get(
                f"/api/v1/reviews/{review['id']}/assets/{source}/metadata", headers=h.headers(who)
            ).status_code
            == 404
        )
    base = f"/api/v1/reviews/{review['id']}/assets/"
    assert h.client.get(base + mask + "/metadata", headers=h.headers("reviewer")).status_code == 200
    assert (
        h.client.get(
            base + source + "/slice", params={"mask_id": mask}, headers=h.headers("reviewer")
        ).status_code
        == 200
    )
    assert h.client.get(base + mask + "/surface", headers=h.headers("reviewer")).status_code == 200
    assert h.client.get(base + mask + "/download", headers=h.headers("reviewer")).status_code == 200
    assert h.client.get("/api/v1/conversations/c/messages", headers=h.headers("reviewer")).status_code == 404
    decision_url = f"/api/v1/reviews/{review['id']}/decision"
    assert (
        h.client.post(decision_url, json={"decision": "approved"}, headers=h.headers("doctor")).status_code
        == 403
    )
    response = h.client.post(
        decision_url,
        json={"decision": "approved", "comment": "Synthetic review"},
        headers=h.headers("reviewer"),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert (
        h.client.post(
            decision_url,
            json={"decision": "changes_requested", "comment": "Changed"},
            headers=h.headers("reviewer"),
        ).status_code
        == 409
    )
    assert (
        h.client.get("/api/v1/reviews", headers=h.headers("doctor")).json()[0]["comment"]
        == "Synthetic review"
    )


def test_review_rejects_unrelated_deleted_or_expired_mask(hospital):
    h = hospital
    job, source, mask = h.case()
    review = h.client.post(f"/api/v1/jobs/{job}/review", headers=h.headers("doctor")).json()
    _, unrelated, _ = h.case(2)
    base = f"/api/v1/reviews/{review['id']}/assets/"
    assert h.client.get(base + unrelated + "/metadata", headers=h.headers("reviewer")).status_code == 404
    assert (
        h.client.get(
            base + source + "/slice", params={"mask_id": unrelated}, headers=h.headers("reviewer")
        ).status_code
        == 404
    )
    with Session(h.engine) as session:
        meta = h.svc.imaging.metadata(session.get(Asset, mask))
    meta["expires_at"] = (now() - timedelta(days=1)).isoformat()
    h.svc.imaging.save_meta(mask, meta)
    assert h.client.get(base + mask + "/metadata", headers=h.headers("reviewer")).status_code == 410
    with Session(h.engine) as session, session.begin():
        session.get(Asset, mask).status = "deleted"
    assert h.client.get(base + mask + "/metadata", headers=h.headers("reviewer")).status_code == 404


def test_user_mode_role_revocation_and_request_snapshot(hospital):
    h = hospital
    path = "/api/v1/users/me/execution-mode"
    assert (
        h.client.patch(
            path, json={"execution_mode": "full_access"}, headers=h.headers("reviewer")
        ).status_code
        == 403
    )
    assert (
        h.client.patch(path, json={"execution_mode": "full_access"}, headers=h.headers("doctor")).status_code
        == 200
    )
    assert (
        h.client.get("/api/v1/users/me", headers=h.headers("doctor")).json()["execution_mode"]
        == "full_access"
    )
    hdrs = {**h.headers("doctor"), "Idempotency-Key": "same-request"}
    payload = {"message": "synthetic", "execution_mode": "full_access"}
    response = h.client.post("/api/v1/conversations/c/runs", headers=hdrs, json=payload)
    assert response.status_code == 202
    rid = response.json()["run_id"]
    assert h.client.post("/api/v1/conversations/c/runs", headers=hdrs, json=payload).json()["run_id"] == rid
    assert (
        h.client.post(
            "/api/v1/conversations/c/runs",
            headers=hdrs,
            json={**payload, "execution_mode": "ask_for_approval"},
        ).status_code
        == 409
    )
    with Session(h.engine) as session, session.begin():
        assert session.get(AgentRun, rid).model_snapshot["execution_mode"] == "full_access"
        session.get(User, "doctor").role = "reviewer"
    # Token contains identity, not the old role; the next call reads current DB permissions.
    assert (
        h.client.patch(path, json={"execution_mode": "full_access"}, headers=h.headers("doctor")).status_code
        == 403
    )


@pytest.mark.parametrize("mode", ["ask_for_approval", "full_access"])
async def test_real_tool_policy_queues_once_without_running_inference(hospital, mode):
    h = hospital
    _, source, _ = h.case()
    context = ToolContext(h.actors["doctor"], "req", "run-policy", h.svc, "c", (source,), execution_mode=mode)
    args = {"file_id": source, "model_id": "femoralhead-t1-best", "task_type": "femoral_head_segmentation"}
    result = await h.svc.tools.invoke("segment_medical_volume", args, context)
    assert result["status"] == ("queued" if mode == "full_access" else "draft")
    assert result["requires_confirmation"] == (mode == "ask_for_approval")
    again = await h.svc.tools.invoke("segment_medical_volume", args, context)
    assert again["job_id"] == result["job_id"]
    with Session(h.engine) as session:
        outbox = session.scalars(select(TaskOutbox)).all()
        assert len(outbox) == (1 if mode == "full_access" else 0)
        if outbox:
            assert not outbox[0].published
    with pytest.raises(AppError) as failure:
        await h.svc.tools.invoke(
            "segment_medical_volume",
            args,
            ToolContext(h.actors["reviewer"], "req", "forbidden", h.svc, "c", (source,), execution_mode=mode),
        )
    assert failure.value.code == "PERMISSION_DENIED"
    with pytest.raises(ValidationError):
        SegmentationInput(**args, execution_mode="full_access")


def test_review_rejection_requires_comment():
    with pytest.raises(ValueError):
        ReviewDecision(decision="changes_requested", comment=" ")


async def test_auto_confirm_cannot_escalate_a_manual_draft(hospital):
    h = hospital
    _, source, _ = h.case()
    args = {"file_id": source, "model_id": "femoralhead-t1-best", "task_type": "femoral_head_segmentation"}
    context = ToolContext(h.actors["doctor"], "req", "manual", h.svc, "c", (source,))
    result = await h.svc.tools.invoke("segment_medical_volume", args, context)
    with pytest.raises(AppError) as failure:
        await h.svc.queue.confirm(h.actors["doctor"], result["job_id"], automatic=True)
    assert failure.value.code == "PERMISSION_DENIED"
    await h.svc.queue.confirm(h.actors["doctor"], result["job_id"])
    await h.svc.queue.confirm(h.actors["doctor"], result["job_id"])
    with Session(h.engine) as session:
        assert len(session.scalars(select(TaskOutbox)).all()) == 1


async def test_role_revocation_blocks_a_previously_authorized_tool_context(hospital):
    h = hospital
    _, source, _ = h.case()
    args = {"file_id": source, "model_id": "femoralhead-t1-best", "task_type": "femoral_head_segmentation"}
    context = ToolContext(
        h.actors["doctor"], "req", "old-context", h.svc, "c", (source,), execution_mode="full_access"
    )
    with Session(h.engine) as session, session.begin():
        session.get(User, "doctor").role = "reviewer"
    with pytest.raises(AppError) as failure:
        await h.svc.tools.invoke("segment_medical_volume", args, context)
    assert failure.value.code == "PERMISSION_DENIED"
    with Session(h.engine) as session:
        assert session.scalars(select(TaskOutbox)).all() == []


def test_submission_requires_department_and_a_completed_result(hospital):
    h = hospital
    job, _, _ = h.case()
    with Session(h.engine) as session, session.begin():
        session.get(User, "doctor").department_id = None
    assert h.client.post(f"/api/v1/jobs/{job}/review", headers=h.headers("doctor")).status_code == 422
    with Session(h.engine) as session, session.begin():
        session.get(User, "doctor").department_id = "bone"
        session.get(ToolJob, job).status = "running"
    assert h.client.post(f"/api/v1/jobs/{job}/review", headers=h.headers("doctor")).status_code == 409
    assert h.client.post(f"/api/v1/jobs/{job}/review", headers=h.headers("reviewer")).status_code == 403


def test_admin_cannot_review_their_own_submission(hospital):
    h = hospital
    job, source, mask = h.case()
    with Session(h.engine) as session, session.begin():
        session.get(ToolJob, job).user_id = "admin"
        for id in (source, mask):
            session.get(Asset, id).owner_user_id = "admin"
    review = h.client.post(f"/api/v1/jobs/{job}/review", headers=h.headers("admin")).json()
    path = f"/api/v1/reviews/{review['id']}/decision"
    assert h.client.post(path, json={"decision": "approved"}, headers=h.headers("admin")).status_code == 403
    result = h.client.post(
        path,
        json={"decision": "changes_requested", "comment": "Synthetic reason"},
        headers=h.headers("reviewer"),
    )
    assert result.status_code == 200
    assert result.json()["status"] == "changes_requested"


def test_review_permission_follows_live_department_and_account_status(hospital):
    h = hospital
    job, source, _ = h.case()
    review = h.client.post(f"/api/v1/jobs/{job}/review", headers=h.headers("doctor")).json()
    path = f"/api/v1/reviews/{review['id']}/assets/{source}/metadata"
    assert h.client.get(path, headers=h.headers("reviewer")).status_code == 200
    with Session(h.engine) as session, session.begin():
        session.get(User, "reviewer").department_id = "heart"
    assert h.client.get(path, headers=h.headers("reviewer")).status_code == 404
    with Session(h.engine) as session, session.begin():
        session.get(User, "reviewer").status = "disabled"
    assert h.client.get(path, headers=h.headers("reviewer")).status_code == 401


async def test_full_access_policy_reaches_real_graph_and_outbox(hospital):
    h = hospital
    _, source, _ = h.case()
    response = h.client.post(
        "/api/v1/conversations/c/runs",
        headers=h.headers("doctor"),
        json={"message": "帮我分割股骨头和髋臼", "asset_ids": [source], "execution_mode": "full_access"},
    )
    assert response.status_code == 202
    rid = response.json()["run_id"]
    with Session(h.engine) as session:
        snapshot = session.get(AgentRun, rid).model_snapshot
    route = {
        "action": "tool",
        "tool_name": "segment_medical_volume",
        "arguments": {
            "file_id": source,
            "model_id": "femoralhead-t1-best",
            "task_type": "femoral_head_segmentation",
        },
    }
    router = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(content=json.dumps(route), finish_reason="stop", usage={})
        )
    )
    answer = SimpleNamespace(stream=AsyncMock(side_effect=AssertionError("Use validated task receipt")))
    h.svc.models = SimpleNamespace(get=lambda role, snapshot: router if role == "agent_router" else answer)
    h.svc.segmentation.infer = AsyncMock(side_effect=AssertionError("No inference in this test"))
    ctx = RunContext(h.actors["doctor"], "req", rid, "c", snapshot, h.svc, AsyncMock())
    result = await build_graph(h.svc, InMemorySaver()).ainvoke(
        {"status": "running"}, {"configurable": {"thread_id": "synthetic-full-access"}}, context=ctx
    )
    assert "已排队" in result["final_answer"] and "点击" not in result["final_answer"]
    assert ctx.tool_context().execution_mode == "full_access"
    with Session(h.engine) as session:
        outbox = session.scalars(select(TaskOutbox)).all()
        assert len(outbox) == 1
        assert session.get(ToolJob, outbox[0].job_id).status == "queued"
    h.svc.segmentation.infer.assert_not_awaited()
