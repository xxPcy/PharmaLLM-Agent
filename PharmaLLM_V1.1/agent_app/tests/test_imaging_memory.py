"""Real CPU NIfTI measurements and isolated SQLite ownership/context queries; no inference."""

from contextlib import asynccontextmanager
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import nibabel as nib
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable
from langgraph.checkpoint.memory import InMemorySaver

from pharma_agent.db import AgentRun, Asset, AuditEvent, Message, ToolJob, User, now
from pharma_agent.graph import RunContext, build_graph
from pharma_agent.imaging import ImagingStore
from pharma_agent.imaging_context import imaging_evidence, load_conversation_imaging, measurements
from pharma_agent.models import config_load
from pharma_agent.security import Actor
from pharma_agent.segmentation import SegmentationGateway
from pharma_agent.settings import Settings
from pharma_agent.storage import LocalStorage
from pharma_agent.tools import ToolGateway, ToolRegistry


@pytest.fixture
def imaging_session(tmp_path):
    engine = create_engine("sqlite://")
    # PostgreSQL's partial active-run index has no equivalent in this SQLite fixture.
    # Exercise portable queries/ownership using real table constraints, without PG-only indexes.
    with engine.begin() as connection:
        for cls in (Asset, ToolJob, AgentRun, Message, AuditEvent, User):
            connection.execute(CreateTable(cls.__table__))
    imaging = ImagingStore(
        LocalStorage(tmp_path / "assets"), Settings(_env_file=None, jwt_secret="isolated-imaging-memory-" * 3)
    )

    class Connection:
        def __init__(self, session):
            self.session = session

        async def scalars(self, query):
            return self.session.scalars(query)

        async def scalar(self, query):
            return self.session.scalar(query)

        async def flush(self):
            self.session.flush()

        def add(self, row):
            self.session.add(row)

        @asynccontextmanager
        async def begin(self):
            with self.session.begin():
                yield

    @asynccontextmanager
    async def sessions():
        with Session(engine, expire_on_commit=False) as session:
            yield Connection(session)

    def add_case(number, tenant="t", user="u", conversation="c", units="mm"):
        source_id, mask_id = "asset_" + f"{number:032x}", "asset_" + f"{number + 100:032x}"
        image = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.diag([2, 3, 4, 1]))
        image.header.set_xyzt_units(units)
        source = imaging.save_volume(source_id, image)
        source.update(
            kind="volume",
            name=f"synthetic-{number}.nii.gz",
            channels=1,
            modality="T1",
            conversation_id=conversation,
            expires_at=(now() + timedelta(days=1)).isoformat(),
        )
        imaging.save_meta(source_id, source)
        data = np.zeros(image.shape, np.uint8)
        data[:2, :2, :2] = 1
        data[2:, 2:, 2:] = 2
        mask = nib.Nifti1Image(data, image.affine, image.header)
        meta = imaging.accept_mask(
            mask_id, mask.to_bytes(), source_id, {"0": "background", "1": "femoralhead", "2": "acetabulum"}
        )
        meta.update(conversation_id=conversation, expires_at=source["expires_at"])
        imaging.save_meta(mask_id, meta)
        run_id, job_id = "run_" + f"{number:032x}", "job_" + f"{number:032x}"
        with Session(engine) as session, session.begin():
            session.add_all(
                [
                    Asset(
                        id=asset_id,
                        tenant_id=tenant,
                        owner_user_id=user,
                        object_key=asset_id + ".bin",
                        file_hash="synthetic",
                        mime="application/gzip",
                        size_bytes=1024,
                        status="ready",
                    )
                    for asset_id in (source_id, mask_id)
                ]
            )
            session.add(
                AgentRun(
                    id=run_id,
                    tenant_id=tenant,
                    user_id=user,
                    conversation_id=conversation,
                    idempotency_key=run_id,
                    request_hash="synthetic",
                    request_id="req",
                    trace_id="0" * 32,
                    status="completed",
                    model_snapshot={"asset_ids": [source_id]},
                    graph_version="fixture",
                )
            )
            session.add(
                ToolJob(
                    id=job_id,
                    tenant_id=tenant,
                    user_id=user,
                    tool_name="segment_medical_volume",
                    tool_version="1.0",
                    idempotency_key=job_id,
                    status="succeeded",
                    input_json={
                        "file_id": source_id,
                        "file_name": source["name"],
                        "conversation_id": conversation,
                    },
                    result_json={"asset_id": mask_id, "source_asset_id": source_id},
                )
            )
        return source_id, mask_id, run_id, job_id

    yield SimpleNamespace(engine=engine, sessions=sessions, imaging=imaging, add_case=add_case)
    engine.dispose()


async def test_context_uses_current_conversation_and_real_volume_after_attachment_is_sent(imaging_session):
    svc = imaging_session
    source, mask, _, job = svc.add_case(1)
    svc.add_case(2, conversation="another-conversation")
    svc.add_case(3, user="another-user")
    svc.add_case(4, tenant="another-tenant")
    context = await load_conversation_imaging(svc, Actor("u", "t", "member"), "c", "new-run")
    assert [row["asset_id"] for row in context["files"]] == [source]
    assert [row["job_id"] for row in context["jobs"]] == [job]
    result = context["jobs"][0]
    assert result["result_available"] and result["result_asset_id"] == mask
    femoralhead = result["measurements"][0]
    assert femoralhead["display_name"] == "股骨头"
    assert femoralhead["voxel_count"] == 8
    assert femoralhead["volume_mm3"] == pytest.approx(192)
    assert femoralhead["volume_ml"] == pytest.approx(0.192)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_accepted_attachment_survives_later_generation_failure_or_cancel(imaging_session, status):
    svc = imaging_session
    source, _, run, job = svc.add_case(1)
    with Session(svc.engine) as session, session.begin():
        session.get(AgentRun, run).status = status
        session.delete(session.get(ToolJob, job))
    context = await load_conversation_imaging(svc, Actor("u", "t", "member"), "c", "new-run")
    assert [row["asset_id"] for row in context["files"]] == [source]
    assert context["jobs"] == []


async def test_an_upload_never_sent_or_used_in_a_task_is_not_implicitly_attached(imaging_session):
    svc = imaging_session
    _, _, run, job = svc.add_case(1)
    with Session(svc.engine) as session, session.begin():
        session.delete(session.get(ToolJob, job))
        session.delete(session.get(AgentRun, run))
    context = await load_conversation_imaging(svc, Actor("u", "t", "member"), "c", "new-run")
    assert context == {"files": [], "jobs": []}


@pytest.mark.parametrize(
    "unavailable", ["source_deleted", "result_deleted", "result_expired", "wrong_source"]
)
async def test_unavailable_or_mismatched_results_never_reuse_old_measurements(imaging_session, unavailable):
    svc = imaging_session
    source, mask, _, _ = svc.add_case(1)
    if unavailable.endswith("deleted"):
        with Session(svc.engine) as session, session.begin():
            session.get(Asset, source if unavailable == "source_deleted" else mask).status = "deleted"
    else:
        with Session(svc.engine) as session:
            meta = svc.imaging.metadata(session.get(Asset, mask))
        if unavailable == "result_expired":
            meta["expires_at"] = (now() - timedelta(days=1)).isoformat()
        else:
            meta["source_asset_id"] = "asset_" + "f" * 32
        svc.imaging.save_meta(mask, meta)
    context = await load_conversation_imaging(svc, Actor("u", "t", "member"), "c", "new-run")
    assert not context["jobs"][0]["result_available"]
    assert "volume_mm3" not in json.dumps(context)


async def test_unknown_physical_units_keep_voxel_counts_without_inventing_volume(imaging_session):
    svc = imaging_session
    svc.add_case(1, units="unknown")
    context = await load_conversation_imaging(svc, Actor("u", "t", "member"), "c", "new-run")
    assert context["jobs"][0]["result_available"]
    assert context["jobs"][0]["measurements"][0]["voxel_count"] == 8
    assert "volume_mm3" not in json.dumps(context) and "volume_ml" not in json.dumps(context)
    assert "volume_mm3" not in json.dumps(measurements([{"volume_mm3": float("nan")}]))


async def test_default_job_query_keeps_other_conversation_segmentations_out(imaging_session, monkeypatch):
    from datetime import timezone
    from pharma_agent import tools as module
    from pharma_agent.schemas import JobQuery
    from pharma_agent.tools import GetMyJob, ToolContext

    svc = imaging_session
    _, _, _, own = svc.add_case(1)
    svc.add_case(2, conversation="another-conversation")
    svc.add_case(3, user="another-user")
    original = module.job_dict

    def job_dict(row):
        # SQLite doesn't preserve DateTime(timezone=True); production PostgreSQL does.
        row.created_at = row.created_at.replace(tzinfo=timezone.utc)
        return original(row)

    monkeypatch.setattr(module, "job_dict", job_dict)
    context = ToolContext(Actor("u", "t", "member"), "req", "run", svc, "c")
    result = await GetMyJob().execute(JobQuery(), context)
    assert [row.id for row in result.jobs] == [own]


@pytest.mark.parametrize("route", ['{"action":"respond"}', "invalid-json"])
async def test_followup_answer_receives_measurements_without_reattachment_or_new_segmentation(
    imaging_session, monkeypatch, route
):
    from pharma_agent import graph as module

    svc = imaging_session
    source, _, old_run, _ = svc.add_case(1)
    decoding = {
        "history_message_limit": 20,
        "prompt_byte_budget": 12000,
        "router_max_tokens": 384,
        "answer_temperature": 0.2,
        "answer_max_tokens": 768,
    }
    snapshot = {"asset_ids": [], "decoding": decoding}
    with Session(svc.engine) as session, session.begin():
        session.add(
            AgentRun(
                id="run_current",
                tenant_id="t",
                user_id="u",
                conversation_id="c",
                idempotency_key="current",
                request_hash="synthetic",
                request_id="req",
                trace_id="0" * 32,
                status="queued",
                model_snapshot=snapshot,
                graph_version="fixture",
            )
        )
        for index, (role, content, run) in enumerate(
            [
                ("user", "帮我分割股骨头和髋臼", old_run),
                ("assistant", "已准备好分割任务，请确认开始。", old_run),
                ("user", "请问这个股骨头的体积有多少", "run_current"),
            ]
        ):
            session.add(
                Message(
                    id=f"m{index}",
                    conversation_id="c",
                    run_id=run,
                    role=role,
                    content=content,
                    created_at=now() + timedelta(seconds=index),
                )
            )
    monkeypatch.setattr(module, "preferences", AsyncMock(return_value={}))
    captured = []

    class Answer:
        async def stream(self, request):
            captured.append(request.messages)
            payload = request.messages[-1]["content"].split("UNTRUSTED_CONVERSATION_IMAGING", 1)[1]
            evidence = json.loads(payload.split("：\n", 1)[1])
            value = evidence["jobs"][0]["measurements"][0]
            yield SimpleNamespace(
                text=f"股骨头标签总体积为 {value['volume_mm3']:.0f} mm³（{value['volume_ml']:.3f} mL）。",
                usage={},
                finish_reason="stop",
            )

    router = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(content=route, finish_reason="stop", usage={}))
    )
    svc.graph_policy = config_load(Path("configs/graph.yaml"))
    svc.segmentation = SimpleNamespace(public_config=lambda: {"models": []})
    svc.tools = ToolGateway(ToolRegistry(SimpleNamespace(tool_config=Path("configs/tools.yaml"))))
    svc.tools.invoke = AsyncMock(side_effect=AssertionError("A volume followup must not run segmentation"))
    svc.models = SimpleNamespace(get=lambda role, snapshot: router if role == "agent_router" else Answer())
    ctx = RunContext(Actor("u", "t", "member"), "req", "run_current", "c", snapshot, svc, AsyncMock())
    result = await build_graph(svc, InMemorySaver()).ainvoke(
        {"status": "running"}, {"configurable": {"thread_id": "fixture-volume-followup"}}, context=ctx
    )
    assert "192 mm³" in result["final_answer"] and "0.192 mL" in result["final_answer"]
    assert source in ctx.tool_context().asset_ids and ctx.model_snapshot["asset_ids"] == []
    assert [item["role"] for item in captured[0][1:]] == ["user", "assistant", "user"]
    svc.tools.invoke.assert_not_awaited()


async def test_natural_language_reuses_authorized_previous_image_to_create_a_real_draft(
    imaging_session, monkeypatch
):
    from pharma_agent import graph as module

    svc = imaging_session
    source, _, previous_run, _ = svc.add_case(1)
    settings = Settings(
        _env_file=None, jwt_secret="isolated-draft-" * 3, segmentation_base_url="http://no-network-fixture"
    )
    svc.settings = settings
    svc.segmentation = SegmentationGateway(settings)
    spec = next(iter(svc.segmentation.models.values()))
    args = {"file_id": source, "model_id": spec.id, "task_type": spec.task_types[0]}
    router = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                content=json.dumps(
                    {"action": "tool", "tool_name": "segment_medical_volume", "arguments": args}
                ),
                finish_reason="stop",
                usage={},
            )
        )
    )
    answer = SimpleNamespace(stream=AsyncMock(side_effect=AssertionError("Use the actual draft receipt")))
    svc.models = SimpleNamespace(get=lambda role, snapshot: router if role == "agent_router" else answer)
    svc.graph_policy = config_load(Path("configs/graph.yaml"))
    svc.tools = ToolGateway(ToolRegistry(settings))
    # This executes the real draft tool and validation, never HTTP inference or confirmation.
    svc.segmentation.infer = AsyncMock(side_effect=AssertionError("Drafts must not start inference"))
    snapshot = {
        "asset_ids": [],
        "decoding": {
            "history_message_limit": 20,
            "prompt_byte_budget": 12000,
            "router_max_tokens": 384,
            "answer_temperature": 0.2,
            "answer_max_tokens": 768,
        },
    }
    with Session(svc.engine) as session, session.begin():
        session.add(User(id="u", tenant_id="t", email="synthetic@example.org",
                         password_hash="unused-synthetic", role="doctor", status="active"))
        session.add(
            Message(
                id="m_resegment",
                conversation_id="c",
                run_id=previous_run,
                role="user",
                content="再对刚才的影像分割股骨头和髋臼",
            )
        )
    monkeypatch.setattr(module, "preferences", AsyncMock(return_value={}))
    ctx = RunContext(Actor("u", "t", "member"), "req", "run_resegment", "c", snapshot, svc, AsyncMock())
    result = await build_graph(svc, InMemorySaver()).ainvoke(
        {"status": "running"}, {"configurable": {"thread_id": "fixture-natural-language-draft"}}, context=ctx
    )
    assert "确认并开始分割" in result["final_answer"]
    created = result["tool_observations"][-1]["output"]
    with Session(svc.engine) as session:
        job = session.get(ToolJob, created["job_id"])
        assert job.status == "draft" and job.confirmed_at is None
        assert job.input_json["file_id"] == source and job.input_json["model_id"] == spec.id
    assert ctx.tool_context().segmentation_model_id is None  # No dropdown preference selected a model.
    svc.segmentation.infer.assert_not_awaited()


def test_context_budget_retains_a_whole_latest_measurement_without_cutting_json():
    context = {
        "files": [],
        "jobs": [{"job_id": "latest", "measurements": [{"volume_mm3": 192}]}]
        + [{"job_id": str(i), "note": "旧数据" * 1000} for i in range(2)],
    }
    data = json.loads(imaging_evidence(context, byte_budget=500))
    assert len(data["jobs"]) == 1 and data["jobs"][0]["measurements"][0]["volume_mm3"] == 192
