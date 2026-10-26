"""Regression checks for conversational follow-ups, model choice and physical 3D results."""

import io
import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import nibabel as nib
import numpy as np
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from pharma_agent.checkpoint_metadata import MetadataReader
from pharma_agent.errors import AppError
from pharma_agent.graph import RunContext, build_graph
from pharma_agent.models import ModelGateway, config_load
from pharma_agent.security import Actor
from pharma_agent.settings import Settings
from pharma_agent.surfaces import mask_surface
from pharma_agent.tools import ToolGateway, ToolRegistry


@pytest.mark.parametrize("route", ['{"action":"clarify"}', "not-json"])
async def test_followup_reads_history_and_never_executes_unvalidated_actions(monkeypatch, route):
    from pharma_agent import graph as module

    monkeypatch.setattr(module, "preferences", AsyncMock(return_value={}))
    history = [
        SimpleNamespace(id="m3", role="user", content="你刚才说的这个名词不太正确吧？"),
        SimpleNamespace(id="m2", role="assistant", content="上轮回答包含需要重新核对的术语。"),
        SimpleNamespace(id="m1", role="user", content="请解释上一个医学概念。"),
    ]

    @asynccontextmanager
    async def sessions():
        yield SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: history)))

    captured = []
    router = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(content=route, finish_reason="stop", usage={}))
    )

    class Answer:
        async def stream(self, request):
            captured.append(request.messages)
            yield SimpleNamespace(
                text="应结合前文重新核对术语，不能把旧回答当作证据。", usage={}, finish_reason="stop"
            )

    models = SimpleNamespace(get=lambda role, snapshot: router if role == "agent_router" else Answer())
    tools = ToolGateway(ToolRegistry(SimpleNamespace(tool_config=Path("configs/tools.yaml"))))
    tools.invoke = AsyncMock(side_effect=AssertionError("no tool may execute"))
    policy = config_load(Path("configs/graph.yaml"))
    svc = SimpleNamespace(graph_policy=policy, sessions=sessions, models=models, tools=tools)
    snapshot = {
        "decoding": {
            "history_message_limit": 20,
            "prompt_byte_budget": 12000,
            "router_max_tokens": 384,
            "answer_temperature": 0.2,
            "answer_max_tokens": 768,
        }
    }
    ctx = RunContext(
        Actor("fixture-user", "fixture-tenant", "member"), "req", "run", "conv", snapshot, svc, AsyncMock()
    )
    state = await build_graph(svc, InMemorySaver()).ainvoke(
        {"status": "running"}, {"configurable": {"thread_id": "fixture-followup"}}, context=ctx
    )
    assert [row["role"] for row in captured[0][1:]] == ["user", "assistant", "user"]
    assert captured[0][-1]["content"] == history[0].content
    assert "不能把旧回答" in state["final_answer"]
    assert "未能可靠识别" not in state["final_answer"]
    if route == "not-json":
        assert state["error_code"] == "ROUTE_FALLBACK"
        assert "没有执行任何工具" in captured[0][0]["content"]
    tools.invoke.assert_not_awaited()


async def test_chat_choice_is_allowlisted_and_pinned_per_run():
    settings = Settings(_env_file=None, jwt_secret="fixture-model-choice-" * 3, model_mode="openai")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        gateway = ModelGateway(settings, client)
        first = gateway.snapshot()
        gateway.registry["models"]["authorized_second"] = {
            **gateway.registry["models"]["pharma_llm_sft"],
            "model_name": "another-configured-model",
        }
        gateway.registry["selectable_chat_models"] = ["pharma_llm_sft", "authorized_second"]
        selected = gateway.snapshot("authorized_second")
        assert selected["roles"]["medical_answer"] == "authorized_second"
        assert selected["models"]["authorized_second"]["model_name"] == "another-configured-model"
        assert first["roles"]["medical_answer"] == "pharma_llm_sft"
        assert gateway.registry["roles"]["medical_answer"] == "pharma_llm_sft"
        assert "base_url" not in json.dumps(gateway.chat_choices())
        with pytest.raises(AppError) as error:
            gateway.snapshot("arbitrary-model")
        assert error.value.code == "MODEL_NOT_ALLOWED"


@pytest.mark.parametrize("status", ["draft", "queued", "succeeded"])
async def test_segmentation_reply_follows_real_tool_state_without_medical_prose(monkeypatch, status):
    from pharma_agent import graph as module
    from pharma_agent import security

    file_id = "asset_" + "1" * 32
    monkeypatch.setattr(module, "preferences", AsyncMock(return_value={}))
    monkeypatch.setattr(
        module, "load_conversation_imaging", AsyncMock(return_value={"files": [], "jobs": []})
    )
    monkeypatch.setattr(security, "require_owned", AsyncMock(return_value=SimpleNamespace(id=file_id)))

    @asynccontextmanager
    async def sessions():
        yield SimpleNamespace(
            scalars=AsyncMock(
                return_value=SimpleNamespace(
                    all=lambda: [
                        SimpleNamespace(id="m1", role="user", content="帮我分割附件中的股骨头和髋臼")
                    ]
                )
            )
        )

    args = {"file_id": file_id, "model_id": "fixture-t1", "task_type": "femoral_head_segmentation"}
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
    answer = SimpleNamespace(stream=AsyncMock(side_effect=AssertionError("No medical prose for a receipt")))
    tools = ToolGateway(ToolRegistry(SimpleNamespace(tool_config=Path("configs/tools.yaml"))))
    tools.invoke = AsyncMock(
        return_value={
            "job_id": "job_fixture",
            "status": status,
            "requires_confirmation": status == "draft",
            "model_id": "fixture-t1",
            "labels": {"1": "femoralhead", "2": "acetabulum"},
            "result_asset_id": "asset_mask" if status == "succeeded" else None,
        }
    )
    svc = SimpleNamespace(
        graph_policy=config_load(Path("configs/graph.yaml")),
        sessions=sessions,
        tools=tools,
        models=SimpleNamespace(get=lambda role, snapshot: router if role == "agent_router" else answer),
        imaging=SimpleNamespace(
            metadata=lambda asset: {
                "asset_id": file_id,
                "kind": "volume",
                "dimensions": [8, 8, 8],
                "modality": "T1",
                "channels": 1,
                "status": "ready",
            }
        ),
        segmentation=SimpleNamespace(public_config=lambda: {"models": []}),
    )
    snapshot = {
        "asset_ids": [file_id],
        "segmentation_model_id": "fixture-t1",
        "decoding": {
            "history_message_limit": 20,
            "prompt_byte_budget": 12000,
            "router_max_tokens": 384,
            "answer_temperature": 0.2,
            "answer_max_tokens": 768,
        },
    }
    ctx = RunContext(Actor("u", "t", "member"), "req", "run", "conv", snapshot, svc, AsyncMock())
    state = await build_graph(svc, InMemorySaver()).ainvoke(
        {"status": "running"}, {"configurable": {"thread_id": "fixture-seg-receipt"}}, context=ctx
    )
    tools.invoke.assert_awaited_once()
    answer.stream.assert_not_called()
    text = state["final_answer"]
    assert "fixture-t1" not in text and "模型" not in text and "医生" not in text
    if status == "draft":
        assert "确认并开始分割" in text and "分割已完成" not in text
    elif status == "queued":
        assert "已排队" in text and "分割已完成" not in text
    else:
        assert "分割已完成" in text


def test_answer_evidence_keeps_measurements_but_omits_internal_model_identifiers():
    from pharma_agent.graph import answer_evidence, segmentation_receipt

    evidence = [
        {
            "tool": "get_my_job",
            "output": {
                "jobs": [
                    {
                        "input": {"model": {"id": "private-registry-id", "name": "internal-model-name"}},
                        "model_id": "internal-model-name",
                        "result": {
                            "model_version": "private-weight-version",
                            "labels": {"1": "femoralhead"},
                            "measurements": [{"volume_mm3": 123.0}],
                        },
                    }
                ]
            },
        }
    ]
    cleaned = answer_evidence(evidence)
    assert "internal-model-name" not in json.dumps(cleaned)
    assert "private-weight-version" not in json.dumps(cleaned)
    assert "private-registry-id" not in json.dumps(cleaned)
    assert cleaned[0]["output"]["jobs"][0]["result"]["measurements"][0]["volume_mm3"] == 123.0
    assert "internal-model-name" in json.dumps(evidence)  # The durable source is untouched.
    assert (
        segmentation_receipt([{"tool": "segment_medical_volume", "output": {"status": "succeeded"}}]) is None
    )


def test_surface_uses_actual_labels_affine_and_face_budget():
    data = np.zeros((12, 14, 10), dtype=np.uint8)
    data[3:8, 3:10, 2:7] = 1
    affine = np.array([[-2, 0, 0, 40], [0, 3, 0, -10], [0, 0, 4, 5], [0, 0, 0, 1]], float)
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    result = mask_surface(image, {"0": "background", "1": "actual_femoralhead", "2": "actual_acetabulum"})
    assert result["segments"][0]["name"] == "actual_femoralhead"
    assert len(result["segments"]) == 1
    vertices = np.asarray(result["segments"][0]["quads"]).reshape(-1, 3)
    assert np.allclose(vertices.min(0), [25, -2.5, 11])
    assert np.allclose(vertices.max(0), [35, 18.5, 31])
    bounded = mask_surface(image, {"0": "background", "1": "actual"}, face_budget=24)
    assert bounded["face_count"] <= 24 and bounded["stride"] > 1
    assert result["space"] == "RAS" and result["units"] == "mm"


def test_checkpoint_metadata_reader_rejects_executable_globals():
    class Malicious:
        def __reduce__(self):
            return (eval, ("1+1",))

    with pytest.raises(ValueError):
        MetadataReader(io.BytesIO(pickle.dumps(Malicious()))).load()


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (200, {"status": "ready", "model_id": "fixture", "model_version": "v1"}, "ready"),
        (200, {"status": "ready", "model_id": "fixture", "model_version": "wrong"}, "incompatible"),
        (503, {}, "unavailable"),
    ],
)
async def test_segmentation_probe_reports_real_service_state(tmp_path, status, body, expected):
    from pharma_agent.segmentation import SegmentationGateway

    config = tmp_path / "segmentation.yaml"
    config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "fixture",
                        "name": "fixture",
                        "version": "v1",
                        "task_types": ["segment"],
                        "modalities": ["T1"],
                        "channels": 1,
                        "labels": {"0": "background", "1": "label"},
                    }
                ]
            }
        )
    )
    settings = Settings(
        _env_file=None,
        jwt_secret="fixture-probe-jwt-" * 3,
        segmentation_base_url="http://synthetic-service",
        segmentation_models_config=config,
    )
    g = SegmentationGateway(settings)
    assert await g.probe(httpx.MockTransport(lambda request: httpx.Response(status, json=body))) == expected


def test_noisy_surface_is_bounded_before_vertex_allocation():
    data = (np.indices((40, 40, 20)).sum(0) % 2).astype(np.uint8)
    image = nib.Nifti1Image(data, np.diag([0.8, 0.8, 5, 1]))
    result = mask_surface(image, {0: "background", 1: "fixture"}, face_budget=100)
    assert result["face_count"] <= 100
    assert sum(len(row["quads"]) // 12 for row in result["segments"]) == result["face_count"]
    assert result["sampling"][2] <= max(result["sampling"][:2])


def test_tool_cannot_override_explicit_segmentation_selection():
    from pharma_agent.tools import ToolContext

    tools = ToolGateway(ToolRegistry(SimpleNamespace(tool_config=Path("configs/tools.yaml"))))
    context = ToolContext(
        Actor("fixture-user", "fixture-tenant", "member"),
        "req",
        "run",
        None,
        segmentation_model_id="explicit-choice",
    )
    args = {"file_id": "asset_" + "a" * 32, "model_id": "other-choice", "task_type": "segmentation"}
    with pytest.raises(AppError) as failure:
        tools.validate("segment_medical_volume", args, context)
    assert failure.value.code == "MODEL_SELECTION_MISMATCH"
