"""Specific failure categories, using synthetic transports only."""

from unittest.mock import AsyncMock

import httpx
import pytest

from pharma_agent.errors import AppError
from pharma_agent.segmentation import SegmentationGateway
from pharma_agent.settings import Settings


@pytest.mark.parametrize(
    "failure,code",
    [
        ("connect", "SEGMENTATION_CONNECTION_FAILED"),
        ("timeout", "SEGMENTATION_CONNECTION_TIMEOUT"),
        ("auth", "SEGMENTATION_AUTH_FAILED"),
        ("endpoint", "SEGMENTATION_ENDPOINT_NOT_FOUND"),
        ("json", "SEGMENTATION_PROTOCOL_ERROR"),
        ("queue", "SEGMENTATION_BUSY"),
        ("loading", "SEGMENTATION_LOADING"),
        ("model", "SEGMENTATION_MODEL_NOT_READY"),
    ],
)
async def test_connection_and_contract_failures_are_distinct_without_upstream_echo(tmp_path, failure, code):
    settings = Settings(
        _env_file=None,
        jwt_secret="diagnostic-fixture-" * 3,
        segmentation_base_url="http://fixture",
        segmentation_api_key="fixture-secret-" * 3,
    )
    gateway = SegmentationGateway(settings)
    path = tmp_path / "volume.nii.gz"
    path.write_bytes(b"synthetic")

    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("private-upstream-secret", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private-upstream-secret", request=request)
        return {
            "auth": httpx.Response(401, text="private-upstream-secret"),
            "endpoint": httpx.Response(404, text="private-upstream-secret"),
            "json": httpx.Response(200, text="private-upstream-secret"),
            "queue": httpx.Response(429, text="private-upstream-secret"),
            "loading": httpx.Response(503, json={"status": "loading"}),
            "model": httpx.Response(503, json={"status": "failed", "error_code": "MODEL_LOAD_FAILED"}),
        }[failure]

    transport = httpx.MockTransport(handler)
    detail = await gateway.probe_detail(transport)
    assert detail["code"] == code and "private-upstream-secret" not in detail["message"]
    with pytest.raises(AppError) as caught:
        await gateway.infer(
            path,
            next(iter(gateway.models.values())),
            "femoral_head_segmentation",
            "fixture",
            AsyncMock(),
            transport,
        )
    assert caught.value.code == code and "private-upstream-secret" not in caught.value.message


async def test_readiness_validates_identity_version_and_protocol():
    gateway = SegmentationGateway(
        Settings(_env_file=None, jwt_secret="readiness-fixture-" * 3, segmentation_base_url="http://fixture")
    )
    model = next(iter(gateway.models.values()))

    async def probe(payload):
        return await gateway.probe_detail(httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))

    assert (await probe({"status": "ready", "model_id": model.id, "model_version": model.version}))[
        "status"
    ] == "ready"
    assert (await probe({"status": "ready", "model_id": model.id, "model_version": "other"}))[
        "code"
    ] == "MODEL_VERSION_MISMATCH"
    assert (await probe({"status": "ok"}))["code"] == "SEGMENTATION_PROTOCOL_ERROR"
    assert (
        await probe(
            {"status": "ready", "model_id": model.id, "model_version": model.version, "api_version": 99}
        )
    )["code"] == "SEGMENTATION_PROTOCOL_ERROR"
