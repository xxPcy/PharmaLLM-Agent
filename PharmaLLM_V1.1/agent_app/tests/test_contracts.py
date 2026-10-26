import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from pharma_agent.errors import AppError
from pharma_agent.models import (
    GenerationRequest,
    ModelGateway,
    OpenAICompatibleAdapter,
    bounded_messages,
    config_load,
)
from pharma_agent.schemas import MemoryInput, RouteDecision
from pharma_agent.security import Actor
from pharma_agent.settings import Settings
from pharma_agent.storage import LocalStorage, detect_upload
from pharma_agent.tools import ToolContext, ToolGateway, ToolRegistry


def test_auth_validation_identifies_fields_without_echoing_credentials():
    from fastapi.testclient import TestClient
    from pharma_agent.main import create_app, validation_message

    settings = Settings(_env_file=None, jwt_secret="isolated-validation-secret-" * 3,
                        app_env="test", model_mode="fake", enable_queue=False)
    # No lifespan: the request fails schema validation before any DB or service access.
    client = TestClient(create_app(settings))
    response = client.post("/api/v1/auth/register", json={"tenant_code": "20_26",
                           "email": "synthetic@example.org", "password": "hid1"})
    assert response.status_code == 422
    message = response.json()["error"]["message"]
    assert "团队登录代号" in message and "密码" in message and "12" in message
    for secret in ("hid1", "synthetic@example.org", "20_26", '"input"', '"ctx"'):
        assert secret not in response.text
    errors = [{"loc": ("body", "password"), "msg": "must-not-echo", "input": "private-value",
               "ctx": {"error": "private-value"}}]
    assert "private-value" not in validation_message("/api/v1/auth/login", errors)
    assert "密码" in validation_message("/api/v1/auth/login", errors)
    assert validation_message("/api/v1/jobs/drafts", errors) == "请求参数未通过校验，请检查接口格式"
    client.close()


def test_identity_cannot_enter_tool_arguments():
    registry = ToolRegistry(SimpleNamespace(tool_config=__import__("pathlib").Path("configs/tools.yaml")))
    gateway = ToolGateway(registry)
    context = ToolContext(Actor("real-user", "real-tenant", "member"), "req", "run", None)
    with pytest.raises(AppError) as failure:
        gateway.validate("get_my_job", {"job_id": "job_" + "a" * 32, "user_id": "victim"}, context)
    assert failure.value.code == "INVALID_TOOL_ARGS"
    with pytest.raises(AppError):
        gateway.validate("execute_shell", {"command": "anything"}, context)


def test_route_and_memory_whitelists():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate({"action": "tool", "tool_name": "arbitrary_sql"})
    with pytest.raises(ValidationError):
        RouteDecision.model_validate({"action": "respond", "arguments": {"job_id": "x"}})
    with pytest.raises(ValidationError):
        MemoryInput.model_validate({"kind": "diagnosis", "value": "private", "consent": True})
    with pytest.raises(ValidationError):
        MemoryInput.model_validate({"kind": "language_preference", "value": "zh", "consent": False})


def test_context_window_preserves_system_and_latest_turn():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "old" * 300},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
    ]
    window = bounded_messages(messages, 400)
    assert window == [messages[0], messages[-1]]
    with pytest.raises(AppError) as failure:
        bounded_messages([messages[0], {"role": "user", "content": "长" * 1000}], 400)
    assert failure.value.code == "MODEL_CONTEXT_OVERFLOW"


def test_yaml_substitution_cannot_change_yaml_structure(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("key: ${SECRET}\n")
    assert config_load(config, {"SECRET": "a: b\nother: injected"}) == {"key": "a: b\nother: injected"}
    with pytest.raises(ValueError):
        config_load(config, {})


def test_storage_rejects_traversal_and_unsafe_types(tmp_path):
    storage = LocalStorage(tmp_path)
    with pytest.raises(AppError):
        storage.path("../secret")
    with pytest.raises(AppError):
        detect_upload(b"<script>anything</script>", "text/html")
    with pytest.raises(AppError):
        detect_upload(b"not a png", "image/png")


@pytest.mark.parametrize(
    "status,code", [(503, "MODEL_UNAVAILABLE"), (429, "MODEL_RATE_LIMIT"), (400, "MODEL_BAD_RESPONSE")]
)
async def test_model_errors_do_not_expose_upstream_content(status, code):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"secret": "DO_NOT_LEAK"}))
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = OpenAICompatibleAdapter(
            "mock",
            {"base_url": "http://model/v1", "model_name": "fixture", "timeout_seconds": 1, "max_retries": 0},
            client,
        )
        with pytest.raises(AppError) as failure:
            await adapter.generate(GenerationRequest([{"role": "user", "content": "private"}]))
        assert failure.value.code == code
        assert "DO_NOT_LEAK" not in failure.value.message


async def test_truncated_stream_is_a_failure():
    body = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            "mock", {"base_url": "http://model/v1", "model_name": "fixture", "timeout_seconds": 1}, client
        )
        with pytest.raises(AppError) as failure:
            _ = [
                piece async for piece in adapter.stream(GenerationRequest([{"role": "user", "content": "x"}]))
            ]
        assert failure.value.code == "MODEL_BAD_RESPONSE"


async def test_model_stream_usage_is_preserved():
    events = [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 1, "prompt_tokens": 4}},
    ]
    body = "".join("data: " + json.dumps(row) + "\n\n" for row in events) + "data: [DONE]\n\n"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            "mock", {"base_url": "http://model/v1", "model_name": "fixture", "timeout_seconds": 1}, client
        )
        pieces = [
            piece async for piece in adapter.stream(GenerationRequest([{"role": "user", "content": "x"}]))
        ]
        assert "".join(piece.text for piece in pieces) == "ok"
        assert pieces[-1].usage["completion_tokens"] == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_local_model_answer_sends_chatml_stop_and_repeat_controls(streaming):
    settings = Settings(
        _env_file=None,
        jwt_secret="synthetic-generation-test-" * 3,
        model_mode="openai",
        router_model_alias="pharma_llm_sft",
    )

    def handler(request):
        payload = json.loads(request.content)
        # SFT ChatML terminates with im_end, while the exported generation config
        # only lists endoftext. Both must reach vLLM on actual HTTP requests.
        assert payload["stop_token_ids"] == [151643, 151645]
        assert payload["repetition_penalty"] == 1.1
        assert payload["stream"] is streaming
        assert "response_format" not in payload
        if streaming:
            event = {"choices": [{"delta": {"content": "synthetic answer"}, "finish_reason": "stop"}]}
            return httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "synthetic answer"}, "finish_reason": "stop"}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        gateway = ModelGateway(settings, client)
        snapshot = gateway.snapshot()
        # Queued runs retain their own generation settings after registry edits.
        gateway.registry["models"]["pharma_llm_sft"]["answer_repetition_penalty"] = 1.2
        adapter = gateway.get("medical_answer", snapshot)
        request = GenerationRequest([{"role": "user", "content": "synthetic question"}])
        if streaming:
            pieces = [piece async for piece in adapter.stream(request)]
            assert "".join(piece.text for piece in pieces) == "synthetic answer"
        else:
            result = await adapter.generate(request)
            assert result.content == "synthetic answer"


async def test_generic_endpoint_does_not_receive_unconfigured_vllm_extensions():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        adapter = OpenAICompatibleAdapter("generic", {"model_name": "generic-model"}, client)
        payload = adapter.payload(GenerationRequest([{"role": "user", "content": "synthetic"}]))
    assert "stop_token_ids" not in payload
    assert "repetition_penalty" not in payload
