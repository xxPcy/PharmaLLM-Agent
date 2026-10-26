import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import HumanMessage

from pharma_agent.errors import AppError
from pharma_agent.graph import RunContext, route_decision
from pharma_agent.models import GenerationRequest, ModelGateway
from pharma_agent.security import Actor
from pharma_agent.settings import Settings
from pharma_agent.tools import ToolGateway, ToolRegistry


@pytest.fixture
def route_setup():
    settings = Settings(
        _env_file=None,
        jwt_secret="synthetic-router-test-" * 3,
        model_mode="openai",
        router_model_alias="pharma_llm_sft",
    )
    tools = ToolGateway(ToolRegistry(SimpleNamespace(tool_config=Path("configs/tools.yaml"))))
    tools.invoke = AsyncMock(side_effect=AssertionError("Routing must not execute tools"))
    state = {
        "messages": [HumanMessage(content="糖原分子中一个葡萄糖单位经糖酵解途径分解成乳酸时能产生几分子ATP")],
        "tool_observations": [],
    }
    return settings, tools, state


async def route_with_responses(route_setup, responses):
    settings, tools, state = route_setup
    requests = []
    pending = iter(responses)

    def handler(request):
        requests.append(json.loads(request.content))
        result = next(pending)
        if isinstance(result, int):
            return httpx.Response(result, json={"error": {"code": "synthetic"}})
        content, finish_reason = result
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                "usage": {"completion_tokens": 384 if finish_reason == "length" else 20},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        models = ModelGateway(settings, client)
        ctx = RunContext(
            Actor("synthetic-user", "synthetic-tenant", "member"),
            "synthetic-request",
            "synthetic-run",
            "synthetic-conversation",
            models.snapshot(),
            SimpleNamespace(models=models, tools=tools),
            AsyncMock(),
        )
        result = await route_decision(state, ctx, {"max_route_retries": 1})
    tools.invoke.assert_not_awaited()
    return result, requests, ctx


async def test_medical_question_requests_constrained_route_then_plain_answer(route_setup):
    result, requests, ctx = await route_with_responses(route_setup, [('{"action":"respond"}', "stop")])
    assert result["decision"]["action"] == "respond"
    payload = requests[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["stop_token_ids"] == [151643, 151645]
    assert "repetition_penalty" not in payload
    assert payload["temperature"] == 0
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["action"]["enum"]) == {"respond", "retrieve", "tool", "clarify"}
    assert ctx.model_usage[0]["role"] == "agent_router"
    settings, _, _ = route_setup
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        answer = ModelGateway(settings, client).get("medical_answer")
        answer_payload = answer.payload(GenerationRequest(payload["messages"]), streaming=True)
    assert "response_format" not in answer_payload
    assert answer_payload["stream"] is True
    assert answer_payload["stop_token_ids"] == [151643, 151645]
    assert answer_payload["repetition_penalty"] == 1.1


async def test_invalid_route_retry_has_feedback_without_echoing_raw_output(route_setup, caplog):
    caplog.set_level(logging.INFO, logger="pharma_agent")
    secret_output = "PRIVATE_INVALID_RESPONSE_DO_NOT_ECHO"
    result, requests, _ = await route_with_responses(
        route_setup, [(secret_output, "stop"), ('{"action":"respond"}', "stop")]
    )
    assert result["decision"]["action"] == "respond"
    assert requests[0]["messages"] != requests[1]["messages"]
    assert "json_invalid" in requests[1]["messages"][0]["content"]
    assert secret_output not in json.dumps(requests[1], ensure_ascii=False)
    assert secret_output not in caplog.text
    assert '"validation_codes": ["json_invalid"]' in caplog.text


async def test_even_parseable_truncated_route_is_retried(route_setup):
    result, requests, _ = await route_with_responses(
        route_setup, [('{"action":"respond"}', "length"), ('{"action":"respond"}', "stop")]
    )
    assert result["decision"]["action"] == "respond"
    assert len(requests) == 2
    assert "OUTPUT_TOKEN_LIMIT" in requests[1]["messages"][0]["content"]


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        '{"action":"tool","tool_name":"execute_shell","arguments":{}}',
        '{"action":"tool","tool_name":"get_my_job","arguments":{"user_id":"victim"}}',
        '{"action":"respond","arguments":{"job_id":"forged"}}',
    ],
)
async def test_invalid_decisions_stay_bounded_and_never_execute_tools(route_setup, content):
    result, requests, _ = await route_with_responses(route_setup, [(content, "stop")] * 2)
    assert len(requests) == 2
    assert result == {"decision": {"action": "clarify"}, "error_code": "ROUTE_INVALID"}


async def test_valid_tool_route_is_validated_without_executing_it(route_setup):
    result, _, _ = await route_with_responses(
        route_setup, [('{"action":"tool","tool_name":"get_my_job","arguments":{}}', "stop")]
    )
    assert result["decision"]["tool_name"] == "get_my_job"


async def test_endpoint_error_does_not_become_a_successful_route(route_setup):
    with pytest.raises(AppError) as failure:
        await route_with_responses(route_setup, [400])
    assert failure.value.code == "MODEL_BAD_RESPONSE"
