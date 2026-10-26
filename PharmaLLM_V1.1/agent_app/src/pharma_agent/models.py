import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

import httpx
import yaml
from opentelemetry.propagate import inject

from pharma_agent.errors import AppError
from pharma_agent.observability import digest, log, span


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict[str, Any]]
    temperature: float = 0.2
    max_tokens: int = 768
    response_schema: dict | None = None
    purpose: str = "answer"


@dataclass
class GenerationResult:
    content: str
    model_id: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


@dataclass
class StreamPiece:
    text: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


class ChatModelAdapter(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamPiece]: ...


def safe_usage(value):
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if type(value.get(key)) is int and value[key] >= 0
    }


def config_load(path, environment: dict[str, str] | None = None) -> dict:
    raw = path.read_text(encoding="utf-8")
    # Substitute parsed scalar values, not raw YAML (secrets may contain YAML syntax).
    data = yaml.safe_load(raw)

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):

            def replacement(match):
                if environment is None or match[1] not in environment:
                    raise ValueError(f"配置变量未定义：{match[1]}")
                return environment[match[1]]

            return re.sub(r"\$\{([A-Z0-9_]+)\}", replacement, value)
        return value

    return expand(data)


def bounded_messages(messages: list[dict], byte_budget: int) -> list[dict]:
    """Conservative byte window; preserve system policy and the newest user turn."""
    system = [m for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]

    def cost(message):
        return len(str(message["content"]).encode("utf-8")) + 64

    used = sum(cost(m) for m in system)
    selected = []
    for message in reversed(turns):
        size = cost(message)
        if used + size > byte_budget:
            if not selected:
                raise AppError("MODEL_CONTEXT_OVERFLOW", "本轮内容过长，请缩短问题或减少资料", 422)
            break
        selected.append(message)
        used += size
    # A window must not begin with an orphaned assistant turn.
    selected.reverse()
    while selected and selected[0]["role"] != "user":
        selected.pop(0)
    if not selected or used > byte_budget:
        raise AppError("MODEL_CONTEXT_OVERFLOW", "上下文预算不足，请缩短问题", 422)
    return system + selected


class OpenAICompatibleAdapter:
    def __init__(self, alias: str, config: dict, client: httpx.AsyncClient):
        self.alias, self.config, self.client = alias, config, client

    def headers(self):
        headers = {"Content-Type": "application/json"}
        if self.config.get("api_key"):
            headers["Authorization"] = "Bearer " + self.config["api_key"]
        inject(headers)
        return headers

    def payload(self, request: GenerationRequest, streaming=False):
        data = {
            "model": self.config["model_name"],
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": streaming,
        }
        if self.config.get("chat_template_kwargs"):
            data["chat_template_kwargs"] = self.config["chat_template_kwargs"]
        # Explicit endpoint settings, pinned in the run snapshot. Do not assume
        # that every OpenAI-compatible service accepts these vLLM extensions.
        if self.config.get("stop_token_ids"):
            data["stop_token_ids"] = self.config["stop_token_ids"]
        if request.purpose == "answer" and "answer_repetition_penalty" in self.config:
            data["repetition_penalty"] = self.config["answer_repetition_penalty"]
        if streaming:
            data["stream_options"] = {"include_usage": True}
        # This endpoint capability enables server-side constrained decoding. It does
        # not imply trained tool selection; application argument checks still apply.
        if request.response_schema and "structured_output" in self.config.get("capabilities", []):
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "route_decision", "strict": True, "schema": request.response_schema},
            }
        return data

    def raise_status(self, response):
        if response.status_code < 400:
            return
        if response.status_code == 429:
            raise AppError("MODEL_RATE_LIMIT", "模型服务繁忙，请稍后重试", 503, True)
        if response.status_code == 400:
            # Inspect only a machine-readable error code; never log upstream text.
            try:
                code = response.json().get("error", {}).get("code", "")
            except (ValueError, AttributeError):
                code = ""
            if code in {"context_length_exceeded", "max_context_length_exceeded"}:
                raise AppError("MODEL_CONTEXT_OVERFLOW", "模型上下文长度超限", 422)
            raise AppError("MODEL_BAD_RESPONSE", "模型拒绝了请求，请检查模型配置", 502)
        raise AppError("MODEL_UNAVAILABLE", "模型服务暂不可用", 503, True)

    async def generate(self, request):
        with span(
            "model.generate",
            model_id=self.alias,
            model_revision=self.config.get("revision", "unknown"),
            purpose=request.purpose,
            input_hash=digest(request.messages),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            for attempt in range(self.config.get("max_retries", 0) + 1):
                try:
                    response = await self.client.post(
                        self.config["base_url"].rstrip("/") + "/chat/completions",
                        headers=self.headers(),
                        json=self.payload(request),
                        timeout=self.config["timeout_seconds"],
                    )
                    self.raise_status(response)
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("empty content")
                    usage = safe_usage(body.get("usage"))
                    log("model.result", model_id=self.alias, response_hash=digest(content), usage=usage)
                    return GenerationResult(content, self.alias, usage, choice.get("finish_reason"))
                except httpx.TimeoutException:
                    error = AppError("MODEL_TIMEOUT", "模型响应超时，请稍后重试", 504, True)
                except httpx.RequestError:
                    error = AppError("MODEL_UNAVAILABLE", "无法连接模型服务", 503, True)
                except (ValueError, KeyError, IndexError, TypeError):
                    raise AppError("MODEL_BAD_RESPONSE", "模型返回格式不符合约定", 502) from None
                except AppError as exc:
                    error = exc
                if not error.retryable or attempt == self.config.get("max_retries", 0):
                    raise error
                await asyncio.sleep(0.25 * (attempt + 1))

    async def stream(self, request):
        started, first = time.monotonic(), True
        with span(
            "model.stream",
            model_id=self.alias,
            model_revision=self.config.get("revision", "unknown"),
            input_hash=digest(request.messages),
            max_tokens=request.max_tokens,
        ):
            # Never restart a stream after emitting tokens: that would duplicate the answer.
            try:
                async with self.client.stream(
                    "POST",
                    self.config["base_url"].rstrip("/") + "/chat/completions",
                    headers=self.headers(),
                    json=self.payload(request, True),
                    timeout=self.config["timeout_seconds"],
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        self.raise_status(response)
                    done = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            done = True
                            break
                        item = json.loads(raw)
                        if item.get("error"):
                            raise AppError("MODEL_BAD_RESPONSE", "模型流异常终止", 502, True)
                        choices = item.get("choices", [])
                        text = choices[0].get("delta", {}).get("content") or "" if choices else ""
                        if not isinstance(text, str):
                            raise ValueError("non-text content")
                        if text and first:
                            log(
                                "model.first_token",
                                model_id=self.alias,
                                ttft_ms=round((time.monotonic() - started) * 1000),
                            )
                            first = False
                        yield StreamPiece(
                            text,
                            safe_usage(item.get("usage")),
                            choices[0].get("finish_reason") if choices else None,
                        )
                    if not done:
                        raise AppError("MODEL_BAD_RESPONSE", "模型流未完整结束", 502, True)
            except httpx.TimeoutException:
                raise AppError("MODEL_TIMEOUT", "模型流响应超时", 504, True) from None
            except httpx.RequestError:
                raise AppError("MODEL_UNAVAILABLE", "模型连接中断", 503, True) from None
            except (ValueError, KeyError, IndexError, TypeError):
                raise AppError("MODEL_BAD_RESPONSE", "模型流格式异常", 502) from None


class FakeModelAdapter:
    """Deterministic CPU adapter: engineering fixtures, never a medical model."""

    def __init__(self, faults=False):
        self.faults = faults

    async def generate(self, request):
        last = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
        if self.faults and "[[fault:model_503]]" in last:
            raise AppError("MODEL_UNAVAILABLE", "模拟模型 503", 503, True)
        if self.faults and "[[fault:model_timeout]]" in last:
            raise AppError("MODEL_TIMEOUT", "模拟模型超时", 504, True)
        if self.faults and "[[fault:slow]]" in last:
            await asyncio.sleep(3)
        if request.purpose == "route":
            if self.faults and "[[fault:bad_route]]" in last:
                return GenerationResult("not-json", "fake")
            decision = {"action": "respond"}
            if self.faults and "[[fault:loop]]" in last:
                return GenerationResult(
                    json.dumps(
                        {"action": "tool", "tool_name": "get_my_job", "arguments": {}, "continue": True}
                    ),
                    "fake",
                )
            job = re.search(r"job_[a-f0-9]{32}", last)
            if "任务" in last and any(word in last for word in ("创建", "新建", "草稿")):
                decision = {
                    "action": "tool",
                    "tool_name": "create_task_draft",
                    "arguments": {"title": "合成演示任务"},
                    "reason_code": "DRAFT",
                }
            elif job or any(word in last for word in ("任务结果", "任务状态", "我的任务")):
                decision = {
                    "action": "tool",
                    "tool_name": "get_my_job",
                    "arguments": {"job_id": job[0]} if job else {},
                    "reason_code": "JOB_STATUS",
                }
            elif any(word in last for word in ("检索", "参考", "资料", "[[fault:tool_timeout]]")):
                decision = {"action": "retrieve", "arguments": {"query": last[:1000]}}
            elif "偏好" in last:
                decision = {"action": "tool", "tool_name": "get_user_preference", "arguments": {}}
            return GenerationResult(json.dumps(decision, ensure_ascii=False), "fake")
        return GenerationResult("这是 FakeModel 工程演示响应。请切换真实 PharmaLLM 后验证医学回答。", "fake")

    async def stream(self, request):
        result = await self.generate(request)
        # Include bounded observations so the full tool loop is inspectable in CPU mode.
        evidence = next(
            (
                m["content"].split("UNTRUSTED_EVIDENCE", 1)[1]
                for m in reversed(request.messages)
                if m["role"] == "user" and "UNTRUSTED_EVIDENCE" in m["content"]
            ),
            "",
        )
        answer = result.content + ("\n工具/检索结果：" + evidence[20:1400] if evidence else "")
        for start in range(0, len(answer), 32):
            await asyncio.sleep(0)
            yield StreamPiece(answer[start : start + 32])
        yield StreamPiece(finish_reason="stop")


class ModelGateway:
    def __init__(self, settings, client: httpx.AsyncClient):
        self.settings, self.client = settings, client
        self.registry = config_load(settings.model_config_path, settings.config_environment())
        if settings.model_mode == "fake":
            self.registry["roles"] = {k: "fake" for k in self.registry["roles"]}
        for role in ("medical_answer", "general_chat", "agent_router"):
            alias = self.registry["roles"][role]
            if alias not in self.registry["models"]:
                raise ValueError(f"角色 {role} 引用了不存在的模型")
        for config in self.registry["models"].values():
            if config["provider"] not in {"fake", "openai_compatible"}:
                raise ValueError("未知模型 provider")

    def chat_choices(self):
        aliases = (
            [self.registry["roles"]["medical_answer"]]
            if self.settings.model_mode == "fake"
            else self.registry.get("selectable_chat_models", [self.registry["roles"]["medical_answer"]])
        )
        if not isinstance(aliases, list) or not aliases:
            raise ValueError("selectable_chat_models 必须是非空模型别名列表")
        result = []
        for alias in dict.fromkeys(aliases):
            config = self.registry["models"].get(alias)
            if not config or "chat" not in config.get("capabilities", []):
                raise ValueError("可选对话模型不在已配置的聊天模型中")
            result.append(
                {
                    "alias": alias,
                    "name": config.get("display_name", config.get("model_name", alias)),
                    "model_id": config.get("model_name", alias),
                }
            )
        return result

    def snapshot(self, model_alias: str | None = None) -> dict:
        # API keys never enter a business row, graph state, trace or exported fixture.
        roles = self.registry["roles"].copy()
        if model_alias is not None:
            if model_alias not in {row["alias"] for row in self.chat_choices()}:
                raise AppError("MODEL_NOT_ALLOWED", "对话模型不在服务端允许列表中", 422)
            roles["medical_answer"] = model_alias
        aliases = set(roles.values())
        return {
            "config_version": self.registry["version"],
            "roles": roles,
            "decoding": {
                "answer_temperature": 0.2,
                "answer_max_tokens": self.settings.answer_max_tokens,
                "router_max_tokens": 384,
                "prompt_byte_budget": self.settings.prompt_byte_budget,
                "history_message_limit": self.settings.history_message_limit,
            },
            "models": {
                alias: {k: v for k, v in self.registry["models"][alias].items() if k != "api_key"}
                for alias in aliases
            },
        }

    def get(self, role: str, snapshot: dict | None = None) -> ChatModelAdapter:
        source = snapshot or self.registry
        alias = source["roles"][role]
        config = dict(source["models"][alias])
        if config["provider"] == "fake":
            return FakeModelAdapter(self.settings.enable_faults)
        # Fetch credentials from current deployment only, while pinning endpoint/model/revision to the run.
        current = self.registry["models"].get(alias, {})
        config["api_key"] = current.get("api_key", "")
        config["timeout_seconds"] = min(config["timeout_seconds"], self.settings.model_timeout_seconds)
        return OpenAICompatibleAdapter(alias, config, self.client)

    async def health(self):
        for alias in set(self.registry["roles"].values()):
            config = self.registry["models"][alias]
            if config["provider"] == "fake":
                continue
            adapter = OpenAICompatibleAdapter(alias, config, self.client)
            try:
                response = await self.client.get(
                    config["base_url"].rstrip("/") + "/models", headers=adapter.headers(), timeout=5
                )
                adapter.raise_status(response)
                if config["model_name"] not in {item["id"] for item in response.json()["data"]}:
                    raise AppError("MODEL_UNAVAILABLE", "配置的模型名称未在模型服务中注册", 503, True)
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise AppError("MODEL_UNAVAILABLE", "模型健康检查失败", 503, True) from None
