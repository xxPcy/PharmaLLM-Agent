"""Optional P0 model connectivity app: no business DB, no Redis, no graph."""

from contextlib import asynccontextmanager
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from pharma_agent.db import new_id
from pharma_agent.errors import AppError
from pharma_agent.main import error_response
from pharma_agent.models import GenerationRequest, ModelGateway
from pharma_agent.observability import configure, log, request_context, span, trace_id
from pharma_agent.schemas import DemoChatInput
from pharma_agent.settings import Settings


def create_app():
    settings = Settings()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if settings.app_env == "production":
        raise ValueError("P0 匿名连通性接口仅用于本地开发")

    @asynccontextmanager
    async def lifespan(app):
        configure(settings)
        async with httpx.AsyncClient(trust_env=False) as client:
            app.state.models = ModelGateway(settings, client)
            yield

    app = FastAPI(title="PharmaLLM P0 local connectivity", lifespan=lifespan)

    @app.middleware("http")
    async def context(request, call_next):
        request.state.request_id = new_id("req")
        token = request_context.set({"request_id": request.state.request_id})
        try:
            with span("p0.request"):
                request.state.trace_id = trace_id()
                try:
                    response = await call_next(request)
                except Exception as exc:
                    log("p0.error", error_code="INTERNAL_ERROR", exception_type=type(exc).__name__)
                    response = error_response(request, AppError("INTERNAL_ERROR", "模型连通性检查失败", 500))
                response.headers["X-Request-ID"] = request.state.request_id
                response.headers["X-Trace-ID"] = request.state.trace_id
                return response
        finally:
            request_context.reset(token)

    @app.exception_handler(AppError)
    async def error(request, exc):
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return error_response(request, AppError("INVALID_REQUEST", "请求格式不正确", 422))

    @app.get("/health/live")
    async def live():
        return {"status": "ok", "phase": "P0"}

    @app.get("/health/ready")
    async def ready():
        await app.state.models.health()
        return {"status": "ready"}

    @app.post("/api/v1/demo/chat")
    async def chat(data: DemoChatInput, request: Request):
        result = await app.state.models.get("medical_answer").generate(
            GenerationRequest(
                [
                    {"role": "system", "content": "你是医学信息工程演示助手，不替代临床诊疗。"},
                    {"role": "user", "content": data.message},
                ],
                max_tokens=256,
            )
        )
        return {
            "answer": result.content,
            "model_id": result.model_id,
            "usage": result.usage,
            "trace_id": request.state.trace_id,
            "request_id": request.state.request_id,
        }

    return app
