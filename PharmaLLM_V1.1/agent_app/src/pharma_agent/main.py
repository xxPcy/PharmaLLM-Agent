import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis
from sqlalchemy import text
from starlette.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from pharma_agent.api import router
from pharma_agent.hospital import router as hospital_router
from pharma_agent.db import create_database, new_id
from pharma_agent.errors import AppError
from pharma_agent.graph import build_graph
from pharma_agent.jobs import QueueService
from pharma_agent.imaging import ImagingStore
from pharma_agent.segmentation import SegmentationGateway
from pharma_agent.imaging_api import router as imaging_router
from pharma_agent.models import ModelGateway, config_load
from pharma_agent.observability import configure, log, request_context, span, trace_id
from pharma_agent.retrieval import EmbeddingGateway, RetrievalService
from pharma_agent.runs import RunService
from pharma_agent.settings import Settings
from pharma_agent.schemas import TEAM_CODE_HELP
from pharma_agent.storage import LocalStorage
from pharma_agent.tools import ToolGateway, ToolRegistry


def validation_message(path, errors):
    """Only static field guidance, never rejected inputs, raw messages or validator context."""
    if path not in {"/api/v1/auth/register", "/api/v1/auth/login", "/api/v1/organization",
                    "/api/v1/organization/members"}:
        return "请求参数未通过校验，请检查接口格式"
    hints = {
        "tenant_code": TEAM_CODE_HELP,
        "login_code": TEAM_CODE_HELP,
        "tenant_id": "请填写团队登录代号或原团队 ID（最多 64 个字符）",
        "tenant_name": "团队名称需为 1–120 个字符",
        "name": "团队名称需为 1–120 个字符",
        "email": "请填写有效的邮箱地址（最多 254 个字符）",
        "password": (
            "密码需为 6–256 个字符（原较长密码可继续登录）"
            if path == "/api/v1/auth/login" else "新密码需为 6–12 个字符"
        ),
    }
    messages = []
    for error in errors:
        location = error.get("loc", ())
        if len(location) < 2 or location[0] != "body":
            continue
        hint = hints.get(location[1])
        if hint and hint not in messages:
            messages.append(hint)
    return "；".join(messages[:3]) if messages else "提交信息格式不正确，请检查表单后重试"


def error_response(request, error):
    return JSONResponse(
        status_code=error.status,
        content={
            "error": {"code": error.code, "message": error.message, "retryable": error.retryable},
            "request_id": getattr(request.state, "request_id", "unknown"),
            "trace_id": getattr(request.state, "trace_id", "unknown"),
        },
    )


class BodyLimitMiddleware:
    """Bound total request bodies before multipart parsing, including chunked uploads."""

    def __init__(self, app, upload_limit, upload_paths=None):
        self.app, self.upload_limit = app, upload_limit
        self.upload_paths = upload_paths or {"/api/v1/assets", "/api/v1/imaging/uploads"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        limit = self.upload_limit + 2 * 1024 * 1024 if scope["path"] in self.upload_paths else 1024 * 1024
        body, length = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            length += len(data)
            if length > limit:
                response = error_response(
                    Request(scope), AppError("FILE_TOO_LARGE", "请求体超过大小限制", 413)
                )
                return await response(scope, receive, send)
            body.append(data)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    # This application exports only explicitly configured, content-free OTel spans.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if settings.app_env == "production" and (
        not settings.enable_queue
        or settings.rate_limit_per_minute == 0
        or not settings.rate_limit_fail_closed
    ):
        raise ValueError("生产模式需要 Redis 和 fail-closed 限流")

    @asynccontextmanager
    async def lifespan(app):
        provider = configure(settings)
        engine, sessions = create_database(settings.database_url)
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            follow_redirects=False,
            trust_env=False,
        )
        pool = AsyncConnectionPool(
            settings.checkpoint_url,
            min_size=1,
            max_size=8,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        dispatcher = None
        svc = SimpleNamespace(
            settings=settings,
            engine=engine,
            sessions=sessions,
            client=client,
            redis=Redis.from_url(
                settings.redis_url.get_secret_value(), socket_timeout=3, socket_connect_timeout=3
            )
            if settings.enable_queue
            else None,
        )
        app.state.services = svc
        try:
            async with engine.connect() as connection:
                revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
                if revision != "0002":
                    raise RuntimeError("请先执行 scripts/migrate.sh，当前数据库版本不匹配")
                await connection.execute(text("SELECT 1 FROM checkpoints LIMIT 1"))
            await pool.open()
            await pool.wait(timeout=20)
            svc.checkpointer = AsyncPostgresSaver(pool)
            svc.models = ModelGateway(settings, client)
            svc.graph_policy = config_load(settings.graph_config)
            if not 1 <= svc.graph_policy["max_tool_calls"] <= 10:
                raise ValueError("max_tool_calls 必须为 1–10")
            if not 0 <= svc.graph_policy["max_route_retries"] <= 3:
                raise ValueError("max_route_retries 必须为 0–3")
            if svc.graph_policy["recursion_limit"] < 4 * svc.graph_policy["max_tool_calls"] + 8:
                raise ValueError("recursion_limit 必须至少为 4*max_tool_calls+8")
            svc.storage = LocalStorage(settings.storage_root)
            svc.imaging = ImagingStore(svc.storage, settings)
            svc.segmentation = SegmentationGateway(settings)
            svc.retrieval = RetrievalService(sessions, EmbeddingGateway(settings, client))
            svc.tools = ToolGateway(ToolRegistry(settings))
            svc.queue = QueueService(svc)
            svc.graph = build_graph(svc, svc.checkpointer)
            svc.runs = RunService(svc)
            dispatcher = asyncio.create_task(svc.runs.dispatch())
            log(
                "application.started",
                graph_version=svc.graph_policy["version"],
                model_mode=settings.model_mode,
            )
            yield
        finally:
            if dispatcher:
                dispatcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await dispatcher
                await svc.runs.shutdown()
                if svc.queue.celery:
                    svc.queue.celery.close()
            if svc.redis:
                await svc.redis.aclose()
            await pool.close()
            await client.aclose()
            await engine.dispose()
            provider.force_flush(timeout_millis=1000)

    app = FastAPI(
        title="PharmaLLM Agent",
        version="1.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None,
    )
    app.add_middleware(BodyLimitMiddleware, upload_limit=settings.max_upload_bytes)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request.state.request_id = new_id("req")
        token = request_context.set({"request_id": request.state.request_id})
        started = time.monotonic()
        try:
            with span("http.request", http_method=request.method):
                request.state.trace_id = trace_id()
                try:
                    response = await call_next(request)
                except Exception as exc:
                    # Contain raw driver exceptions here: ASGI's outer error middleware would re-raise
                    # them to Uvicorn, whose traceback could include SQL parameters or request contents.
                    log("http.unexpected", error_code="INTERNAL_ERROR", exception_type=type(exc).__name__)
                    response = error_response(
                        request, AppError("INTERNAL_ERROR", "服务内部错误，请提供 trace_id 排查", 500)
                    )
                response.headers["X-Request-ID"] = request.state.request_id
                response.headers["X-Trace-ID"] = request.state.trace_id
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["Referrer-Policy"] = "no-referrer"
                response.headers["Cache-Control"] = "no-store"
                route = request.scope.get("route")
                log(
                    "http.response",
                    route=getattr(route, "path", "unmatched"),
                    status=response.status_code,
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                return response
        finally:
            request_context.reset(token)

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        log("http.error", error_code=exc.code)
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not return Pydantic's rejected `input` values (could include passwords or PHI).
        message = validation_message(request.url.path, exc.errors())
        return error_response(request, AppError("INVALID_REQUEST", message, 422))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return error_response(request, AppError("HTTP_ERROR", "请求无法处理", exc.status_code))

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        log("http.unexpected", error_code="INTERNAL_ERROR", exception_type=type(exc).__name__)
        return error_response(request, AppError("INTERNAL_ERROR", "服务内部错误，请提供 trace_id 排查", 500))

    @app.get("/health/live")
    async def live():
        return {"status": "ok", "version": "1.1.0"}

    @app.get("/health/ready")
    async def ready():
        svc = app.state.services
        try:
            async with svc.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            if svc.redis:
                await svc.redis.ping()
            await svc.models.health()
        except AppError:
            raise
        except Exception:
            raise AppError("DEPENDENCY_UNAVAILABLE", "数据库或队列依赖不可用", 503, True) from None
        return {
            "status": "ready",
            "model_mode": settings.model_mode,
            "retrieval_mode": settings.embedding_provider,
            "queue_enabled": settings.enable_queue,
        }

    app.include_router(router)
    app.include_router(imaging_router)
    app.include_router(hospital_router)
    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(
            web / "index.html",
            headers={
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                "img-src 'self' blob: data:; frame-ancestors 'none'; base-uri 'self'"
            },
        )

    return app
