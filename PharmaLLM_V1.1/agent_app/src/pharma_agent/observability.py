import contextvars
import hashlib
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from pharma_agent.errors import AppError

request_context = contextvars.ContextVar("request_context", default={})
tracer = trace.get_tracer("pharma-agent", "1.1.0")


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def trace_id() -> str:
    return format(trace.get_current_span().get_span_context().trace_id, "032x")


def configure(settings):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s", force=True)
    for name in ("httpx", "httpcore", "sqlalchemy.engine", "uvicorn.access", "celery"):
        logging.getLogger(name).setLevel(logging.WARNING)
    provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=3)))
    # A process has one provider. Tests creating several apps reuse the first provider.
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(provider)
    return provider


def log(event: str, **fields):
    logging.getLogger("pharma_agent").info(
        json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": event,
                **request_context.get(),
                "trace_id": trace_id(),
                **fields,
            },
            ensure_ascii=False,
            default=str,
        )
    )


@contextmanager
def span(name: str, **attributes):
    start = time.monotonic()
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        log(name + ".start", **attributes)
        try:
            yield current
        except BaseException as exc:
            code = exc.code if isinstance(exc, AppError) else "INTERNAL_ERROR"
            current.set_status(Status(StatusCode.ERROR, code))
            current.set_attribute("error_code", code)
            log(
                name + ".failed",
                error_code=code,
                duration_ms=round((time.monotonic() - start) * 1000),
                **attributes,
            )
            raise
        else:
            log(name + ".completed", duration_ms=round((time.monotonic() - start) * 1000), **attributes)
