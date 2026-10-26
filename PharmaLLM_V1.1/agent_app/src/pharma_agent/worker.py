"""Celery entrypoint; only the dedicated worker process imports this module."""

from pharma_agent.jobs import worker_app

app = worker_app()
