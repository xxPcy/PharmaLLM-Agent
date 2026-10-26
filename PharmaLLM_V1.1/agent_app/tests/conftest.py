import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pharma_agent.main import create_app
from pharma_agent.settings import Settings


@pytest.fixture
def settings(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Integration tests require TEST_DATABASE_URL pointing to an isolated *_test database")
    if not url.split("?", 1)[0].endswith("_test"):
        pytest.fail("TEST_DATABASE_URL 必须使用独立、名称以 _test 结尾的数据库")
    return Settings(
        _env_file=None,
        database_url=url,
        jwt_secret="test-only-secret-" * 4,
        model_mode="fake",
        app_env="test",
        enable_faults=True,
        rate_limit_per_minute=0,
        enable_queue=bool(os.environ.get("TEST_REDIS_URL")),
        redis_url=os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6380/15"),
        storage_root=tmp_path / "assets",
        model_config_path=Path("configs/models.yaml"),
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as instance:
        yield instance
